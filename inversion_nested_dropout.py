"""
inversion_nested_dropout.py — Variant of inversion.py with Nested Dropout.

Trains low-rank prompts U (77×rank) and V (rank×1024) with nested dropout:
at each iteration, a random k ∈ [min_rank, rank] is sampled, and only the
first k columns/rows of U/V participate in the forward pass. This forces
the most important information into leading dimensions, making the prompt
gracefully degradable under bandwidth-adaptive truncation.

Key optimizations over a naive implementation:
  1. Warmup: first N iterations use full rank before enabling dropout
  2. LR scaling: effective_lr = base_lr * sqrt(k / rank) — stabilizes training
  3. Auxiliary regularization on trailing dims to prevent "dimension dormancy"
  4. Loss EMA for dynamic training (robust to k-induced noise)
  5. Deterministic RNG for reproducible k sampling

Usage:
    python inversion_nested_dropout.py \
        -frame_path "data/sky" \
        -max_id 140 \
        -rank 16 \
        -min_rank 4 \
        -interval 10 \
        --nested_dropout

Output prompt format is backward-compatible with generation.py (adds
'train_rank' and 'min_rank' keys).
"""

import cv2 as cv
import torch.nn.functional as nnF
import open_clip
from scripts.demo.streamlit_helpers import *
from sgm.modules.diffusionmodules.sampling import EulerAncestralSampler
from lossbuilder import LossBuilder
from quantization import QParam, FakeQuantize
from diffusers import AutoencoderTiny
import argparse

VERSION2SPECS = {
    "SDXL-Turbo": {
        "H": 512,
        "W": 512,
        "C": 4,
        "f": 8,
        "is_legacy": False,
        "config": "configs/inference/sd_xl_base.yaml",
        "ckpt": "checkpoints/sd_xl_turbo_1.0_fp16.safetensors",
    },
    "SD-Turbo": {
        "H": 512,
        "W": 512,
        "C": 4,
        "f": 8,
        "is_legacy": False,
        "config": "configs/inference/sd_2_1.yaml",
        "ckpt": "checkpoints/sd_turbo.safetensors",
    },
}


class SubstepSampler(EulerAncestralSampler):
    def __init__(self, n_sample_steps=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_sample_steps = n_sample_steps
        self.steps_subset = [0, 100, 200, 300, 1000]

    def prepare_sampling_loop(self, x, cond, uc=None, num_steps=None):
        sigmas = self.discretization(
            self.num_steps if num_steps is None else num_steps, device=self.device
        )
        sigmas = sigmas[
            self.steps_subset[: self.n_sample_steps] + self.steps_subset[-1:]
            ]
        uc = cond
        x = x * torch.sqrt(1.0 + sigmas[0] ** 2.0)
        num_sigmas = len(sigmas)
        s_in = x.new_ones([x.shape[0]])
        return x, s_in, sigmas, num_sigmas, cond, uc


def seeded_randn(shape, seed):
    randn = np.random.RandomState(seed).randn(*shape)
    randn = torch.from_numpy(randn).to(device="cuda", dtype=torch.float32)
    return randn


class SeededNoise:
    def __init__(self, seed):
        self.seed = seed

    def __call__(self, x):
        self.seed = self.seed + 1
        return seeded_randn(x.shape, self.seed)


def slerp(a, b, t, eps=1e-5):
    """球面线性插值, 沿最后一维计算.

    Args:
        a, b: 同形状张量 [..., D].
        t: 标量或广播到 [...] 的张量.
        eps: θ 阈值, 低于此值时退化为线性 Lerp.

    Returns:
        插值结果, 与 a, b 同形状.
    """
    a_n = a / (a.norm(dim=-1, keepdim=True) + 1e-12)
    b_n = b / (b.norm(dim=-1, keepdim=True) + 1e-12)
    cos_theta = (a_n * b_n).sum(dim=-1, keepdim=True).clamp(-1.0, 1.0)
    theta = torch.acos(cos_theta)
    sin_theta = torch.sin(theta)
    use_lerp = sin_theta < eps
    factor_a = torch.sin((1.0 - t) * theta) / sin_theta
    factor_b = torch.sin(t * theta) / sin_theta
    result = factor_a * a + factor_b * b
    lerp_result = (1.0 - t) * a + t * b
    result = torch.where(use_lerp.expand_as(result), lerp_result, result)
    return result


def clip_preprocess(img_tensor):
    """Convert [-1,1] image tensor to CLIP input format [0,1] + normalize.
       img_tensor: [1, C, H, W] in [-1, 1], differentiable."""
    img_tensor = (img_tensor + 1.0) / 2.0  # [-1, 1] -> [0, 1]
    img_tensor = nnF.interpolate(img_tensor, size=(224, 224), mode='bilinear', align_corners=False)
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=img_tensor.device).view(1, 3, 1, 1)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=img_tensor.device).view(1, 3, 1, 1)
    img_tensor = (img_tensor - mean) / std
    return img_tensor


def color_stats_loss(pred, gt):
    """Color statistics loss: match per-channel mean and std.
       pred, gt: [1, 3, H, W] in [-1, 1], differentiable."""
    pred_mean = pred.mean(dim=[2, 3])  # [1, 3]
    pred_std = pred.std(dim=[2, 3])    # [1, 3]
    gt_mean = gt.mean(dim=[2, 3])
    gt_std = gt.std(dim=[2, 3])
    return nnF.mse_loss(pred_mean, gt_mean) + nnF.mse_loss(pred_std, gt_std)


def inversion(
        model,
        sampler,
        decoder,
        rank,
        min_rank,
        interval,
        frame_path,
        max_id,
        H=512,
        W=512,
        seed=0,
        filter=None,
        clip_model=None,
        clip_weight=0.0,
        temp_weight=0.0,
        color_weight=0.0,
        baseline=False,
        slerp_mode=True,
        nested_dropout=True,
        dropout_schedule=False,
        dropout_warmup_first=500,
        dropout_warmup_sub=100,
        reg_weight=0.001,
):
    """Run inversion with nested dropout and associated optimizations.

    Args:
        rank: Full training rank (e.g. 16).
        min_rank: Minimum rank to retain during nested dropout (e.g. 4).
        nested_dropout: If True, enable nested dropout during training.
        dropout_schedule: If True, bias sampling toward lower ranks early.
        dropout_warmup_first: Iterations of full-rank warmup for first frame.
        dropout_warmup_sub: Iterations of full-rank warmup for subsequent frames.
        reg_weight: Weight for trailing-dimension regularization (prevents dormancy).
    """
    F = 8
    C = 4
    shape = (1, C, H // F, W // F)

    if seed is None:
        seed = torch.seed()
    # Deterministic RNG for reproducible k sampling across runs
    rng = np.random.RandomState(seed)

    precision_scope = autocast
    with precision_scope("cuda"):
        def denoiser(input, sigma, c):
            return model.denoiser(
                model.model,
                input,
                sigma,
                c,
            )

        def load_img(path):
            img = cv.imread(path)
            img = img[:, :, ::-1]
            H, W, C = img.shape
            l, r = int(W / 2 - H / 2), int(W / 2 + H / 2)
            img = img[:, l:r, :]
            img = cv.resize(img, [512, 512])
            img = (img / 255) * 2 - 1
            img = torch.from_numpy(img)
            img = img.float()
            img = img.permute(2, 0, 1)
            img = img.unsqueeze(dim=0)
            img = img.cuda()
            return img

        uc = None
        rand_noise = seeded_randn(shape, seed)
        sigma = torch.Tensor([0.05]).float().cuda()

        # Set the loss functions.
        # Reconstruction loss.
        mse_loss = torch.nn.MSELoss()
        # Perceptual loss.
        builder = LossBuilder('cuda')
        content_layers = [('conv_1', 1), ('conv_2', 1), ('conv_3', 1), ('conv_4', 1),
                          ('conv_5', 1), ('conv_6', 1), ('conv_7', 1), ('conv_8', 1),
                          ('conv_9', 1), ('conv_10', 1), ('conv_11', 1), ('conv_12', 1),
                          ('conv_13', 1), ('conv_14', 1), ('conv_15', 1),
                          ('conv_16', 1)]
        vgg_model, lpips_nodes = builder.get_style_and_content_loss(dict(content_layers))

        # Inversion.
        suffix = '_baseline' if baseline else ''
        suffix = suffix + args.suffix
        prompt_path = os.path.join('/root/autodl-tmp/sky', 'results/rank{}_interval{}{}/'.format(rank, interval, suffix))

        for f_id in range(0, max_id, interval):
            # Initialize the low-rank factor U (always full rank).
            U = torch.rand([77, rank]).float().cuda()
            U.requires_grad = True
            # Fake Quantizer for U
            Quant_Param_U = QParam(num_bits=8)
            # Initialize the low-rank factor V (always full rank).
            V = torch.rand([rank, 1024]).float().cuda()
            V.requires_grad = True
            # Fake Quantizer for V
            Quant_Param_V = QParam(num_bits=8)

            # Base learning rate (will be scaled by sqrt(k/rank) every iter).
            base_lr = 0.1
            optimizer = torch.optim.Adam([U, V], lr=base_lr)

            # for learning rate scheduler and logging
            min_loss = 1e9
            latest_min_loss = 1e9

            # logs and results path
            log_path = os.path.join(prompt_path,'{:05d}'.format(f_id))
            if not os.path.exists(log_path):
                os.makedirs(log_path)
            log_output = open(os.path.join(log_path,'log.txt'), 'a')

            if f_id > 0:
                ckpt_prev = torch.load(os.path.join(prompt_path,'{:05d}/ckpt.pth'.format(f_id - interval)))
                U_prev, V_prev = ckpt_prev["U"], ckpt_prev["V"]
                prev_frame = ckpt_prev["z"]
                # Subsequent frames require fewer iterations.
                total_iterations = 1500
                lr_schedule_cnt = 20
                step_list_base = [_ for _ in range(1, interval + 1)]
                warmup_iters = dropout_warmup_sub
            else:
                # Initialization of the first frame.
                prev_frame = model.encode_first_stage(load_img(os.path.join(frame_path,'00000.png')))
                torch.save(prev_frame, os.path.join(prompt_path, 'init.pth'))
                # The first frame requires more iterations.
                total_iterations = 10000
                lr_schedule_cnt = 300
                step_list_base = [0]
                warmup_iters = dropout_warmup_first
            # add random noise to the previous frame
            randn = (prev_frame * sigma + rand_noise * (1 - sigma)).detach()
            step_list = step_list_base

            # Loss EMA dictionary for noise-robust dynamic training
            loss_ema = {}

            for iter in range(total_iterations):
                # === 1) Warmup: use full rank during early iterations ===
                in_warmup = (iter < warmup_iters)

                # === 2) Nested Dropout: sample active rank k ===
                if nested_dropout and not in_warmup and rank > min_rank:
                    if dropout_schedule:
                        progress = (iter - warmup_iters) / max(1, total_iterations - warmup_iters)
                        # p_low decays from 0.7 to 0.0 as progress → 1.0
                        p_low = max(0.0, 0.7 * (1.0 - progress))
                        if rng.random() < p_low:
                            k = int(rng.randint(min_rank, (min_rank + rank) // 2 + 1))
                        else:
                            k = int(rng.randint(min_rank, rank + 1))
                    else:
                        # Uniform sampling from [min_rank, rank] (inclusive)
                        k = int(rng.randint(min_rank, rank + 1))
                else:
                    k = rank  # No dropout or in warmup → full rank

                # === 3) LR scaling by sqrt(k / rank) ===
                # Smaller k → smaller effective_lr, since fewer dimensions carry the gradient
                effective_lr = base_lr * np.sqrt(float(k) / float(rank))
                optimizer.param_groups[0]['lr'] = effective_lr

                loss_list = {}
                clip_embed_buffer = []  # buffer for temporal direction consistency
                for step in step_list:
                    # Fake Quantification (always on full U, V, so quant params track all dims)
                    Quant_Param_U.update(U)
                    Q_U = FakeQuantize.apply(U, Quant_Param_U)
                    Quant_Param_V.update(V)
                    Q_V = FakeQuantize.apply(V, Quant_Param_V)

                    # === Truncate to active rank k ===
                    Q_U_trunc = Q_U[:, :k]      # [77, k]
                    Q_V_trunc = Q_V[:k, :]      # [k, 1024]

                    if f_id > 0:
                        # Also truncate previous frame's prompt
                        U_prev_trunc = U_prev[:, :k]
                        V_prev_trunc = V_prev[:k, :]

                        # perform interpolation on keyframe prompts
                        # approximating the intermediate prompts.
                        t = step / interval
                        if slerp_mode:
                            u = slerp(U_prev_trunc.T, Q_U_trunc.T, t).T
                            v = slerp(V_prev_trunc, Q_V_trunc, t)
                        else:
                            u = (1 - t) * U_prev_trunc + t * Q_U_trunc
                            v = (1 - t) * V_prev_trunc + t * Q_V_trunc
                        # prompt composition — scale by sqrt(k) (actual active rank)
                        c = (u @ v / np.sqrt(k)).unsqueeze(dim=0)
                        cur_id = f_id - interval + step
                    else:
                        # for the first frame
                        c = (Q_U_trunc @ Q_V_trunc / np.sqrt(k)).unsqueeze(dim=0)
                        cur_id = 0
                    c = {'crossattn': c}

                    # generating a frame
                    samples_z = sampler(denoiser, randn, cond=c, uc=uc)
                    samples_x = decoder(samples_z)

                    # Calculating loss
                    gt = load_img(os.path.join(frame_path, '{:05d}.png'.format(cur_id)))
                    gt.requires_grad = True
                    # Perceptual loss.
                    vgg_model(torch.cat([gt, samples_x], dim=0))
                    lpips_loss = 0
                    for node in lpips_nodes:
                        lpips_loss += node.loss
                    lpips_loss = lpips_loss / (len(lpips_nodes) + 1e-9)
                    # Combine the perceptual loss and reconstruction loss.
                    loss = 0.2 * lpips_loss + 0.8 * mse_loss(samples_x, gt)

                    # --- CLIP semantic loss ---
                    clip_loss_val = 0.0
                    if clip_model is not None and clip_weight > 0:
                        pred_clip = clip_preprocess(samples_x)
                        pred_embed = nnF.normalize(clip_model.encode_image(pred_clip), dim=-1)
                        gt_clip = clip_preprocess(gt)
                        gt_embed = nnF.normalize(clip_model.encode_image(gt_clip), dim=-1)
                        clip_loss_val = (1.0 - (gt_embed * pred_embed).sum(dim=-1)).mean()
                        loss = loss + clip_weight * clip_loss_val

                    # --- Temporal direction consistency loss ---
                    temp_loss_val = 0.0
                    if clip_model is not None and temp_weight > 0 and f_id > 0:
                        pred_clip = clip_preprocess(samples_x)
                        pred_embed = nnF.normalize(clip_model.encode_image(pred_clip), dim=-1)
                        clip_embed_buffer.append((cur_id, pred_embed))
                        if len(clip_embed_buffer) >= 3:
                            id_a, emb_a = clip_embed_buffer[-3]
                            id_b, emb_b = clip_embed_buffer[-2]
                            id_c, emb_c = clip_embed_buffer[-1]
                            if id_b == id_a + 1 and id_c == id_b + 1:
                                dir_ab = nnF.normalize(emb_b.detach() - emb_a.detach(), dim=-1)
                                dir_bc = nnF.normalize(emb_c - emb_b.detach(), dim=-1)
                                temp_loss_val = (1.0 - (dir_ab * dir_bc).sum(dim=-1)).mean()
                                loss = loss + temp_weight * temp_loss_val

                    # --- Color statistics loss ---
                    color_loss_val = 0.0
                    if color_weight > 0:
                        color_loss_val = color_stats_loss(samples_x, gt)
                        loss = loss + color_weight * color_loss_val

                    # regularization
                    loss_regu = torch.mean(torch.abs(c['crossattn']))
                    loss = loss + 0.1 * loss_regu

                    # === 4) Auxiliary regularization on trailing dims ===
                    # Prevents "dimension dormancy" — ensures later dimensions remain active
                    # even when they are not trained in the current iteration.
                    # Using std (not norm) so we don't force them to zero — they need to
                    # preserve some signal for when high-rank recovery is needed.
                    if nested_dropout and not in_warmup and reg_weight > 0 and k < rank:
                        # Penalize the trailing dims (from min_rank onward) to avoid explosion
                        loss_aux = reg_weight * (
                            Q_U[:, min_rank:].std() + Q_V[min_rank:, :].std()
                        )
                        loss = loss + loss_aux
                    else:
                        loss_aux = 0.0

                    # logging
                    print('iter: {}, cur_id: {}, k: {}, lr: {:.6f}, loss: {:.4f}, aux: {:.6f}, clip_loss: {:.4f}, temp_loss: {:.4f}, color_loss: {:.4f}, c_max: {}, c_mean: {}, c_std: {}'.format(
                        iter, cur_id, k, effective_lr, loss, loss_aux if isinstance(loss_aux, float) else loss_aux.item(),
                        clip_loss_val, temp_loss_val, color_loss_val,
                        c['crossattn'].max(), c['crossattn'].mean(), c['crossattn'].std()))
                    log_output.write('iter: {}, cur_id: {}, k: {}, lr: {:.6f}, loss: {:.4f}, aux: {:.6f}, clip_loss: {:.4f}, temp_loss: {:.4f}, color_loss: {:.4f}, c_max: {}, c_mean: {}, c_std: {}\n'.format(
                        iter, cur_id, k, effective_lr, loss, loss_aux if isinstance(loss_aux, float) else loss_aux.item(),
                        clip_loss_val, temp_loss_val, color_loss_val,
                        c['crossattn'].max(), c['crossattn'].mean(), c['crossattn'].std()))
                    log_output.flush()

                    # saving the generated frames
                    if iter % 10 == 0:
                        img = torch.clamp((samples_x + 1.0) / 2.0, min=0.0, max=1.0)
                        img = (
                            (255 * img)
                                .to(dtype=torch.uint8)
                                .permute(0, 2, 3, 1)
                                .detach()
                                .cpu()
                                .numpy()
                        )
                        img = img[0][:, :, ::-1]
                        cv2.imwrite(os.path.join(log_path,'iter_{:05d}_id_{:05d}.png'.format(iter, cur_id)), img)

                    # Optimization
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    model.model.zero_grad()

                    # for learning rate scheduler and logging
                    if cur_id not in loss_list.keys():
                        loss_list[cur_id] = loss.item()
                mean_loss = np.mean(list(loss_list.values()))
                print('iter: {}, mean loss: {}, lr: {:.6f}'.format(iter, mean_loss, effective_lr))
                log_output.write('iter: {}, mean loss: {}, lr: {:.6f}\n'.format(iter, mean_loss, effective_lr))
                log_output.flush()
                if mean_loss < min_loss:
                    # saving the ckpt (store full-rank Q_U, Q_V)
                    min_loss = mean_loss
                    ckpt = {
                        'U': Q_U,                     # full rank [77, rank]
                        'U_scale': Quant_Param_U.scale,
                        'U_zero_point': Quant_Param_U.zero_point,
                        'U_bits': Quant_Param_U.num_bits,
                        'V': Q_V,                     # full rank [rank, 1024]
                        'V_scale': Quant_Param_V.scale,
                        'V_zero_point': Quant_Param_V.zero_point,
                        'V_bits': Quant_Param_V.num_bits,
                        'z': samples_z,
                        'randn': randn,
                        'iter': iter,
                        'loss': mean_loss,
                        'train_rank': rank,
                        'min_rank': min_rank,
                    }
                    torch.save(ckpt, os.path.join(log_path, 'ckpt.pth'))
                    # saving the prompt (full rank)
                    U_Byte = Quant_Param_U.quantize_tensor(Q_U).byte()
                    V_Byte = Quant_Param_V.quantize_tensor(Q_V).byte()
                    prompt = {
                        'U': U_Byte,
                        'V': V_Byte,
                        'U_scale': Quant_Param_U.scale,
                        'U_zero_point': Quant_Param_U.zero_point,
                        'V_scale': Quant_Param_V.scale,
                        'V_zero_point': Quant_Param_V.zero_point,
                        'train_rank': rank,      # ← metadata for generation
                        'min_rank': min_rank,     # ← metadata for generation
                    }
                    torch.save(prompt, os.path.join(prompt_path, 'frame_{:05d}.prompt'.format(f_id)))

                if f_id > 0:
                    # === 5) Dynamic training with Loss EMA (robust to k-induced noise) ===
                    # Update exponential moving average of loss per step
                    for sid, val in loss_list.items():
                        alpha = 0.7  # higher = smoother
                        if sid in loss_ema:
                            loss_ema[sid] = alpha * loss_ema[sid] + (1.0 - alpha) * val
                        else:
                            loss_ema[sid] = val
                    # Select the worst-performing step using EMA (not raw loss)
                    worst_step = max(loss_ema, key=loss_ema.get) - f_id + interval
                    step_list = step_list_base + [worst_step] * 2
                    step_list = sorted(step_list)

                # Learning rate scheduler (applies to base_lr, effective_lr will follow)
                lr_schedule_cnt = lr_schedule_cnt - 1
                if lr_schedule_cnt == 0:
                    if min_loss == latest_min_loss:
                        # Reduce the learning rate by half.
                        base_lr = max(base_lr * 0.5, 0.001)
                        print('reduce lr to: {}'.format(base_lr))
                        log_output.write('reduce lr to: {}\n'.format(base_lr))
                        log_output.flush()
                    latest_min_loss = min_loss
                    lr_schedule_cnt = 20 if f_id > 0 else 300
            log_output.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-frame_path', type=str, default="data/sky")
    parser.add_argument('-max_id', type=int, default=140)
    parser.add_argument('-rank', type=int, default=16,
                        help='Full training rank (higher than final deploy rank)')
    parser.add_argument('-min_rank', type=int, default=4,
                        help='Minimum rank preserved during nested dropout')
    parser.add_argument('-interval', type=int, default="10")
    parser.add_argument('-clip_weight', type=float, default=0.5,
                        help='Weight for CLIP semantic loss (0.0 to disable)')
    parser.add_argument('-temp_weight', type=float, default=0.1,
                        help='Weight for temporal direction consistency loss (0.0 to disable)')
    parser.add_argument('-color_weight', type=float, default=0.3,
                        help='Weight for color statistics loss (match per-channel mean/std, 0.0 to disable)')
    parser.add_argument('-suffix', type=str, default='',
                        help='Optional suffix appended to the results directory name (e.g. "_ndrop")')
    parser.add_argument('--baseline', action='store_true',
                        help='Use only original losses (LPIPS + MSE + reg), no CLIP, no slerp')
    parser.add_argument('--slerp', action='store_true', default=True,
                        help='Use spherical linear interpolation (default: True)')
    parser.add_argument('--no-slerp', action='store_false', dest='slerp',
                        help='Use linear interpolation instead of slerp')
    parser.add_argument('--nested_dropout', action='store_true', default=True,
                        help='Enable nested dropout training (default: True)')
    parser.add_argument('--no-nested_dropout', action='store_false', dest='nested_dropout',
                        help='Disable nested dropout (train full rank only)')
    parser.add_argument('--dropout_schedule', action='store_true', default=False,
                        help='Use scheduled (biased toward low rank early) instead of uniform dropout')
    parser.add_argument('--dropout_warmup_first', type=int, default=500,
                        help='Full-rank warmup iterations for first frame (default: 500)')
    parser.add_argument('--dropout_warmup_sub', type=int, default=100,
                        help='Full-rank warmup iterations for subsequent frames (default: 100)')
    parser.add_argument('--reg_weight', type=float, default=0.001,
                        help='Weight for trailing-dim regularization (default: 0.001, set 0 to disable)')
    args = parser.parse_args()

    if args.baseline:
        print("Baseline mode: disabling CLIP semantic, temporal, color losses, and slerp.")
        args.clip_weight = 0.0
        args.temp_weight = 0.0
        args.color_weight = 0.0
        args.slerp = False

    if args.nested_dropout:
        print(f"Nested Dropout enabled: rank={args.rank}, min_rank={args.min_rank}, "
              f"warmup_first={args.dropout_warmup_first}, warmup_sub={args.dropout_warmup_sub}, "
              f"reg_weight={args.reg_weight}")
        if args.dropout_schedule:
            print("  Using scheduled dropout (biased toward low rank early in training).")
        else:
            print("  Using uniform dropout across ranks.")
    else:
        print(f"Nested Dropout disabled: training full rank={args.rank} only.")

    # Set up and load the models.
    version_dict = VERSION2SPECS['SD-Turbo']
    state = init_st(version_dict, load_filter=True)
    if state["msg"]:
        st.info(state["msg"])
    model = state["model"]
    load_model(model)
    taesd = AutoencoderTiny.from_pretrained("madebyollin/taesd", torch_dtype=torch.float32).cuda()
    sampler = SubstepSampler(
        n_sample_steps=1,
        num_steps=1000,
        eta=1.0,
        discretization_config=dict(
            target="sgm.modules.diffusionmodules.discretizer.LegacyDDPMDiscretization"
        ),
    )
    seed_ = 88
    sampler.noise_sampler = SeededNoise(seed=seed_)

    # Load CLIP model for semantic loss if enabled
    clip_model = None
    if args.clip_weight > 0 or args.temp_weight > 0:
        print("Loading CLIP model (ViT-B/32) for semantic loss...")
        clip_model, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k")
        clip_model = clip_model.cuda().eval()
        for p in clip_model.parameters():
            p.requires_grad_(False)
        print("CLIP model loaded.")

    # Inversion: from video to prompts
    inversion(
       model, sampler, decoder=taesd.decoder, rank=args.rank, min_rank=args.min_rank,
       interval=args.interval, frame_path=args.frame_path, max_id=args.max_id,
       H=512, W=512, seed=seed_, filter=state.get("filter"),
       clip_model=clip_model, clip_weight=args.clip_weight,
       temp_weight=args.temp_weight, color_weight=args.color_weight,
       baseline=args.baseline, slerp_mode=args.slerp,
       nested_dropout=args.nested_dropout, dropout_schedule=args.dropout_schedule,
       dropout_warmup_first=args.dropout_warmup_first,
       dropout_warmup_sub=args.dropout_warmup_sub,
       reg_weight=args.reg_weight,
    )