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
        t: 标量或广播到 [...] 的张量. 逐向量不同速率时 t 形状应匹配 a.shape[:-1].
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


def compute_rank_weights(all_U_list):
    """从所有关键帧 U 矩阵计算逐秩全局活跃度权重.

    将所有关键帧 U 堆叠, 计算每列标准差作为时间活跃度, 然后归一化.

    Args:
        all_U_list: 张量列表, 每个形状 [77, rank], 每关键帧一个.

    Returns:
        weights: 张量 [rank], 归一化后的逐秩活跃度, 范围 (0, 1].
    """
    stacked = torch.stack(all_U_list, dim=0)         # [K, 77, rank]
    flat = stacked.reshape(-1, stacked.shape[-1])     # [K*77, rank]
    std = flat.std(dim=0)                             # [rank]
    w = std / (std.max() + 1e-8)                      # 归一化到 (0, 1]
    w = w.clamp(min=1e-6)                             # 防零
    return w


def inversion(
        model,
        sampler,
        decoder,
        rank,
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
        slerp_mode=True
):
    F = 8
    C = 4
    shape = (1, C, H // F, W // F)

    if seed is None:
        seed = torch.seed()
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
        all_U_list = []  # 收集所有关键帧 U 用于全局活跃度计算
        # ===== 路径设置 (提到循环外, Phase 2 复用) =====
        suffix = '_baseline' if baseline else ''
        suffix = suffix + args.suffix
        prompt_path = os.path.join('/root/autodl-tmp/sky', 'results/rank{}_interval{}{}/'.format(rank, interval, suffix))

        # ===== Phase 1: 所有关键帧独立训练 (不做逐帧插值) =====
        for f_id in range(0, max_id, interval):
            # Initialize the low-rank factor U.
            U = torch.rand([77, rank]).float().cuda()
            U.requires_grad = True
            # Fake Quantizer for U
            Quant_Param_U = QParam(num_bits=8)
            # Initialize the low-rank factor V.
            V = torch.rand([rank, 1024]).float().cuda()
            V.requires_grad = True
            # Fake Quantizer for V
            Quant_Param_V = QParam(num_bits=8)

            # Initialize the learning rate and the optimizer.
            lr = 0.1
            optimizer = torch.optim.Adam([U, V], lr=lr)

            # for learning rate scheduler and logging
            min_loss = 1e9
            latest_min_loss = 1e9

            # logs path per keyframe
            log_path = os.path.join(prompt_path,'{:05d}'.format(f_id))
            if not os.path.exists(log_path):
                os.makedirs(log_path)
            log_output = open(os.path.join(log_path,'log.txt'), 'a')

            if f_id > 0:
                ckpt_prev = torch.load(os.path.join(prompt_path,'{:05d}/ckpt.pth'.format(f_id - interval)), weights_only=False)
                U_prev, V_prev = ckpt_prev["U"], ckpt_prev["V"]
                prev_frame = ckpt_prev["z"]
                # Subsequent frames require fewer iterations.
                # Reduce total_iterations to speed up inversion, but this may lower quality.
                total_iterations = 1500
                lr_schedule_cnt = 20
                step_list_base = [interval]  # Phase 1: 仅关键帧端点, t=1.0 退化为恒等
            else:
                # Initialization of the first frame.
                prev_frame = model.encode_first_stage(load_img(os.path.join(frame_path,'00000.png')))
                torch.save(prev_frame, os.path.join(prompt_path, 'init.pth'))
                # The first frame requires more iterations.
                total_iterations = 10000
                lr_schedule_cnt = 300
                step_list_base = [0]
            # add random noise to the previous frame
            randn = (prev_frame * sigma + rand_noise * (1 - sigma)).detach()
            step_list = step_list_base

            for iter in range(total_iterations):
                loss_list = {}
                clip_embed_buffer = []  # buffer for temporal direction consistency
                for step in step_list:
                    # Fake Quantification
                    Quant_Param_U.update(U)
                    Q_U = FakeQuantize.apply(U, Quant_Param_U)
                    Quant_Param_V.update(V)
                    Q_V = FakeQuantize.apply(V, Quant_Param_V)

                    if f_id > 0:
                        t = step / interval  # 归一化插值进度 [0, 1]
                        if slerp_mode:
                            # 球面线性插值: Phase 1 用统一速率 (t 为标量)
                            u = slerp(U_prev.T, Q_U.T, t).T
                            v = slerp(V_prev, Q_V, t)
                        else:
                            # 线性插值 (原始行为)
                            u = (1 - t) * U_prev + t * Q_U
                            v = (1 - t) * V_prev + t * Q_V
                        # prompt composition
                        c = (u @ v / np.sqrt(rank)).unsqueeze(dim=0)
                        cur_id = f_id - interval + step
                    else:
                        # for the first frame
                        c = (Q_U @ Q_V / np.sqrt(rank)).unsqueeze(dim=0)
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

                    # CLIP semantic loss + temporal consistency (if enabled)
                    clip_loss_val = 0.0
                    temp_loss_val = 0.0
                    color_loss_val = 0.0
                    if clip_model is not None and (clip_weight > 0 or (temp_weight > 0 and f_id > 0)):
                        pred_clip = clip_preprocess(samples_x)
                        pred_embed = nnF.normalize(clip_model.encode_image(pred_clip), dim=-1)

                        # CLIP semantic loss (align with ground truth)
                        if clip_weight > 0:
                            gt_clip = clip_preprocess(gt)
                            gt_embed = nnF.normalize(clip_model.encode_image(gt_clip), dim=-1)
                            clip_loss_val = (1.0 - (gt_embed * pred_embed).sum(dim=-1)).mean()
                            loss = loss + clip_weight * clip_loss_val

                        # Temporal direction consistency loss
                        # Encourages frame-to-frame CLIP embedding changes to be
                        # directionally consistent (smooth semantic transitions).
                        if temp_weight > 0 and f_id > 0:
                            clip_embed_buffer.append((cur_id, pred_embed))
                            if len(clip_embed_buffer) >= 3:
                                id_a, emb_a = clip_embed_buffer[-3]
                                id_b, emb_b = clip_embed_buffer[-2]
                                id_c, emb_c = clip_embed_buffer[-1]
                                # Only compute if cur_ids are strictly consecutive
                                if id_b == id_a + 1 and id_c == id_b + 1:
                                    dir_ab = nnF.normalize(emb_b.detach() - emb_a.detach(), dim=-1)
                                    dir_bc = nnF.normalize(emb_c - emb_b.detach(), dim=-1)
                                    temp_loss_val = (1.0 - (dir_ab * dir_bc).sum(dim=-1)).mean()
                                    loss = loss + temp_weight * temp_loss_val

                    # Color statistics loss (match per-channel mean and std)
                    if color_weight > 0:
                        color_loss_val = color_stats_loss(samples_x, gt)
                        loss = loss + color_weight * color_loss_val

                    # regularization
                    loss_regu = torch.mean(torch.abs(c['crossattn']))
                    loss = loss + 0.1 * loss_regu

                    # logging
                    print('iter: {}, cur_id: {}, loss: {}, clip_loss: {:.4f}, temp_loss: {:.4f}, color_loss: {:.4f}, c_max: {}, c_mean: {}, c_std: {}'.format(iter, cur_id, loss, clip_loss_val, temp_loss_val, color_loss_val, c['crossattn'].max(), c['crossattn'].mean(), c['crossattn'].std()))
                    log_output.write('iter: {}, cur_id: {}, loss: {}, clip_loss: {:.4f}, temp_loss: {:.4f}, color_loss: {:.4f}, c_max: {}, c_mean: {}, c_std: {}\n'.format(iter, cur_id, loss, clip_loss_val, temp_loss_val, color_loss_val, c['crossattn'].max(), c['crossattn'].mean(), c['crossattn'].std()))
                    log_output.flush()

                    # saving the generated frames
                    if iter % 50 == 0:
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
                print('iter: {}, mean loss: {}'.format(iter, mean_loss))
                log_output.write('iter: {}, mean loss: {}\n'.format(iter, mean_loss))
                log_output.flush()
                if mean_loss < min_loss:
                    # saving the ckpt
                    min_loss = mean_loss
                    ckpt = {
                        'U': Q_U,
                        'U_scale': Quant_Param_U.scale,
                        'U_zero_point': Quant_Param_U.zero_point,
                        'U_bits': Quant_Param_U.num_bits,
                        'V': Q_V,
                        'V_scale': Quant_Param_V.scale,
                        'V_zero_point': Quant_Param_V.zero_point,
                        'V_bits': Quant_Param_V.num_bits,
                        'z': samples_z,
                        'randn': randn,
                        'iter': iter,
                        'loss': mean_loss,
                    }
                    torch.save(ckpt, os.path.join(log_path, 'ckpt.pth'))
                    # saving the prompt
                    U_Byte = Quant_Param_U.quantize_tensor(Q_U).byte()
                    V_Byte = Quant_Param_V.quantize_tensor(Q_V).byte()
                    prompt = {
                        'U': U_Byte,
                        'V': V_Byte,
                        'U_scale': Quant_Param_U.scale,
                        'U_zero_point': Quant_Param_U.zero_point,
                        'V_scale': Quant_Param_V.scale,
                        'V_zero_point': Quant_Param_V.zero_point,
                    }
                    torch.save(prompt, os.path.join(prompt_path, 'frame_{:05d}.prompt'.format(f_id)))
                    if not baseline:
                        all_U_list.append(Q_U.detach().cpu())  # 收集 U 用于全局活跃度
                if f_id > 0:
                    # Dynamic training.
                    # Allocating more training to the frames with the worst performance.
                    worst_step = max(loss_list, key=loss_list.get) - f_id + interval
                    step_list = step_list_base + [worst_step] * 2
                    step_list = sorted(step_list)
                # Learning rate scheduler
                lr_schedule_cnt = lr_schedule_cnt - 1
                if lr_schedule_cnt == 0:
                    if min_loss == latest_min_loss:
                        # Reduce the learning rate by half.
                        lr = max(lr * 0.5, 0.001)
                        optimizer = torch.optim.Adam([U, V], lr=lr)
                        print('reduce lr to: {}'.format(optimizer.param_groups[0]['lr']))
                        log_output.write('reduce lr to: {}\n'.format(optimizer.param_groups[0]['lr']))
                        log_output.flush()
                    latest_min_loss = min_loss
                    lr_schedule_cnt = 20 if f_id > 0 else 300
            log_output.close()

        # --- 主循环结束后: 计算全局逐秩活跃度权重并写入 prompt 文件 ---
        if not baseline and len(all_U_list) > 1:
            weights = compute_rank_weights(all_U_list)  # [rank], CPU
            print('Per-rank activity weights: min={:.4f}, max={:.4f}, mean={:.4f}'.format(
                weights.min().item(), weights.max().item(), weights.mean().item()))
            weights_cuda = weights.cuda()

            # ===== Phase 2: 逐秩权重微调中间帧 =====
            print('=== Phase 2: Fine-tuning intermediate frames with per-rank weights ===')
            for f_id in range(interval, max_id, interval):
                # 加载前一个关键帧的 ckpt (固定, 无梯度)
                ckpt_prev = torch.load(
                    os.path.join(prompt_path, '{:05d}/ckpt.pth'.format(f_id - interval)),
                    weights_only=False)
                U_prev = ckpt_prev["U"].detach()
                V_prev = ckpt_prev["V"].detach()
                prev_frame = ckpt_prev["z"]

                # 从 Phase 1 ckpt 初始化当前关键帧的 U, V (fine-tunable)
                ckpt_curr = torch.load(
                    os.path.join(prompt_path, '{:05d}/ckpt.pth'.format(f_id)),
                    weights_only=False)
                U = ckpt_curr["U"].detach().clone().cuda().requires_grad_(True)
                V = ckpt_curr["V"].detach().clone().cuda().requires_grad_(True)

                # 初始化量化参数
                Quant_Param_U = QParam(num_bits=8)
                Quant_Param_U.scale = ckpt_curr["U_scale"]
                Quant_Param_U.zero_point = ckpt_curr["U_zero_point"]
                Quant_Param_V = QParam(num_bits=8)
                Quant_Param_V.scale = ckpt_curr["V_scale"]
                Quant_Param_V.zero_point = ckpt_curr["V_zero_point"]

                lr = 0.05  # 微调学习率 (Phase 1 的一半)
                optimizer = torch.optim.Adam([U, V], lr=lr)
                min_loss = 1e9
                latest_min_loss = 1e9

                log_path = os.path.join(prompt_path, '{:05d}'.format(f_id))
                log_output = open(os.path.join(log_path, 'log_phase2.txt'), 'a')

                step_list_base = [_ for _ in range(1, interval + 1)]  # [1, 2, ..., interval] 含关键帧
                step_list = step_list_base
                total_iterations_p2 = 1500
                lr_schedule_cnt = 10

                randn = (prev_frame * sigma + rand_noise * (1 - sigma)).detach()

                for iter in range(total_iterations_p2):
                    loss_list = {}
                    clip_embed_buffer = []  # buffer for temporal direction consistency
                    for step in step_list:
                        Quant_Param_U.update(U)
                        Q_U = FakeQuantize.apply(U, Quant_Param_U)
                        Quant_Param_V.update(V)
                        Q_V = FakeQuantize.apply(V, Quant_Param_V)

                        t = step / interval
                        t_i = t ** (1.0 / weights_cuda)  # [rank], 逐秩差异化

                        if slerp_mode:
                            u = slerp(U_prev.T, Q_U.T, t_i.view(-1, 1)).T
                            v = slerp(V_prev, Q_V, t_i.view(-1, 1))
                        else:
                            u = (1 - t_i.view(1, -1)) * U_prev + t_i.view(1, -1) * Q_U
                            v = (1 - t_i.view(-1, 1)) * V_prev + t_i.view(-1, 1) * Q_V

                        c = (u @ v / np.sqrt(rank)).unsqueeze(dim=0)
                        cur_id = f_id - interval + step
                        c = {'crossattn': c}

                        samples_z = sampler(denoiser, randn, cond=c, uc=uc)
                        samples_x = decoder(samples_z)

                        gt = load_img(os.path.join(frame_path, '{:05d}.png'.format(cur_id)))
                        gt.requires_grad = True
                        vgg_model(torch.cat([gt, samples_x], dim=0))
                        lpips_loss = 0
                        for node in lpips_nodes:
                            lpips_loss += node.loss
                        lpips_loss = lpips_loss / (len(lpips_nodes) + 1e-9)
                        loss = 0.2 * lpips_loss + 0.8 * mse_loss(samples_x, gt)

                        # CLIP + temporal + color (if enabled, same as Phase 1)
                        clip_loss_val = 0.0
                        temp_loss_val = 0.0
                        color_loss_val = 0.0
                        if clip_model is not None and (clip_weight > 0 or temp_weight > 0):
                            pred_clip = clip_preprocess(samples_x)
                            pred_embed = nnF.normalize(clip_model.encode_image(pred_clip), dim=-1)
                            if clip_weight > 0:
                                gt_clip = clip_preprocess(gt)
                                gt_embed = nnF.normalize(clip_model.encode_image(gt_clip), dim=-1)
                                clip_loss_val = (1.0 - (gt_embed * pred_embed).sum(dim=-1)).mean()
                                loss = loss + clip_weight * clip_loss_val
                            if temp_weight > 0:
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
                        if color_weight > 0:
                            color_loss_val = color_stats_loss(samples_x, gt)
                            loss = loss + color_weight * color_loss_val

                        loss_regu = torch.mean(torch.abs(c['crossattn']))
                        loss = loss + 0.1 * loss_regu

                        # 每 50 次迭代保存每个中间帧图像到子文件夹
                        if iter % 50 == 0:
                            subfolder = os.path.join(log_path, 'phase2_iter_{:05d}'.format(iter))
                            if not os.path.exists(subfolder):
                                os.makedirs(subfolder)
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
                            cv2.imwrite(os.path.join(subfolder, 'id_{:05d}.png'.format(cur_id)), img)

                        loss.backward()
                        optimizer.step()
                        optimizer.zero_grad()
                        model.model.zero_grad()

                        if cur_id not in loss_list.keys():
                            loss_list[cur_id] = loss.item()

                    mean_loss = np.mean(list(loss_list.values()))
                    if iter % 50 == 0:
                        print('Phase2 f_id={}, iter={}, mean_loss={}'.format(f_id, iter, mean_loss))
                        log_output.write('Phase2 f_id={}, iter={}, mean_loss={}\n'.format(f_id, iter, mean_loss))
                        log_output.flush()

                    if mean_loss < min_loss:
                        min_loss = mean_loss
                        ckpt = {
                            'U': Q_U, 'U_scale': Quant_Param_U.scale,
                            'U_zero_point': Quant_Param_U.zero_point, 'U_bits': Quant_Param_U.num_bits,
                            'V': Q_V, 'V_scale': Quant_Param_V.scale,
                            'V_zero_point': Quant_Param_V.zero_point, 'V_bits': Quant_Param_V.num_bits,
                            'z': samples_z, 'randn': randn, 'iter': iter, 'loss': mean_loss,
                        }
                        torch.save(ckpt, os.path.join(log_path, 'ckpt.pth'))
                        U_Byte = Quant_Param_U.quantize_tensor(Q_U).byte()
                        V_Byte = Quant_Param_V.quantize_tensor(Q_V).byte()
                        prompt = {
                            'U': U_Byte, 'V': V_Byte,
                            'U_scale': Quant_Param_U.scale, 'U_zero_point': Quant_Param_U.zero_point,
                            'V_scale': Quant_Param_V.scale, 'V_zero_point': Quant_Param_V.zero_point,
                            'weights': weights,
                        }
                        torch.save(prompt, os.path.join(prompt_path, 'frame_{:05d}.prompt'.format(f_id)))

                    # 动态训练 + LR 调度 (简化版)
                    if len(loss_list) > 0:
                        worst_step = max(loss_list, key=loss_list.get) - (f_id - interval)
                        step_list = step_list_base + [worst_step] * 2
                        step_list = sorted(step_list)
                    lr_schedule_cnt -= 1
                    if lr_schedule_cnt == 0:
                        if min_loss == latest_min_loss:
                            lr = max(lr * 0.5, 0.001)
                            optimizer = torch.optim.Adam([U, V], lr=lr)
                        latest_min_loss = min_loss
                        lr_schedule_cnt = 10

                log_output.close()

            # 将 weights 写入第一个关键帧的 prompt (Phase 2 未覆盖)
            prompt_file = os.path.join(prompt_path, 'frame_{:05d}.prompt'.format(0))
            prompt_data = torch.load(prompt_file, weights_only=True)
            prompt_data['weights'] = weights
            torch.save(prompt_data, prompt_file)
            print('Per-rank weights saved to all prompt files (including frame 0).')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-frame_path', type=str, default="data/sky")
    parser.add_argument('-max_id', type=int, default=140)
    parser.add_argument('-rank', type=int, default="8")
    parser.add_argument('-interval', type=int, default="10")
    parser.add_argument('-clip_weight', type=float, default=0.5,
                        help='Weight for CLIP semantic loss (0.0 to disable)')
    parser.add_argument('-temp_weight', type=float, default=0.1,
                        help='Weight for temporal direction consistency loss (0.0 to disable)')
    parser.add_argument('-color_weight', type=float, default=0.3,
                        help='Weight for color statistics loss (match per-channel mean/std, 0.0 to disable)')
    parser.add_argument('-suffix', type=str, default='',
                        help='Optional suffix appended to the results directory name (e.g. "_color")')
    parser.add_argument('--baseline', action='store_true',
                        help='Use only original losses (LPIPS + MSE + reg), no CLIP')
    parser.add_argument('--slerp', action='store_true', default=True,
                        help='Use spherical linear interpolation (default: True)')
    parser.add_argument('--no-slerp', action='store_false', dest='slerp',
                        help='Use linear interpolation instead of slerp')
    args = parser.parse_args()

    if args.baseline:
        print("Baseline mode: disabling CLIP semantic, temporal, color losses, and slerp.")
        args.clip_weight = 0.0
        args.temp_weight = 0.0
        args.color_weight = 0.0
        args.slerp = False

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
    if args.clip_weight > 0:
        print("Loading CLIP model (ViT-B/32) for semantic loss...")
        clip_model, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k")
        clip_model = clip_model.cuda().eval()
        for p in clip_model.parameters():
            p.requires_grad_(False)
        print("CLIP model loaded.")

    # Inversion: from video to prompts
    inversion(
       model, sampler, decoder=taesd.decoder, rank=args.rank, interval=args.interval,
       frame_path=args.frame_path, max_id=args.max_id, H=512, W=512, seed=seed_,
       filter=state.get("filter"), clip_model=clip_model, clip_weight=args.clip_weight,
       temp_weight=args.temp_weight, color_weight=args.color_weight, baseline=args.baseline,
       slerp_mode=args.slerp
    )