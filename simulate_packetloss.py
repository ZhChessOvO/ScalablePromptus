"""
simulate_packetloss.py — Simulate packet loss during prompt transmission using
the Gilbert-Elliott (GE) model.

Evaluates the quality (PSNR / SSIM / LPIPS) of generated frames when
prompt components are randomly lost during transmission, mimicking what
would happen under unreliable network conditions.

Transmission order (as required):
    Frame-by-frame, rank-by-rank (small→large):
        keyframe 0: rank 0, rank 1, ..., rank R-1
        keyframe 1: rank 0, rank 1, ..., rank R-1
        ...

The GE model alternates between Good and Bad states:
  - Good state: no packet loss
  - Bad state:  high probability of packet loss
Transitions: p_gb (Good→Bad), p_bg (Bad→Good).

The GE Markov chain runs continuously across the entire transmission
sequence (state persists across keyframes). Essential components
(rank 0 .. min_rank-1) are always preserved. Lost components are
ZEROED and stay zero — safe_slerp_col prevents the other keyframe
from "filling in" lost information during interpolation.

Usage:
    python simulate_packetloss.py \\
        -frame_path "data/sky" \\
        -prompt_dir "data/sky/results/rank16_interval10" \\
        -rank 16 \\
        -interval 10 \\
        --min_rank 4 \\
        --loss_rates 0.0 0.1 0.2 0.3 0.4 0.5
"""

import os
import re
import argparse
import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from PIL import Image
from glob import glob
from diffusers import AutoencoderTiny
from torchvision.utils import save_image
from scripts.demo.streamlit_helpers import *
from sgm.modules.diffusionmodules.sampling import EulerAncestralSampler
from quantization import QParam
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

VERSION2SPECS = {
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


def load_image(path):
    """Load an image and convert to numpy array (H, W, C) in [0, 1] range."""
    img = Image.open(path).convert("RGB")
    return np.array(img).astype(np.float32) / 255.0


def gilbert_elliott_global_mask(
        n_keyframes, train_rank, min_rank, loss_rate,
        p_gb=0.1, p_bg=0.3, seed=42
):
    """Generate a global Gilbert-Elliott loss mask across ALL keyframes.

    Packet transmission order (as required):
        keyframe 0: rank 0, rank 1, ..., rank R-1
        keyframe 1: rank 0, rank 1, ..., rank R-1
        ...

    The GE Markov chain runs continuously across this entire sequence.
    Ranks 0 .. min_rank-1 are essential and ALWAYS survive (override to no-loss).
    Ranks min_rank .. R-1 are subject to GE burst loss.

    Args:
        n_keyframes: total number of keyframe prompts.
        train_rank:  full rank R.
        min_rank:    essential rank (ranks < min_rank never drop).
        loss_rate:   target average loss rate on non-essential ranks.
        p_gb:        Good → Bad transition probability.
        p_bg:        Bad → Good transition probability.
        seed:        random seed.

    Returns:
        keep_mask: bool array of shape (n_keyframes, train_rank).
                   True = packet arrived, False = lost (zero it out).
    """
    rng = np.random.RandomState(seed)

    # Steady-state probability of Bad state
    p_bad = p_gb / (p_gb + p_bg + 1e-12)

    # Derive loss probability in Bad state to hit target loss_rate.
    # loss_rate = P(bad) * loss_bad  =>  loss_bad = loss_rate / P(bad)
    if p_bad > 0:
        loss_bad = min(loss_rate / p_bad, 1.0)
    else:
        loss_bad = 0.0

    # Total packets: all keyframes × all ranks
    n_total = n_keyframes * train_rank
    state = 0  # start in Good
    packet_dropped = np.zeros(n_total, dtype=bool)

    for i in range(n_total):
        if state == 0:  # Good → no loss, may transition to Bad
            packet_dropped[i] = False
            if rng.rand() < p_gb:
                state = 1
        else:  # Bad → loss with prob loss_bad, may transition to Good
            packet_dropped[i] = rng.rand() < loss_bad
            if rng.rand() < p_bg:
                state = 0

    # Reshape to (n_keyframes, train_rank)
    packet_dropped = packet_dropped.reshape(n_keyframes, train_rank)

    # Essential ranks (0 .. min_rank-1) always survive
    packet_dropped[:, :min_rank] = False

    # keep_mask: True = survived, False = lost
    keep_mask = ~packet_dropped
    return keep_mask


def safe_slerp_col(a, b, t, eps=1e-5):
    """SLERP for a single column vector, handling zero-vector gracefully.

    If either vector is all-zero (lost packet), return the other one as-is
    (or zero if both are lost).  Otherwise perform normal SLERP.

    Args:
        a, b: 1-D tensors of same shape.
        t:    interpolation factor in [0, 1].
    Returns:
        interpolated 1-D tensor.
    """
    a_is_zero = (a.abs().max() < 1e-8)
    b_is_zero = (b.abs().max() < 1e-8)

    if a_is_zero and b_is_zero:
        return torch.zeros_like(a)
    if a_is_zero:
        return b
    if b_is_zero:
        return a

    # Normal SLERP
    a_n = a / (a.norm() + 1e-12)
    b_n = b / (b.norm() + 1e-12)
    cos_theta = (a_n * b_n).sum().clamp(-1.0, 1.0)
    theta = torch.acos(cos_theta)
    sin_theta = torch.sin(theta)

    if sin_theta < eps:
        return (1.0 - t) * a + t * b

    factor_a = torch.sin((1.0 - t) * theta) / sin_theta
    factor_b = torch.sin(t * theta) / sin_theta
    return factor_a * a + factor_b * b


def apply_global_loss_to_prompts(all_prompts, keep_mask):
    """Zero out lost rank components across all keyframe prompts.

    Args:
        all_prompts: list of dicts with keys 'U', 'V' (dequantized tensors).
        keep_mask:   bool array (n_keyframes, train_rank), True = survived.

    Returns:
        all_prompts (modified in-place).
    """
    for k, prompt_data in enumerate(all_prompts):
        lost = ~keep_mask[k]  # ranks to zero out
        lost_indices = np.where(lost)[0]
        if len(lost_indices) > 0:
            prompt_data['U'][:, lost_indices] = 0
            prompt_data['V'][lost_indices, :] = 0
    return all_prompts


@torch.no_grad()
def generate_video_with_loss(
        model, sampler, decoder, prompt_dir, frame_path,
        interval, min_rank, train_rank, loss_rate,
        p_gb=0.1, p_bg=0.3, slerp_mode=True
):
    """Generate video frames with simulated packet loss on each keyframe pair.

    Packet loss is modelled with a GLOBAL Gilbert-Elliott chain that runs
    continuously across ALL keyframes × ALL ranks (frame-by-frame,
    rank-by-rank as required). Lost rank components are set to zero and
    STAY zero — safe_slerp_col prevents the other keyframe from "filling in"
    the lost information.

    Returns:
        frames: dict mapping frame_id → generated image (H, W, 3) float32 [0, 1].
    """
    prompt_dir_full = prompt_dir

    H, W = 512, 512
    F = 8
    C = 4
    shape = (1, C, H // F, W // F)

    uc = None
    rand_noise = seeded_randn(shape, 88)
    sigma = torch.Tensor([0.05]).float().cuda()

    def denoiser(input, sigma, c):
        return model.denoiser(model.model, input, sigma, c)

    # ---- Phase 1: load all keyframe prompts into memory ----
    prompt_paths = sorted(glob(os.path.join(prompt_dir_full, 'frame_*.prompt')))
    all_prompts_raw = []
    for pp in prompt_paths:
        all_prompts_raw.append(torch.load(pp, weights_only=True))

    # Dequantize all
    all_prompts = []
    for pd in all_prompts_raw:
        qp_u = QParam(num_bits=8)
        qp_u.scale = pd['U_scale']
        qp_u.zero_point = pd['U_zero_point']
        U = qp_u.dequantize_tensor(pd['U'])
        qp_v = QParam(num_bits=8)
        qp_v.scale = pd['V_scale']
        qp_v.zero_point = pd['V_zero_point']
        V = qp_v.dequantize_tensor(pd['V'])
        all_prompts.append({'U': U.clone(), 'V': V.clone()})

    # ---- Phase 2: generate GLOBAL GE loss mask across all keyframes ----
    n_keyframes = len(all_prompts)
    keep_mask = gilbert_elliott_global_mask(
        n_keyframes=n_keyframes,
        train_rank=train_rank,
        min_rank=min_rank,
        loss_rate=loss_rate,
        p_gb=p_gb,
        p_bg=p_bg,
        seed=42,
    )

    # ---- Phase 3: apply loss (zero out lost ranks) ----
    all_prompts = apply_global_loss_to_prompts(all_prompts, keep_mask)

    # ---- Phase 4: generate video using lost prompts ----
    prev_frame = None
    frames = {}
    eff_rank = train_rank  # normalise by full training rank

    for idx in range(n_keyframes - 1):
        prompt_curr = all_prompts[idx]
        prompt_next = all_prompts[idx + 1]
        path_curr = prompt_paths[idx]
        path_next = prompt_paths[idx + 1]

        id_curr = int(re.search(r'frame_(\d{5})\.prompt', path_curr).group(1))
        id_next = int(re.search(r'frame_(\d{5})\.prompt', path_next).group(1))

        U_curr, V_curr = prompt_curr['U'], prompt_curr['V']
        U_next, V_next = prompt_next['U'], prompt_next['V']

        # ---- First keyframe (init) ----
        if prev_frame is None:
            prev_frame = torch.load(
                os.path.join(prompt_dir_full, 'init.pth'), weights_only=True
            )
            z = (prev_frame * sigma + rand_noise * (1 - sigma))
            c = (U_curr @ V_curr / np.sqrt(eff_rank)).unsqueeze(dim=0)
            prompt = {'crossattn': c}
            samples_z = sampler(denoiser, z, cond=prompt, uc=uc)
            img = decoder(samples_z)
            img = torch.clamp((img + 1.0) / 2.0, min=0.0, max=1.0)
            frames[id_curr] = img[0].permute(1, 2, 0).cpu().numpy()
            prev_frame = samples_z

        # ---- Interpolate between keyframes ----
        z = (prev_frame * sigma + rand_noise * (1 - sigma))
        for step in range(1, interval + 1):
            t = step / interval
            t_i = torch.tensor(t, device=U_curr.device)

            # Build interpolated U, V rank-by-rank using safe_slerp_col
            u_cols = []
            v_rows = []
            for r in range(train_rank):
                u_c = U_curr[:, r]
                u_n = U_next[:, r]
                v_c = V_curr[r, :]
                v_n = V_next[r, :]

                if slerp_mode:
                    u_interp = safe_slerp_col(u_c, u_n, t_i)
                    v_interp = safe_slerp_col(v_c, v_n, t_i)
                else:
                    # LERP fallback
                    u_interp = (1 - t_i) * u_c + t_i * u_n
                    v_interp = (1 - t_i) * v_c + t_i * v_n

                u_cols.append(u_interp)
                v_rows.append(v_interp)

            u = torch.stack(u_cols, dim=1)  # (d, R)
            v = torch.stack(v_rows, dim=0)  # (R, d)

            c = (u @ v / np.sqrt(eff_rank)).unsqueeze(dim=0)
            prompt = {'crossattn': c}
            samples_z = sampler(denoiser, z, cond=prompt, uc=uc)
            img = decoder(samples_z)
            img = torch.clamp((img + 1.0) / 2.0, min=0.0, max=1.0)
            frame_id = id_curr + step
            frames[frame_id] = img[0].permute(1, 2, 0).cpu().numpy()
            prev_frame = samples_z
            z = (prev_frame * sigma + rand_noise * (1 - sigma))

    return frames


def evaluate_frames(frames, frame_path, gt_total_ids):
    """Compute PSNR / SSIM / LPIPS for generated frames vs ground truth."""
    psnr_list, ssim_list, lpips_list = [], [], []

    loss_lpips = LearnedPerceptualImagePatchSimilarity(net_type='vgg').cuda()

    for f_id, gen_img in sorted(frames.items()):
        if f_id >= gt_total_ids:
            continue
        gt_path = os.path.join(frame_path, '{:05d}.png'.format(f_id))
        if not os.path.exists(gt_path):
            continue
        gt_img = load_image(gt_path)

        p = psnr(gen_img, gt_img, data_range=1.0)
        s = ssim(gen_img, gt_img, data_range=1.0, channel_axis=-1)

        t1 = torch.from_numpy(gen_img).permute(2, 0, 1).unsqueeze(0).cuda() * 2 - 1
        t2 = torch.from_numpy(gt_img).permute(2, 0, 1).unsqueeze(0).cuda() * 2 - 1
        l = loss_lpips(t1, t2).item()

        psnr_list.append(p)
        ssim_list.append(s)
        lpips_list.append(l)

    return {
        'psnr_mean': np.mean(psnr_list),
        'psnr_std':  np.std(psnr_list),
        'ssim_mean': np.mean(ssim_list),
        'ssim_std':  np.std(ssim_list),
        'lpips_mean': np.mean(lpips_list),
        'lpips_std':  np.std(lpips_list),
        'n_frames':   len(psnr_list),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-frame_path', type=str, default="data/sky",
                        help='Path to video frames directory')
    parser.add_argument('-prompt_dir', type=str, default=None,
                        help='Path to prompt directory (overrides auto-resolve)')
    parser.add_argument('-rank', type=int, default=16,
                        help='Training rank')
    parser.add_argument('-interval', type=int, default=10,
                        help='Keyframe interval')
    parser.add_argument('-max_id', type=int, default=140,
                        help='Maximum frame ID')
    parser.add_argument('--min_rank', type=int, default=4,
                        help='Essential rank always preserved (no packet loss)')
    parser.add_argument('--loss_rates', type=float, nargs='+',
                        default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
                        help='List of target packet loss rates to evaluate')
    parser.add_argument('--p_gb', type=float, default=0.1,
                        help='GE model: Good → Bad transition probability')
    parser.add_argument('--p_bg', type=float, default=0.3,
                        help='GE model: Bad → Good transition probability')
    parser.add_argument('--slerp', action='store_true', default=True)
    parser.add_argument('--no-slerp', action='store_false', dest='slerp')
    args = parser.parse_args()

    # Resolve prompt directory
    if args.prompt_dir is None:
        prompt_dir = os.path.join(
            args.frame_path,
            'results/rank{}_interval{}'.format(args.rank, args.interval)
        )
    else:
        prompt_dir = args.prompt_dir

    print("=" * 70)
    print("Packet Loss Simulation: Quality vs Loss Rate (Gilbert-Elliott Model)")
    print("=" * 70)
    print(f"  Frame path:    {args.frame_path}")
    print(f"  Prompt dir:    {prompt_dir}")
    print(f"  Training rank: {args.rank}")
    print(f"  Min rank:      {args.min_rank}")
    print(f"  Interval:      {args.interval}")
    print(f"  Max ID:        {args.max_id}")
    print(f"  Loss rates:    {args.loss_rates}")
    print(f"  GE params:     p_gb={args.p_gb}, p_bg={args.p_bg}")
    print("=" * 70)

    # Load models once
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

    results = {}
    for loss_rate in sorted(args.loss_rates):
        print(f"\n--- Generating with loss_rate={loss_rate:.2f} ---")
        frames = generate_video_with_loss(
            model, sampler, decoder=taesd.decoder,
            prompt_dir=prompt_dir, frame_path=args.frame_path,
            interval=args.interval, min_rank=args.min_rank,
            train_rank=args.rank, loss_rate=loss_rate,
            p_gb=args.p_gb, p_bg=args.p_bg,
            slerp_mode=args.slerp,
        )
        metrics = evaluate_frames(frames, args.frame_path, args.max_id)
        results[loss_rate] = metrics
        print(f"  loss_rate={loss_rate:.2f}: "
              f"PSNR={metrics['psnr_mean']:.4f}±{metrics['psnr_std']:.4f}, "
              f"SSIM={metrics['ssim_mean']:.4f}±{metrics['ssim_std']:.4f}, "
              f"LPIPS={metrics['lpips_mean']:.4f}±{metrics['lpips_std']:.4f}, "
              f"frames={metrics['n_frames']}")

        # Save first frame to image_demo
        save_dir = "/root/autodl-tmp/image_demo"
        os.makedirs(save_dir, exist_ok=True)
        first_fid = min(frames.keys())
        first_img_tensor = torch.from_numpy(frames[first_fid]).permute(2, 0, 1).unsqueeze(0)
        save_path = os.path.join(save_dir, f"frame_loss{loss_rate:.2f}.png")
        save_image(first_img_tensor, save_path)
        print(f"  Saved first frame (id={first_fid}) to {save_path}")

    # Summary table
    print("\n" + "=" * 70)
    print("Summary: Quality vs Packet Loss Rate")
    print("=" * 70)
    print(f"{'Loss':>8} | {'PSNR':>8} | {'SSIM':>8} | {'LPIPS':>8} | {'Frames':>6}")
    print("-" * 70)
    for loss_rate in sorted(results.keys()):
        m = results[loss_rate]
        print(f"{loss_rate:8.2f} | "
              f"{m['psnr_mean']:8.4f} | "
              f"{m['ssim_mean']:8.4f} | "
              f"{m['lpips_mean']:8.4f} | "
              f"{m['n_frames']:6d}")

    # Estimated overhead comparison
    print("\n" + "=" * 70)
    print("Estimated effective components (min_rank + surviving extras)")
    print("=" * 70)
    print(f"{'Loss':>8} | {'Eff_rank':>10} | {'Surviving':>10}")
    print("-" * 50)
    for loss_rate in sorted(results.keys()):
        if loss_rate == 0.0:
            eff = args.rank
        else:
            # Expected surviving: min_rank + (rank-min_rank)*(1-loss_rate)
            eff = args.min_rank + (args.rank - args.min_rank) * (1 - loss_rate)
        print(f"{loss_rate:8.2f} | {eff:10.2f} | {eff / args.rank * 100:9.1f}%")

    print("\nDone.")


if __name__ == "__main__":
    main()
