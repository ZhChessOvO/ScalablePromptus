"""
simulate_adaptive.py — Simulate adaptive bitrate streaming with nested dropout.

Evaluates the quality (PSNR / SSIM / LPIPS) of generated frames for a range
of truncation ranks, mimicking what would happen under varying network bandwidth.

Usage:
    python simulate_adaptive.py \
        -frame_path "data/sky" \
        -prompt_dir "data/sky/results/rank16_interval10" \
        -rank 16 \
        -interval 10 \
        --trunc_ranks 4 6 8 10 12 14 16
"""

import os
import re
import argparse
import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from PIL import Image
from diffusers import AutoencoderTiny
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


@torch.no_grad()
def generate_video_at_rank(
        model, sampler, decoder, prompt_dir, frame_path,
        interval, trunc_rank, train_rank, slerp_mode=True
):
    """Generate video frames at a specific truncation rank.
    
    Returns:
        frames: dict mapping frame_id → generated image (H, W, 3) float32 [0,1]
    """
    prompt_dir_full = prompt_dir  # path where prompts are stored

    H, W = 512, 512
    F = 8
    C = 4
    shape = (1, C, H // F, W // F)

    uc = None
    rand_noise = seeded_randn(shape, 88)
    sigma = torch.Tensor([0.05]).float().cuda()

    def denoiser(input, sigma, c):
        return model.denoiser(model.model, input, sigma, c)

    prev_frame = None
    frames = {}

    prompts = sorted(glob(os.path.join(prompt_dir_full, 'frame_*.prompt')))
    for prompt_pair in zip(prompts[::], prompts[1::]):
        prompt_curr = prompt_pair[0]
        id_curr = int(re.search(r'frame_(\d{5})\.prompt', prompt_curr).group(1))
        prompt_next = prompt_pair[1]
        id_next = int(re.search(r'frame_(\d{5})\.prompt', prompt_next).group(1))

        prompt_curr_data = torch.load(prompt_curr, weights_only=True)
        prompt_next_data = torch.load(prompt_next, weights_only=True)

        U_curr, V_curr = prompt_curr_data['U'], prompt_curr_data['V']
        U_next, V_next = prompt_next_data['U'], prompt_next_data['V']

        # Dequantize
        def dequantize(U_q, V_q, prompt_data):
            qp_u = QParam(num_bits=8)
            qp_u.scale = prompt_data['U_scale']
            qp_u.zero_point = prompt_data['U_zero_point']
            U = qp_u.dequantize_tensor(U_q)
            qp_v = QParam(num_bits=8)
            qp_v.scale = prompt_data['V_scale']
            qp_v.zero_point = prompt_data['V_zero_point']
            V = qp_v.dequantize_tensor(V_q)
            return U, V

        U_curr, V_curr = dequantize(U_curr, V_curr, prompt_curr_data)
        U_next, V_next = dequantize(U_next, V_next, prompt_next_data)

        # Get weights if available
        weights = prompt_curr_data.get('weights', None)
        if weights is not None and trunc_rank < len(weights):
            weights = weights[:trunc_rank].cuda()
        elif weights is not None:
            weights = weights.cuda()

        # Truncate to target rank
        if trunc_rank < train_rank:
            U_curr = U_curr[:, :trunc_rank]
            V_curr = V_curr[:trunc_rank, :]
            U_next = U_next[:, :trunc_rank]
            V_next = V_next[:trunc_rank, :]

        eff_rank = trunc_rank

        if prev_frame is None:
            prev_frame = torch.load(os.path.join(prompt_dir_full, 'init.pth'), weights_only=True)
            z = (prev_frame * sigma + rand_noise * (1 - sigma))
            c = (U_curr @ V_curr / np.sqrt(eff_rank)).unsqueeze(dim=0)
            prompt = {'crossattn': c}
            samples_z = sampler(denoiser, z, cond=prompt, uc=uc)
            img = decoder(samples_z)
            img = torch.clamp((img + 1.0) / 2.0, min=0.0, max=1.0)
            frames[id_curr] = img[0].permute(1, 2, 0).cpu().numpy()
            prev_frame = samples_z

        z = (prev_frame * sigma + rand_noise * (1 - sigma))
        for step in range(1, interval + 1):
            t = step / interval
            if weights is not None:
                t_i = t ** (1.0 / weights)
            else:
                t_i = torch.tensor(t, device=U_curr.device)

            if slerp_mode:
                u = slerp(U_curr.T, U_next.T, t_i.view(-1, 1)).T
                v = slerp(V_curr, V_next, t_i.view(-1, 1))
            else:
                u = (1 - t_i.view(1, -1)) * U_curr + t_i.view(1, -1) * U_next
                v = (1 - t_i.view(-1, 1)) * V_curr + t_i.view(-1, 1) * V_next

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
    """Compute PSNR/SSIM/LPIPS for generated frames vs ground truth."""
    psnr_list, ssim_list, lpips_list = [], [], []

    # LPIPS model
    loss_lpips = LearnedPerceptualImagePatchSimilarity(net_type='vgg').cuda()

    for f_id, gen_img in sorted(frames.items()):
        if f_id >= gt_total_ids:
            continue
        gt_path = os.path.join(frame_path, '{:05d}.png'.format(f_id))
        if not os.path.exists(gt_path):
            continue
        gt_img = load_image(gt_path)

        # gen_img is (H, W, 3) in [0, 1]
        p = psnr(gen_img, gt_img, data_range=1.0)
        s = ssim(gen_img, gt_img, data_range=1.0, channel_axis=-1)

        # LPIPS
        t1 = torch.from_numpy(gen_img).permute(2, 0, 1).unsqueeze(0).cuda() * 2 - 1
        t2 = torch.from_numpy(gt_img).permute(2, 0, 1).unsqueeze(0).cuda() * 2 - 1
        l = loss_lpips(t1, t2).item()

        psnr_list.append(p)
        ssim_list.append(s)
        lpips_list.append(l)

    return {
        'psnr_mean': np.mean(psnr_list),
        'psnr_std': np.std(psnr_list),
        'ssim_mean': np.mean(ssim_list),
        'ssim_std': np.std(ssim_list),
        'lpips_mean': np.mean(lpips_list),
        'lpips_std': np.std(lpips_list),
        'n_frames': len(psnr_list),
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
    parser.add_argument('--trunc_ranks', type=int, nargs='+',
                        default=[4, 6, 8, 10, 12, 14, 16],
                        help='List of truncation ranks to evaluate')
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
    print("Adaptive Bitrate Simulation: Quality vs Truncation Rank")
    print("=" * 70)
    print(f"  Frame path:   {args.frame_path}")
    print(f"  Prompt dir:   {prompt_dir}")
    print(f"  Training rank: {args.rank}")
    print(f"  Interval:     {args.interval}")
    print(f"  Max ID:       {args.max_id}")
    print(f"  Trunc ranks:  {args.trunc_ranks}")
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
    for trunc_rank in sorted(args.trunc_ranks):
        print(f"\n--- Generating with trunc_rank={trunc_rank} ---")
        frames = generate_video_at_rank(
            model, sampler, decoder=taesd.decoder,
            prompt_dir=prompt_dir, frame_path=args.frame_path,
            interval=args.interval, trunc_rank=trunc_rank,
            train_rank=args.rank, slerp_mode=args.slerp,
        )
        metrics = evaluate_frames(frames, args.frame_path, args.max_id)
        results[trunc_rank] = metrics
        print(f"  trunc_rank={trunc_rank:2d}: "
              f"PSNR={metrics['psnr_mean']:.4f}±{metrics['psnr_std']:.4f}, "
              f"SSIM={metrics['ssim_mean']:.4f}±{metrics['ssim_std']:.4f}, "
              f"LPIPS={metrics['lpips_mean']:.4f}±{metrics['lpips_std']:.4f}, "
              f"frames={metrics['n_frames']}")

    # Summary table
    print("\n" + "=" * 70)
    print("Summary: Quality vs Truncation Rank")
    print("=" * 70)
    print(f"{'Rank':>6} | {'PSNR':>8} | {'SSIM':>8} | {'LPIPS':>8} | {'Frames':>6}")
    print("-" * 70)
    for trunc_rank in sorted(results.keys()):
        m = results[trunc_rank]
        print(f"{trunc_rank:6d} | "
              f"{m['psnr_mean']:8.4f} | "
              f"{m['ssim_mean']:8.4f} | "
              f"{m['lpips_mean']:8.4f} | "
              f"{m['n_frames']:6d}")

    # Estimate bitrate (approximate, based on prompt size)
    print("\n" + "=" * 70)
    print("Estimated bitrate (U + V bytes per frame, quantized 8-bit)")
    print("=" * 70)
    # U: [77, rank] → 77 * rank bytes; V: [rank, 1024] → rank * 1024 bytes
    # Total per keyframe: (77 + 1024) * rank bytes
    # Average per frame: total / interval
    print(f"{'Rank':>6} | {'U_MB':>8} | {'V_MB':>8} | {'Total_MB':>10} | {'Avg_kbps':>10}")
    print("-" * 70)
    for trunc_rank in sorted(results.keys()):
        u_bytes = 77 * trunc_rank  # 8-bit quantized
        v_bytes = trunc_rank * 1024
        total_bytes_per_keyframe = u_bytes + v_bytes
        avg_bytes_per_frame = total_bytes_per_keyframe / args.interval
        avg_kbps = avg_bytes_per_frame * 8 / 1000  # kilobits per frame
        # Convert to kbps at 10 fps (typical)
        video_kbps = avg_kbps * 10  # assume 10 fps
        print(f"{trunc_rank:6d} | "
              f"{u_bytes / 1e6:8.4f} | "
              f"{v_bytes / 1e6:8.4f} | "
              f"{total_bytes_per_keyframe / 1e6:10.4f} | "
              f"{video_kbps:10.2f}")

    print("\nDone.")


if __name__ == "__main__":
    main()