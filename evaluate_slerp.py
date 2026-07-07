"""
Evaluate PSNR, SSIM, and LPIPS between generated results and ground truth.

Data structure (rank8_interval10):
  /root/autodl-tmp/sky/results/rank8_interval10/00000/iter_09950_id_00000.png    (id=0)
  /root/autodl-tmp/sky/results/rank8_interval10/00010/phase2_iter_01450/id_00001.png  ~ id_00010.png
  /root/autodl-tmp/sky/results/rank8_interval10/00020/phase2_iter_01450/id_00011.png  ~ id_00020.png
  ...
  /root/autodl-tmp/sky/results/rank8_interval10/00130/phase2_iter_01450/id_00121.png  ~ id_00130.png

GT:
  /root/Promptus/data/sky/{id:05d}.png
"""

import os
import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from PIL import Image
import lpips

# ===== Config =====
DATA_DIR = "/root/autodl-tmp/sky/results/rank8_interval10"
GT_DIR = "/root/Promptus/data/sky"
TOTAL_IDS = 101  # id 00000 ~ 00130

device = "cuda" if torch.cuda.is_available() else "cpu"


def load_image(path):
    img = Image.open(path).convert("RGB")
    return np.array(img).astype(np.float32) / 255.0


def compute_psnr(img1, img2):
    return psnr(img1, img2, data_range=1.0)


def compute_ssim(img1, img2):
    return ssim(img1, img2, data_range=1.0, channel_axis=-1)


def compute_lpips(img1, img2, lpips_model):
    t1 = torch.from_numpy(img1).permute(2, 0, 1).unsqueeze(0).to(device) * 2 - 1
    t2 = torch.from_numpy(img2).permute(2, 0, 1).unsqueeze(0).to(device) * 2 - 1
    with torch.no_grad():
        dist = lpips_model(t1, t2)
    return dist.item()


def get_result_path(id_val):
    """
    Get path to the result image.
    - id=0:   folder 00000, file iter_09950_id_00000.png
    - id>=1:  folder = ((id_val + 9) // 10) * 10,
              subdir phase2_iter_01450, file id_{id_val:05d}.png
    """
    if id_val == 0:
        return os.path.join(DATA_DIR, "00000", f"iter_09950_id_00000.png")
    else:
        folder = ((id_val + 9) // 10) * 10
        return os.path.join(
            DATA_DIR, f"{folder:05d}", "phase2_iter_01450", f"id_{id_val:05d}.png"
        )


def get_gt_path(id_val):
    return os.path.join(GT_DIR, f"{id_val:05d}.png")


def main():
    print("=" * 60)
    print("Evaluation: PSNR, SSIM, LPIPS")
    print(f"  Data dir: {DATA_DIR}")
    print(f"  GT dir:   {GT_DIR}")
    print(f"  IDs: 00000 ~ {TOTAL_IDS - 1:05d}")
    print(f"  Device: {device}")
    print("=" * 60)

    lpips_model = lpips.LPIPS(net="alex").to(device)
    lpips_model.eval()

    psnr_list = []
    ssim_list = []
    lpips_list = []
    missing_files = []

    for id_val in range(TOTAL_IDS):
        result_path = get_result_path(id_val)
        gt_path = get_gt_path(id_val)

        if not os.path.exists(result_path):
            missing_files.append(result_path)
            continue
        if not os.path.exists(gt_path):
            missing_files.append(gt_path)
            continue

        result_img = load_image(result_path)
        gt_img = load_image(gt_path)

        if result_img.shape != gt_img.shape:
            print(f"  [Warning] Shape mismatch for id {id_val:05d}: "
                  f"result {result_img.shape} vs gt {gt_img.shape}")
            continue

        p = compute_psnr(result_img, gt_img)
        s = compute_ssim(result_img, gt_img)
        l_val = compute_lpips(result_img, gt_img, lpips_model)

        psnr_list.append(p)
        ssim_list.append(s)
        lpips_list.append(l_val)

        print(f"  id {id_val:05d}: PSNR={p:.4f}  SSIM={s:.4f}  LPIPS={l_val:.4f}")

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    if missing_files:
        print(f"  Missing files: {len(missing_files)}")
        for f in missing_files[:5]:
            print(f"    - {f}")
        if len(missing_files) > 5:
            print(f"    ... and {len(missing_files) - 5} more")
    if psnr_list:
        print(f"  Number of valid pairs: {len(psnr_list)}")
        print(f"  PSNR:  {np.mean(psnr_list):.4f} ± {np.std(psnr_list):.4f}  "
              f"(min={np.min(psnr_list):.4f}, max={np.max(psnr_list):.4f})")
        print(f"  SSIM:  {np.mean(ssim_list):.4f} ± {np.std(ssim_list):.4f}  "
              f"(min={np.min(ssim_list):.4f}, max={np.max(ssim_list):.4f})")
        print(f"  LPIPS: {np.mean(lpips_list):.4f} ± {np.std(lpips_list):.4f}  "
              f"(min={np.min(lpips_list):.4f}, max={np.max(lpips_list):.4f})")
    else:
        print("  No valid image pairs found.")
    print("=" * 60)


if __name__ == "__main__":
    main()
