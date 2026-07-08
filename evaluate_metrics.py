"""
Evaluate PSNR, SSIM, and LPIPS between generated results and ground truth.

Data structure:
  /root/autodl-tmp/sky/results/rank2_interval10/{folder}/iter_01490_id_{id:05d}.png
  - folder 00000: id 00000
  - folder 00010: id 00001~00010
  - folder 00020: id 00011~00020
  - ...
  - folder 00130: id 00121~00130

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
DATA_DIR = "/root/autodl-tmp/sky/results/rank8_interval10_baseline"
GT_DIR = "/root/Promptus/data/sky"
ITER_STEP = "01450"
TOTAL_IDS = 131  # id 00000 ~ 00130 (inclusive)

device = "cuda" if torch.cuda.is_available() else "cpu"

def load_image(path):
    """Load an image and convert to numpy array (H, W, C) in [0, 1] range."""
    img = Image.open(path).convert("RGB")
    return np.array(img).astype(np.float32) / 255.0


def compute_psnr(img1, img2):
    """Compute PSNR between two images."""
    result = psnr(img1, img2, data_range=1.0)
    return result


def compute_ssim(img1, img2):
    """Compute SSIM between two images."""
    result = ssim(img1, img2, data_range=1.0, channel_axis=-1)
    return result


def compute_lpips(img1, img2, lpips_model):
    """Compute LPIPS distance between two images (lower is better)."""
    # Convert to torch tensors: (1, 3, H, W) in [-1, 1]
    t1 = torch.from_numpy(img1).permute(2, 0, 1).unsqueeze(0).to(device) * 2 - 1
    t2 = torch.from_numpy(img2).permute(2, 0, 1).unsqueeze(0).to(device) * 2 - 1
    with torch.no_grad():
        dist = lpips_model(t1, t2)
    return dist.item()


def get_result_path(id_val):
    """Get the path to the result image for a given id.
    
    Mapping:
      id=0      -> folder 00000
      id=1~10   -> folder 00010
      id=11~20  -> folder 00020
      ...
      id=121~130 -> folder 00130
    """
    # if id_val == 0:
    #     result_path = os.path.join(
    #         DATA_DIR, "00000", "iter_09990_id_00000.png"
    #     )
    # else:
    folder = ((id_val + 9) // 10) * 10
    result_path = os.path.join(
        DATA_DIR, f"{folder:05d}", f"iter_{ITER_STEP}_id_{id_val:05d}.png"
    )
    return result_path


def get_gt_path(id_val):
    """Get the path to the GT image for a given id."""
    return os.path.join(GT_DIR, f"{id_val:05d}.png")


def main():
    print("=" * 60)
    print("Evaluation: PSNR, SSIM, LPIPS")
    print(f"  Data dir: {DATA_DIR}")
    print(f"  GT dir:   {GT_DIR}")
    print(f"  Iter step: {ITER_STEP}")
    print(f"  IDs: 00000 ~ {TOTAL_IDS - 1:05d}")
    print(f"  Device: {device}")
    print("=" * 60)

    # Initialize LPIPS model
    lpips_model = lpips.LPIPS(net="alex").to(device)
    lpips_model.eval()

    psnr_list = []
    ssim_list = []
    lpips_list = []
    missing_files = []

    for id_val in range(0, TOTAL_IDS):
        result_path = get_result_path(id_val)
        gt_path = get_gt_path(id_val)

        if not os.path.exists(result_path):
            missing_files.append(result_path)
            continue
        if not os.path.exists(gt_path):
            missing_files.append(gt_path)
            continue

        # Load images
        result_img = load_image(result_path)
        gt_img = load_image(gt_path)

        # Check shapes match
        if result_img.shape != gt_img.shape:
            print(f"  [Warning] Shape mismatch for id {id_val:05d}: "
                  f"result {result_img.shape} vs gt {gt_img.shape}")
            continue

        # Compute metrics
        p = compute_psnr(result_img, gt_img)
        s = compute_ssim(result_img, gt_img)
        l = compute_lpips(result_img, gt_img, lpips_model)

        psnr_list.append(p)
        ssim_list.append(s)
        lpips_list.append(l)

        print(f"  id {id_val:05d}: PSNR={p:.4f}  SSIM={s:.4f}  LPIPS={l:.4f}")

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
        print(f"  PSNR: {np.mean(psnr_list):.4f} ± {np.std(psnr_list):.4f}  "
              f"(min={np.min(psnr_list):.4f}, max={np.max(psnr_list):.4f})")
        print(f"  SSIM: {np.mean(ssim_list):.4f} ± {np.std(ssim_list):.4f}  "
              f"(min={np.min(ssim_list):.4f}, max={np.max(ssim_list):.4f})")
        print(f"  LPIPS: {np.mean(lpips_list):.4f} ± {np.std(lpips_list):.4f}  "
              f"(min={np.min(lpips_list):.4f}, max={np.max(lpips_list):.4f})")
    else:
        print("  No valid image pairs found.")
    print("=" * 60)


if __name__ == "__main__":
    main()
