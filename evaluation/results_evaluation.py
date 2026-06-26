"""Standalone evaluation of generated flat layouts against ground-truth images.

Computes paired color metrics (MSE, MAE, PSNR, SSIM, LPIPS), distribution
similarity (FID), and semantic segmentation IoU (by mapping each pixel color to
the nearest room class). Only files present in BOTH folders (matched by name)
are evaluated. Results are printed and saved to a CSV next to the generated
folder.

Usage:
    python results_evaluation.py --real_dir <real> --generated_dir <generated>
"""
import argparse
import os
import shutil

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm
from torchmetrics.image import (
    PeakSignalNoiseRatio,
    StructuralSimilarityIndexMeasure,
    LearnedPerceptualImagePatchSimilarity,
)
from torch_fidelity import calculate_metrics

# Room-class color palette (RGB). Pixels are mapped to the nearest class for IoU.
COLOR_PALETTE = {
    "Living room": [250, 224, 176],
    "Master room": [105, 183, 206],
    "Kitchen": [198, 88, 64],
    "Bathroom": [191, 255, 191],
    "Storage": [244, 206, 75],
    "External area": [31, 31, 31],
    "Exterior wall": [95, 95, 95],
}
NUM_CLASSES = len(COLOR_PALETTE)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Evaluation transform; must match the resolution used during training.
EVAL_RESOLUTION = 512
eval_transforms = transforms.Compose([
    transforms.Resize(EVAL_RESOLUTION, interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.CenterCrop(EVAL_RESOLUTION),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])


def rgb_to_class_index(rgb_image, color_palette):
    """Map each RGB pixel to the index of the nearest palette color.

    Args:
        rgb_image: numpy array of shape (H, W, 3), values in [0, 255].
        color_palette: dict of {class_name: [R, G, B]}.

    Returns:
        numpy array of shape (H, W) holding the class index of each pixel.
    """
    h, w, _ = rgb_image.shape
    palette_colors = np.array(list(color_palette.values()), dtype=np.float32)
    pixels = rgb_image.reshape(-1, 3).astype(np.float32)
    distances = np.sqrt(np.sum((pixels[:, np.newaxis, :] - palette_colors[np.newaxis, :, :]) ** 2, axis=2))
    class_indices = np.argmin(distances, axis=1)
    return class_indices.reshape(h, w)


def calculate_iou_metrics(pred_mask, true_mask, num_classes):
    """Compute micro IoU, macro IoU and per-class IoU between two class masks."""
    iou_list = []
    pred_mask_flat = pred_mask.flatten()
    true_mask_flat = true_mask.flatten()

    for c in range(num_classes):
        pred_inds = pred_mask_flat == c
        true_inds = true_mask_flat == c
        intersection = np.logical_and(pred_inds, true_inds).sum()
        union = np.logical_or(pred_inds, true_inds).sum()
        iou = 1.0 if union == 0 else intersection / union
        iou_list.append(iou)

    macro_iou = np.mean(iou_list)

    total_intersection = sum(
        np.logical_and(pred_mask_flat == c, true_mask_flat == c).sum()
        for c in range(num_classes)
    )
    total_union = sum(
        np.logical_or(pred_mask_flat == c, true_mask_flat == c).sum()
        for c in range(num_classes)
    )
    micro_iou = total_intersection / total_union if total_union != 0 else 1.0

    return micro_iou, macro_iou, iou_list


def load_image_tensor_custom(path, device, transform):
    """Load an image and apply the evaluation transform (for color metrics)."""
    img = Image.open(path).convert("RGB")
    tensor = transform(img)
    return tensor.unsqueeze(0).to(device)


def load_rgb_image(path, size=None):
    """Load an RGB image as a numpy array (for IoU calculation)."""
    img = Image.open(path).convert("RGB")
    if size is not None:
        img = img.resize((size, size), resample=Image.NEAREST)
    return np.array(img)


def main(real_folder, generated_folder):
    print(f"Using device: {DEVICE}")

    # 1. Build the list of files present in both folders.
    gen_files = [f for f in os.listdir(generated_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
    common_files = sorted(f for f in gen_files if os.path.exists(os.path.join(real_folder, f)))

    if not common_files:
        print("Error: no files in the generated folder have a matching name in the real folder.")
        return

    print(f"Generated folder contains {len(gen_files)} files.")
    print(f"Found {len(common_files)} matched files (present in both folders) to evaluate.")
    print(f"Number of classes: {NUM_CLASSES}")
    print(f"Classes: {list(COLOR_PALETTE.keys())}\n")

    results = {}

    # 2. Paired color metrics (MSE, MAE, PSNR, SSIM, LPIPS), computed per image.
    print("--- Computing paired color metrics (MSE, MAE, PSNR, SSIM, LPIPS) ---")
    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(DEVICE)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(DEVICE)
    lpips_metric = LearnedPerceptualImagePatchSimilarity(net_type='alex').to(DEVICE)

    mse_scores, mae_scores, psnr_scores, ssim_scores, lpips_scores = [], [], [], [], []

    for fname in tqdm(common_files, desc="Color metrics"):
        real_path = os.path.join(real_folder, fname)
        gen_path = os.path.join(generated_folder, fname)

        real_tensor = load_image_tensor_custom(real_path, DEVICE, transform=eval_transforms)
        gen_tensor = load_image_tensor_custom(gen_path, DEVICE, transform=eval_transforms)

        real_tensor_01 = (real_tensor + 1) / 2
        gen_tensor_01 = (gen_tensor + 1) / 2

        mse_scores.append(torch.mean((gen_tensor_01 - real_tensor_01) ** 2).item())
        mae_scores.append(torch.mean(torch.abs(gen_tensor_01 - real_tensor_01)).item())
        psnr_scores.append(psnr_metric(gen_tensor_01, real_tensor_01).item())
        ssim_scores.append(ssim_metric(gen_tensor_01, real_tensor_01).item())
        lpips_scores.append(lpips_metric(gen_tensor, real_tensor).item())

        # Reset metric state so each image is scored independently.
        psnr_metric.reset()
        ssim_metric.reset()
        lpips_metric.reset()

    results['MSE_mean'] = np.mean(mse_scores)
    results['MSE_std'] = np.std(mse_scores)
    results['MAE_mean'] = np.mean(mae_scores)
    results['MAE_std'] = np.std(mae_scores)
    results['PSNR_mean'] = np.mean(psnr_scores)
    results['PSNR_std'] = np.std(psnr_scores)
    results['SSIM_mean'] = np.mean(ssim_scores)
    results['SSIM_std'] = np.std(ssim_scores)
    results['LPIPS_mean'] = np.mean(lpips_scores)
    results['LPIPS_std'] = np.std(lpips_scores)

    # 3. IoU metrics (pixel color mapped to nearest room class).
    print("\n--- Computing IoU metrics (color mapped to class) ---")
    micro_iou_scores, macro_iou_scores, per_class_ious_all = [], [], []

    for fname in tqdm(common_files, desc="IoU metrics"):
        real_path = os.path.join(real_folder, fname)
        gen_path = os.path.join(generated_folder, fname)

        real_rgb = load_rgb_image(real_path, size=EVAL_RESOLUTION)
        gen_rgb = load_rgb_image(gen_path, size=EVAL_RESOLUTION)

        true_mask = rgb_to_class_index(real_rgb, COLOR_PALETTE)
        pred_mask = rgb_to_class_index(gen_rgb, COLOR_PALETTE)

        micro, macro, per_class = calculate_iou_metrics(pred_mask, true_mask, NUM_CLASSES)
        micro_iou_scores.append(micro)
        macro_iou_scores.append(macro)
        per_class_ious_all.append(per_class)

    results['Micro_IoU_mean'] = np.mean(micro_iou_scores)
    results['Micro_IoU_std'] = np.std(micro_iou_scores)
    results['Macro_IoU_mean'] = np.mean(macro_iou_scores)
    results['Macro_IoU_std'] = np.std(macro_iou_scores)

    per_class_ious_arr = np.array(per_class_ious_all)  # (N, num_classes)
    per_class_ious_avg = np.mean(per_class_ious_arr, axis=0)
    per_class_ious_std = np.std(per_class_ious_arr, axis=0)
    for idx, (class_name, _) in enumerate(COLOR_PALETTE.items()):
        results[f'IoU_{class_name.replace(" ", "_")}_mean'] = per_class_ious_avg[idx]
        results[f'IoU_{class_name.replace(" ", "_")}_std'] = per_class_ious_std[idx]

    # 4. FID (distribution-level metric).
    print("\n--- Computing distribution metric (FID) ---")
    temp_gen_folder = './temp_generated_for_fid'
    if os.path.exists(temp_gen_folder):
        shutil.rmtree(temp_gen_folder)
    os.makedirs(temp_gen_folder)

    for fname in common_files:
        shutil.copy(os.path.join(generated_folder, fname), os.path.join(temp_gen_folder, fname))

    fid_metrics = calculate_metrics(
        input1=temp_gen_folder,
        input2=real_folder,
        cuda=(DEVICE == 'cuda'),
        isc=False,
        fid=True,
        kid=False,
        verbose=False,
        feature_layer_fid='2048',
        batch_size=1000,
    )
    results['FID'] = fid_metrics['frechet_inception_distance']
    shutil.rmtree(temp_gen_folder)

    # 5. Print and save results.
    print("\n" + "=" * 60)
    print("Final evaluation results (mean +/- std)")
    print("=" * 60)

    print("\n[Color similarity]")
    print(f"MSE:   {results['MSE_mean']:.4f} +/- {results['MSE_std']:.4f}")
    print(f"MAE:   {results['MAE_mean']:.4f} +/- {results['MAE_std']:.4f}")
    print(f"PSNR:  {results['PSNR_mean']:.4f} +/- {results['PSNR_std']:.4f}")
    print(f"SSIM:  {results['SSIM_mean']:.4f} +/- {results['SSIM_std']:.4f}")
    print(f"LPIPS: {results['LPIPS_mean']:.4f} +/- {results['LPIPS_std']:.4f}")

    print("\n[Distribution similarity]")
    print(f"FID:   {results['FID']:.4f}")

    print("\n[Semantic segmentation IoU]")
    print(f"Micro IoU: {results['Micro_IoU_mean']:.4f} +/- {results['Micro_IoU_std']:.4f}")
    print(f"Macro IoU: {results['Macro_IoU_mean']:.4f} +/- {results['Macro_IoU_std']:.4f}")

    print("\n[Per-class IoU]")
    for class_name in COLOR_PALETTE:
        mean_key = f'IoU_{class_name.replace(" ", "_")}_mean'
        std_key = f'IoU_{class_name.replace(" ", "_")}_std'
        print(f"{class_name:15s}: {results[mean_key]:.4f} +/- {results[std_key]:.4f}")

    print("=" * 60)

    output_dir = os.path.abspath(os.path.join(generated_folder, os.pardir))
    csv_path = os.path.join(output_dir, 'image_evaluation_results_withstd.csv')
    pd.DataFrame([results]).to_csv(csv_path, index=False, float_format='%.4f')
    print(f"\nResults saved to: {csv_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate generated flat layouts against ground-truth images."
    )
    parser.add_argument("--real_dir", required=True, help="Directory of ground-truth images.")
    parser.add_argument("--generated_dir", required=True, help="Directory of generated images.")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args.real_dir, args.generated_dir)
