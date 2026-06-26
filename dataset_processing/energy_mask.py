"""Generate the "energy field" ControlNet condition images.

For each flat layout, this script computes the bounding box of the building
footprint, expands it to a circumscribed square, and renders a Gaussian-blurred
box whose blur strength is driven by the per-plan normalized energy value. The
result is a smooth grayscale field used as a second ControlNet condition.

The energy value for each file is read from a CSV with columns
`filename` and `energy_normalization` (i.e. `dataset/energy_value.csv`).

Usage:
    python energy_mask.py --input_dir <conditions> --output_dir <energy_fields> \
        --energy_csv dataset/energy_value.csv
"""
import argparse
import os

import cv2
import numpy as np
import pandas as pd


def get_global_bbox_from_mask(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = (gray > 10)
    ys, xs = np.where(mask)
    if ys.size == 0 or xs.size == 0:
        return None
    x_min = int(xs.min())
    y_min = int(ys.min())
    x_max = int(xs.max()) + 1
    y_max = int(ys.max()) + 1
    return (x_min, y_min, x_max, y_max)


def bbox_to_circumscribed_square(bbox, image_shape):
    H, W = image_shape
    x_min, y_min, x_max, y_max = bbox
    w = x_max - x_min
    h = y_max - y_min
    side = max(w, h)
    x_center = (x_min + x_max) // 2
    y_center = (y_min + y_max) // 2
    half_side = side // 2
    new_x_min = x_center - half_side
    new_y_min = y_center - half_side
    new_x_max = x_center + half_side + (side % 2)
    new_y_max = y_center + half_side + (side % 2)

    # Clamp coordinates that fall outside the image to the original bbox edges.
    if new_x_min < 0:
        new_x_min = x_min
    if new_y_min < 0:
        new_y_min = y_min
    if new_x_max > W:
        new_x_max = x_max
    if new_y_max > H:
        new_y_max = y_max

    return (new_x_min, new_y_min, new_x_max, new_y_max)


def make_blurred_bbox_field(image_size, bbox, blur_sigma):
    H, W = image_size
    x_min, y_min, x_max, y_max = bbox
    field = np.zeros((H, W), dtype=np.float32)
    field[y_min:y_max, x_min:x_max] = 1.0
    ksize = int(2 * round(3 * blur_sigma) + 1)
    if ksize % 2 == 0:
        ksize += 1
    blurred = cv2.GaussianBlur(field, (ksize, ksize), blur_sigma)
    blurred = blurred / blurred.max()
    return blurred


def process_image_energy(image_path, energy_level, output_dir):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Warning: Could not read image {image_path}. Skipping.")
        return image_path

    bbox = get_global_bbox_from_mask(img)
    if bbox is None:
        print(f"Warning: No bounding box found for {image_path}. Skipping.")
        return image_path

    H, W = img.shape[:2]
    square_bbox = bbox_to_circumscribed_square(bbox, (H, W))

    blur_sigma = 2 + 20 * float(energy_level)

    warning_triggered = False
    if np.isnan(blur_sigma) or blur_sigma <= 0:
        print(f"Warning: Invalid blur_sigma ({blur_sigma}) for image {image_path}. Defaulting to 2.0.")
        blur_sigma = 2.0
        warning_triggered = True

    old_settings = np.seterr(all='raise')
    try:
        F = make_blurred_bbox_field((H, W), square_bbox, blur_sigma)
        img_uint8 = (F * 255).clip(0, 255).astype(np.uint8)
    except FloatingPointError as e:
        print(f"Warning: Floating point error ('{e}') processing image {image_path}.")
        img_uint8 = np.zeros((H, W), dtype=np.uint8)
        warning_triggered = True
    finally:
        np.seterr(**old_settings)

    output_path = os.path.join(output_dir, os.path.basename(image_path))
    cv2.imwrite(output_path, img_uint8)

    return image_path if warning_triggered else None


def process_folder_with_csv_energy(input_folder, output_folder, csv_path):
    df = pd.read_csv(csv_path)
    energy_map = dict(zip(df['filename'], df['energy_normalization']))
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    warning_files = []
    for filename in os.listdir(input_folder):
        if not filename.lower().endswith(".png"):
            continue

        input_path = os.path.join(input_folder, filename)
        energy_level = energy_map.get(filename)
        if energy_level is None:
            print(f"Warning: No energy value found for {filename}. Skipping.")
            continue

        result = process_image_energy(input_path, energy_level, output_folder)
        if result is not None:
            warning_files.append(result)

    if warning_files:
        print("\n--- Processing Complete with Warnings ---")
        print("Warnings were generated for the following files:")
        for f in warning_files:
            print(f"- {f}")
    else:
        print("\n--- Processing Complete Successfully ---")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate blurred 'energy field' ControlNet condition images."
    )
    parser.add_argument("--input_dir", required=True, help="Directory of input mask/condition PNGs.")
    parser.add_argument("--output_dir", required=True, help="Directory to write energy-field images to.")
    parser.add_argument("--energy_csv", required=True,
                        help="CSV with columns 'filename' and 'energy_normalization' "
                             "(e.g. dataset/energy_value.csv).")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    process_folder_with_csv_energy(args.input_dir, args.output_dir, args.energy_csv)
