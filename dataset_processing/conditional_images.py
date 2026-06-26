"""Build ControlNet condition images (the "room circles") from flat layout masks.

For every room color region in a flat layout image, this script finds the largest
inscribed circle of each connected component and draws a filled circle of the same
color on a black canvas. Walls, doors, external area and very small regions are
skipped. The result is a simplified "bubble diagram" used as a ControlNet condition.

Usage:
    python conditional_images.py --input_dir <flat_layouts> --output_dir <conditions>
"""
import argparse
import os

import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt

# RGB colors that are NOT rooms and should be skipped (walls, doors, background).
COLORS_TO_SKIP = {
    (31, 31, 31),      # External area
    (95, 95, 95),      # Exterior / interior wall
    (200, 227, 145),   # Front door
    (238, 179, 142),   # Interior door
    (0, 0, 0),         # Black background
}

# Minimum inscribed-circle area (in pixels) for a region to be kept.
MIN_CIRCLE_AREA = 20


def process_dataset(dataset_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    plan_name_list = os.listdir(dataset_dir)
    total_plans = len(plan_name_list)
    print(f"Found {total_plans} plans to process.")

    for index, plan_name in enumerate(plan_name_list):
        output_path = os.path.join(output_dir, plan_name)

        if os.path.exists(output_path):
            print(f"Skipping {plan_name} as it already exists.")
            continue

        plan_path = os.path.join(dataset_dir, plan_name)

        try:
            plan_np = np.array(Image.open(plan_path).convert("RGB"))
            output_image = np.zeros_like(plan_np)

            unique_colors = np.unique(plan_np.reshape(-1, 3), axis=0)

            for color in unique_colors:
                if tuple(color) in COLORS_TO_SKIP or np.all(color > 200):
                    continue

                mask = np.all(plan_np == color, axis=-1)
                if not np.any(mask):
                    continue

                num_labels, labels = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
                for label_id in range(1, num_labels):
                    component_mask = labels == label_id

                    # Distance transform: the maximum value is the largest inscribed-circle radius.
                    distance_map = distance_transform_edt(component_mask)
                    radius = np.max(distance_map)
                    if radius < 1:
                        continue

                    # Discard regions whose inscribed circle is too small.
                    if np.pi * (radius ** 2) < MIN_CIRCLE_AREA:
                        continue

                    # Centroid (geometric center) of the region.
                    m = cv2.moments(component_mask.astype(np.uint8))
                    if m['m00'] > 0:
                        cx = m['m10'] / m['m00']
                        cy = m['m01'] / m['m00']
                    else:
                        # Degenerate case: use the bounding-box center.
                        x, y, w, h = cv2.boundingRect(component_mask.astype(np.uint8))
                        cx = x + w / 2.0
                        cy = y + h / 2.0

                    # Candidate points with the maximum radius (there may be several).
                    ys, xs = np.where(distance_map == radius)
                    if len(xs) == 0:
                        continue

                    cy_i = int(round(cy))
                    cx_i = int(round(cx))
                    cy_i = np.clip(cy_i, 0, distance_map.shape[0] - 1)
                    cx_i = np.clip(cx_i, 0, distance_map.shape[1] - 1)

                    if distance_map[cy_i, cx_i] == radius:
                        # The centroid itself attains the maximum radius.
                        center = (cx_i, cy_i)
                    else:
                        # Otherwise, pick the max-radius candidate closest to the centroid.
                        candidates = np.column_stack((ys, xs))  # (row, col)
                        dists2 = (candidates[:, 0] - cy) ** 2 + (candidates[:, 1] - cx) ** 2
                        best_idx = int(np.argmin(dists2))
                        best_row, best_col = candidates[best_idx]
                        center = (int(best_col), int(best_row))

                    cv2.circle(output_image, center, int(radius), color.tolist(), -1)

            Image.fromarray(output_image).save(output_path)

            if (index + 1) % 100 == 0 or index == total_plans - 1:
                print(f"Processed {index + 1}/{total_plans} plans.", flush=True)
        except Exception as e:
            print(f"Failed to process {plan_name}: {e}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate ControlNet 'room circle' condition images from flat layout masks."
    )
    parser.add_argument("--input_dir", required=True, help="Directory of input flat layout PNGs.")
    parser.add_argument("--output_dir", required=True, help="Directory to write condition images to.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    process_dataset(args.input_dir, args.output_dir)
