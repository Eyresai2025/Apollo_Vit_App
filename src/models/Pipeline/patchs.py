import os
import shutil
from pathlib import Path

import cv2


VALID_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def _list_images(folder_path):
    if not os.path.isdir(folder_path):
        raise FileNotFoundError(f"Image folder not found: {folder_path}")

    paths = []

    for name in sorted(os.listdir(folder_path)):
        full_path = os.path.join(folder_path, name)

        if os.path.isfile(full_path) and name.lower().endswith(VALID_EXTS):
            paths.append(full_path)

    return paths


def _positions(length, patch_size, step, cover_edges=True):
    """
    Returns start positions along one axis.
    If cover_edges=True, adds the last possible position so image edge is covered.
    """
    if length <= patch_size:
        return [0]

    positions = list(range(0, length - patch_size + 1, step))

    if cover_edges:
        last_pos = length - patch_size
        if positions[-1] != last_pos:
            positions.append(last_pos)

    return positions


def patchify_index_grouped(
    input_dir,
    patch_h=200,
    patch_w=200,
    step_h=200,
    step_w=200,
    cover_edges=True,
    output_dir=None,
):
    """
    Patchifies images inside input_dir.

    Expected usage from inference files:
        patches_dir = patchify_index_grouped(
            single_crop_dir,
            patch_h=BIG_PATCH_H,
            patch_w=BIG_PATCH_W,
            step_h=BIG_STEP_H,
            step_w=BIG_STEP_W,
            cover_edges=COVER_EDGES,
        )

    Output:
        input_dir/patches_rtor/
            p__r000_c000.png
            p__r000_c001.png
            ...

    Returns:
        patches_dir
    """

    input_dir = str(input_dir)

    if output_dir is None:
        output_dir = os.path.join(input_dir, "patches_rtor")

    output_dir = str(output_dir)

    if os.path.isdir(output_dir):
        shutil.rmtree(output_dir)

    os.makedirs(output_dir, exist_ok=True)

    image_paths = _list_images(input_dir)

    if not image_paths:
        raise RuntimeError(f"No images found to patchify in: {input_dir}")

    patch_count = 0

    for img_path in image_paths:
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

        if img is None:
            print(f"[PATCH][WARN] Cannot read image: {img_path}")
            continue

        h, w = img.shape[:2]

        y_positions = _positions(
            length=h,
            patch_size=patch_h,
            step=step_h,
            cover_edges=cover_edges,
        )

        x_positions = _positions(
            length=w,
            patch_size=patch_w,
            step=step_w,
            cover_edges=cover_edges,
        )

        for r, y in enumerate(y_positions):
            for c, x in enumerate(x_positions):
                patch = img[y:y + patch_h, x:x + patch_w]

                if patch.shape[0] != patch_h or patch.shape[1] != patch_w:
                    continue

                patch_name = f"p__r{r:03d}_c{c:03d}.png"
                patch_path = os.path.join(output_dir, patch_name)

                cv2.imwrite(patch_path, patch)
                patch_count += 1

    if patch_count == 0:
        raise RuntimeError(
            f"No patches created from {input_dir}. "
            f"Check image size and patch size."
        )

    print(f"[PATCH] Created {patch_count} patches -> {output_dir}")

    return output_dir