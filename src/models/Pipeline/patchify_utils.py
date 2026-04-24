#!/usr/bin/env python3
"""
Patchify helper.

- keeps __rXXX_cYYY in filename
- short patch names to avoid Windows path-length issues
- returns patches output folder
"""

import glob
import cv2
import os
import os.path as osp
from patchify import patchify


def patchify_index_grouped(
    source_directory,
    patch_h,
    patch_w,
    step_h=None,
    step_w=None,
    cover_edges=False
):
    """
    Saves ALL patches from ALL images into:
        <source_directory>/patches_rtor

    Filenames:
        p__rXXX_cYYY.png
    """
    base_out = osp.join(source_directory, "patches_rtor")
    os.makedirs(base_out, exist_ok=True)

    image_files = sorted(
        glob.glob(osp.join(source_directory, "*.jpg")) +
        glob.glob(osp.join(source_directory, "*.jpeg")) +
        glob.glob(osp.join(source_directory, "*.png"))
    )

    step_h = patch_h if step_h is None else step_h
    step_w = patch_w if step_w is None else step_w

    for file_path in image_files:
        print(f"Patching: {file_path}")

        img = cv2.imread(file_path)
        if img is None:
            print(f"⚠️ Could not read image: {file_path}")
            continue

        H, W = img.shape[:2]
        if H < patch_h or W < patch_w:
            print(f"⚠️ Skipping small image (H{H}xW{W}): {file_path}")
            continue

        # grayscale or color
        if img.ndim == 2:
            ch = 1
        else:
            ch = img.shape[2]

        if not cover_edges:
            # patchify needs matching dimensions in the window shape
            if ch == 1:
                patches = patchify(img, (patch_h, patch_w), step=(step_h, step_w))
                rows, cols = patches.shape[0], patches.shape[1]

                for i in range(rows):
                    for j in range(cols):
                        patch_img = patches[i, j]
                        out_name = f"p__r{i:03d}_c{j:03d}.png"
                        out_path = osp.join(base_out, out_name)

                        ok = cv2.imwrite(out_path, patch_img)
                        if not ok:
                            raise RuntimeError(f"Could not write patch: {out_path}")

            else:
                patches = patchify(img, (patch_h, patch_w, ch), step=(step_h, step_w, ch))
                rows, cols = patches.shape[0], patches.shape[1]

                for i in range(rows):
                    for j in range(cols):
                        patch_img = patches[i, j, 0, :, :, :]
                        out_name = f"p__r{i:03d}_c{j:03d}.png"
                        out_path = osp.join(base_out, out_name)

                        ok = cv2.imwrite(out_path, patch_img)
                        if not ok:
                            raise RuntimeError(f"Could not write patch: {out_path}")

        else:
            i_starts = list(range(0, H - patch_h + 1, step_h))
            j_starts = list(range(0, W - patch_w + 1, step_w))

            if not i_starts:
                i_starts = [0]
            if not j_starts:
                j_starts = [0]

            if i_starts[-1] != H - patch_h:
                i_starts.append(H - patch_h)
            if j_starts[-1] != W - patch_w:
                j_starts.append(W - patch_w)

            for r, i0 in enumerate(i_starts):
                for c, j0 in enumerate(j_starts):
                    patch_img = img[i0:i0 + patch_h, j0:j0 + patch_w]
                    out_name = f"p__r{r:03d}_c{c:03d}.png"
                    out_path = osp.join(base_out, out_name)

                    ok = cv2.imwrite(out_path, patch_img)
                    if not ok:
                        raise RuntimeError(f"Could not write patch: {out_path}")

    print("\n✅ Done. All patches saved to:")
    print(f"   {base_out}")
    return base_out


if __name__ == "__main__":
    src = r"C:\Users\Admin\OneDrive - radometech.com\Desktop\Apollo\VIT+Autoencoder\Dataset_Captured\165_80_R14_85T_AMZ_4G\165_80_R14_85T_AMZ_4G\Defs_9020081560\250500041_Sidewall\trial"
    patchify_index_grouped(src, 1630, 1024, 1630, 1024, cover_edges=True)