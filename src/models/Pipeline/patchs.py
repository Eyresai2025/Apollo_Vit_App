import glob
import cv2
import os
import os.path as osp
import numpy as np
 
 
def patchify_index_grouped(source_path, patch_h, patch_w,
                           step_h=None, step_w=None, cover_edges=False):

    if osp.isdir(source_path):
        base_out = osp.join(source_path, "patches_rtor")
    else:
        base_out = osp.join(osp.dirname(source_path), "patches_rtor")

    os.makedirs(base_out, exist_ok=True)

    if osp.isfile(source_path):
        image_files = [source_path]
    else:
        image_files = sorted(
            glob.glob(osp.join(source_path, "*.jpg")) +
            glob.glob(osp.join(source_path, "*.jpeg")) +
            glob.glob(osp.join(source_path, "*.png"))
        )

    if len(image_files) == 0:
        print("❌ No images found!")
        return None

    for file_path in image_files:
        print(f"Patching: {file_path}")

        img = cv2.imread(file_path)
        if img is None:
            print(f"⚠️ Could not read image: {file_path}")
            continue

        H, W = img.shape[:2]

        if H < patch_h or W < patch_w:
            print(f"⚠️ Skipping small image (H{H} x W{W})")
            continue

        sh = patch_h if step_h is None else step_h
        sw = patch_w if step_w is None else step_w

        filename_base, ext = osp.splitext(osp.basename(file_path))
        ext = ext.lower()

        if not cover_edges:
            i_starts = list(range(0, H - patch_h + 1, sh))
            j_starts = list(range(0, W - patch_w + 1, sw))
        else:
            i_starts = list(range(0, H - patch_h + 1, sh))
            j_starts = list(range(0, W - patch_w + 1, sw))

            if i_starts[-1] != H - patch_h:
                i_starts.append(H - patch_h)

            if j_starts[-1] != W - patch_w:
                j_starts.append(W - patch_w)

        for r, i0 in enumerate(i_starts):
            for c, j0 in enumerate(j_starts):
                patch_img = img[i0:i0 + patch_h, j0:j0 + patch_w]

                out_name = f"{filename_base}__r{r:03d}_c{c:03d}{ext}"
                out_path = osp.join(base_out, out_name)
                cv2.imwrite(out_path, patch_img)

    print("\n✅ Done! Patches saved in:")
    print(base_out)
    return base_out
 
# ==============================
# RUN SCRIPT
# ==============================
if __name__ == "__main__":
 
    source_path = r"C:\Users\eyres\Downloads\Dataset_Ceat\_tmp_strips\test"
 
    patchify_index_grouped(
        source_path,
        patch_h=200,
        patch_w=200,
        step_h=200,
        step_w=200,
        cover_edges=True   # Recommended
    )
 