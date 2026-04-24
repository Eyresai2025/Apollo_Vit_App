#!/usr/bin/env python3
"""
RUN:
    python prepare_dataset_from_raw_sidewall.py

Pipeline:
    raw image
      -> polarizer(raw_bgr) -> polarized/preprocessed gray
      -> align_and_crop_to_reference(pre_gray, reference_gray, ...)
      -> patchify cropped strip
      -> save patches to train/test good/anomalous

Reference:
    - first image from TRAIN_RAW_DIR is used as the single common reference
    - no separate golden folder required

Routing:
    TRAIN_RAW_DIR:
        all images -> train/good

    TEST_RAW_DIR:
        if path contains "good"      -> test/good
        if path contains "defect"    -> test/anomalous
        if path contains "anomalous" -> test/anomalous
"""

import os
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import cv2
import torch
from polarizer import polarizer
from R_Detection_align_crop import build_r_detector, align_and_crop_to_reference, detect_and_crop_gray
from patchify_utils import patchify_index_grouped
from PIL import Image, ImageEnhance

# =========================
# CONFIG
# =========================

TRAIN_RAW_DIR = r"C:\Users\eyres\OneDrive - radometech.com\Desktop\Apollo\VIT+Autoencoder_AE\VIT_Training\Dataset\Train\Good"
TEST_RAW_DIR  = r"C:\Users\eyres\OneDrive - radometech.com\Desktop\Apollo\VIT+Autoencoder_AE\VIT_Training\Dataset\Test"

DATASET_ROOT  = r"C:\Users\eyres\OneDrive - radometech.com\Desktop\Apollo\VIT+Autoencoder_AE\VIT_Training\PreparedDataset"

YOLO_R_PATH   = r"C:\Users\eyres\Downloads\R_Detection.pt"
DEVICE        = "cuda"

USE_ALIGNMENT = True

SLICE_H       = 1630
SLICE_W       = 1024
CONF_THRES_R  = 0.3

RESIZE_CROP_TO = (2000, 10000)   # (width, height)

BIG_PATCH_H   = 200
BIG_PATCH_W   = 200
BIG_STEP_H    = 200
BIG_STEP_W    = 200
COVER_EDGES   = True

TMP_STRIPS_ROOT = os.path.join(DATASET_ROOT, "_tmp_strips")

IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

# =========================
# AUGMENTATION CONFIG
# =========================
APPLY_AUGMENTATION = True

# since you asked "all patches", all are enabled
AUGMENT_TRAIN_GOOD = True
AUGMENT_TEST_GOOD = False
AUGMENT_TEST_ANOMALOUS = False

# True  -> save inside folder/Augmentation/
# False -> save directly into the same folder
AUGMENT_IN_SUBFOLDER = False
AUGMENT_SUBFOLDER_NAME = "Augmentation"

BRIGHTNESS_UP_FACTOR = 1.5
BRIGHTNESS_DOWN_FACTOR = 0.7


# ============================================================
# HELPERS
# ============================================================

def _list_images(root_dir: Optional[str]) -> List[str]:
    if not root_dir:
        return []

    root_path = Path(root_dir)
    if not root_path.exists():
        return []

    paths = []
    for r, _, files in os.walk(root_dir):
        for f in files:
            if f.lower().endswith(IMG_EXTS):
                paths.append(os.path.join(r, f))
    return sorted(paths)


def _parse_patch_filename(fname: str):
    """
    Expect patch filenames like:
        base__r003_c005.png
    Returns:
        (base, r, c, ext) or None
    """
    stem, ext = os.path.splitext(os.path.basename(fname))
    if "__r" not in stem:
        return None
    try:
        base, rc = stem.split("__r")
        r_str, c_str = rc.split("_c")
        return base, int(r_str), int(c_str), ext
    except Exception:
        return None
    
def folder_has_files(path_str: Optional[str]) -> bool:
    if not path_str:
        return False
    p = Path(path_str)
    return p.exists() and any(x.is_file() for x in p.rglob("*"))

def infer_label_from_path(path: str) -> Optional[str]:
    """
    Safer label inference from path parts + stem.
    """
    p = Path(path)
    parts = [x.lower() for x in p.parts]
    stem = p.stem.lower()

    if "good" in parts or "good" in stem:
        return "good"
    if "defect" in parts or "defect" in stem:
        return "anomalous"
    if "anomalous" in parts or "anomalous" in stem:
        return "anomalous"

    return None

def _safe_aug_ext(ext: str) -> str:
    ext = ext.lower()
    if ext in [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]:
        return ext
    return ".png"


def augment_patch_folder(input_folder: str) -> int:
    """
    Applies augmentation to all image patches inside one final patch folder.

    Augmentations:
        - horizontal flip
        - vertical flip
        - brighter
        - darker

    Returns:
        number of augmented images saved
    """
    if not os.path.isdir(input_folder):
        print(f"[AUG][SKIP] Folder not found: {input_folder}")
        return 0

    if AUGMENT_IN_SUBFOLDER:
        output_folder = os.path.join(input_folder, AUGMENT_SUBFOLDER_NAME)
    else:
        output_folder = input_folder

    os.makedirs(output_folder, exist_ok=True)

    files = []
    for fname in os.listdir(input_folder):
        fpath = os.path.join(input_folder, fname)
        if os.path.isfile(fpath) and fname.lower().endswith(IMG_EXTS):
            files.append(fname)

    files = sorted(files)
    print(f"[AUG] Found {len(files)} base patches in: {input_folder}")

    saved_count = 0

    for filename in files:
        img_path = os.path.join(input_folder, filename)

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"[AUG][WARN] Could not open {img_path}: {e}")
            continue

        base_name, ext = os.path.splitext(filename)
        ext = _safe_aug_ext(ext)

        try:
            # Flip horizontally
            horizontal_flip = image.transpose(Image.FLIP_LEFT_RIGHT)
            horizontal_flip.save(os.path.join(output_folder, f"{base_name}_fH{ext}"))
            saved_count += 1

            # Flip vertically
            vertical_flip = image.transpose(Image.FLIP_TOP_BOTTOM)
            vertical_flip.save(os.path.join(output_folder, f"{base_name}_fpV{ext}"))
            saved_count += 1

            # Increase brightness
            enhancer = ImageEnhance.Brightness(image)
            brighter = enhancer.enhance(BRIGHTNESS_UP_FACTOR)
            brighter.save(os.path.join(output_folder, f"{base_name}_bt{ext}"))
            saved_count += 1

            # Decrease brightness
            darker = enhancer.enhance(BRIGHTNESS_DOWN_FACTOR)
            darker.save(os.path.join(output_folder, f"{base_name}_d{ext}"))
            saved_count += 1

        except Exception as e:
            print(f"[AUG][WARN] Augmentation failed for {img_path}: {e}")
            continue

    print(f"[AUG] Saved {saved_count} augmented patches -> {output_folder}")
    return saved_count


def run_post_patch_augmentation(
    train_good_dir: str,
    test_good_dir: str,
    test_anom_dir: str,
):
    """
    Runs augmentation after all final patches are already written.
    """
    if not APPLY_AUGMENTATION:
        print("[AUG] APPLY_AUGMENTATION=False -> skipping augmentation")
        return

    total_saved = 0
    print("\n[AUG] Starting augmentation on final patch folders...")

    if AUGMENT_TRAIN_GOOD:
        total_saved += augment_patch_folder(train_good_dir)

    if AUGMENT_TEST_GOOD:
        total_saved += augment_patch_folder(test_good_dir)

    if AUGMENT_TEST_ANOMALOUS:
        total_saved += augment_patch_folder(test_anom_dir)

    print(f"[AUG] DONE | Total augmented patches saved: {total_saved}")


# ============================================================
# REFERENCE FROM FIRST TRAIN IMAGE
# ============================================================

def prepare_reference_from_first_train_image(train_raw_dir: str) -> Tuple[Optional[any], Optional[str]]:
    """
    Uses the first image from TRAIN_RAW_DIR as the single common reference.
    Returns:
        reference_gray, reference_raw_path
    """
    raw_paths = _list_images(train_raw_dir)
    if not raw_paths:
        print(f"[REF] No images found in TRAIN_RAW_DIR: {train_raw_dir}")
        return None, None

    ref_raw_path = raw_paths[0]
    print(f"[REF] Using first train image as reference: {ref_raw_path}")

    raw_bgr = cv2.imread(ref_raw_path)
    if raw_bgr is None:
        print(f"[REF][ERR] Cannot read reference image: {ref_raw_path}")
        return None, None

    reference_gray = polarizer(raw_bgr)
    if reference_gray is None:
        print(f"[REF][ERR] polarizer failed for reference image: {ref_raw_path}")
        return None, None

    return reference_gray, ref_raw_path


# ============================================================
# CORE PER-IMAGE PIPELINE
# ============================================================

def process_one_raw_image(
    raw_bgr,
    r_detector,
    reference_gray,
    use_alignment=True,
):
    """
    raw -> preprocess -> align_and_crop_to_reference -> final cropped strip
    """
    if raw_bgr is None:
        return None, {"status": "fail", "reason": "raw_none"}

    pre_gray = polarizer(raw_bgr)
    if pre_gray is None:
        return None, {"status": "fail", "reason": "preprocess_failed"}

    if use_alignment and reference_gray is not None:
        crop_bgr, aligned_bgr, meta = align_and_crop_to_reference(
            image_bgr=pre_gray,
            reference_bgr=reference_gray,
            det_model=r_detector,
            slice_h=SLICE_H,
            slice_w=SLICE_W,
            target_size=RESIZE_CROP_TO,
        )
        return crop_bgr, meta

    crop_bgr, top_offset, detections = detect_and_crop_gray(
        pre_gray,
        r_detector,
        SLICE_H,
        SLICE_W,
    )

    if crop_bgr is None:
        return None, {
            "status": "fail",
            "reason": "crop_failed_no_alignment",
            "detections": detections,
        }

    if RESIZE_CROP_TO is not None:
        target_w, target_h = RESIZE_CROP_TO
        crop_bgr = cv2.resize(
            crop_bgr,
            (target_w, target_h),
            interpolation=cv2.INTER_LINEAR
        )

    meta = {
        "status": "ok",
        "mode": "no_alignment",
        "crop_top_offset": top_offset,
        "crop_r_detections": detections,
        "final_h": int(crop_bgr.shape[0]),
        "final_w": int(crop_bgr.shape[1]),
    }
    return crop_bgr, meta


# ============================================================
# TRAIN SPLIT
# ============================================================

def process_split_train(
    split_name: str,
    raw_root: str,
    out_patches_dir: str,
    r_detector,
    use_alignment: bool,
    reference_gray,
):
    """
    All train raw images are GOOD.
    """
    os.makedirs(out_patches_dir, exist_ok=True)

    tmp_strips_dir = os.path.join(TMP_STRIPS_ROOT, split_name)
    os.makedirs(tmp_strips_dir, exist_ok=True)

    raw_paths = _list_images(raw_root)
    print(f"\n[TRAIN] Raw images found: {len(raw_paths)}")

    for raw_path in raw_paths:
        name = Path(raw_path).stem
        print(f"   [TRAIN] {name}")

        raw_bgr = cv2.imread(raw_path)
        if raw_bgr is None:
            print(f"   [SKIP] Cannot read {raw_path}")
            continue

        crop_bgr, meta = process_one_raw_image(
            raw_bgr=raw_bgr,
            r_detector=r_detector,
            reference_gray=reference_gray,
            use_alignment=use_alignment,
        )

        if crop_bgr is None:
            print(f"   [SKIP] Failed: {meta}")
            continue

        strip_path = os.path.join(tmp_strips_dir, f"{name}_crop.png")
        cv2.imwrite(strip_path, crop_bgr)

    print(f"[TRAIN] Patchifying strips from: {tmp_strips_dir}")
    patches_dir = patchify_index_grouped(
        tmp_strips_dir,
        patch_h=BIG_PATCH_H,
        patch_w=BIG_PATCH_W,
        step_h=BIG_STEP_H,
        step_w=BIG_STEP_W,
        cover_edges=COVER_EDGES,
    )

    patch_paths = _list_images(patches_dir)
    print(f"[TRAIN] Patches found: {len(patch_paths)}")

    for p in patch_paths:
        img = cv2.imread(p)
        if img is None:
            print(f"   [WARN] Cannot read patch: {p}")
            continue

        out_path = os.path.join(out_patches_dir, os.path.basename(p))
        ok = cv2.imwrite(out_path, img)
        if not ok:
            raise RuntimeError(f"Could not write patch: {out_path}")

    print(f"[TRAIN] DONE -> {out_patches_dir}")


# ============================================================
# TEST SPLIT
# ============================================================

def process_split_test(
    split_name: str,
    raw_root: str,
    out_good_dir: str,
    out_anom_dir: str,
    r_detector,
    use_alignment: bool,
    reference_gray,
):
    """
    Mixed data -> route to good / anomalous
    """
    os.makedirs(out_good_dir, exist_ok=True)
    os.makedirs(out_anom_dir, exist_ok=True)

    tmp_strips_dir = os.path.join(TMP_STRIPS_ROOT, split_name)
    os.makedirs(tmp_strips_dir, exist_ok=True)

    raw_paths = _list_images(raw_root)
    print(f"\n[TEST] Raw images found: {len(raw_paths)}")

    strip_label_map: Dict[str, str] = {}

    for raw_path in raw_paths:
        name = Path(raw_path).stem
        label = infer_label_from_path(raw_path)

        if label is None:
            print(f"   [WARN] Could not infer label from: {raw_path}")
            continue

        print(f"   [TEST] {name} | label={label}")

        raw_bgr = cv2.imread(raw_path)
        if raw_bgr is None:
            print(f"   [SKIP] Cannot read {raw_path}")
            continue

        crop_bgr, meta = process_one_raw_image(
            raw_bgr=raw_bgr,
            r_detector=r_detector,
            reference_gray=reference_gray,
            use_alignment=use_alignment,
        )

        if crop_bgr is None:
            print(f"   [SKIP] Failed: {meta}")
            continue

        strip_stem = f"{name}_crop"
        strip_path = os.path.join(tmp_strips_dir, f"{strip_stem}.png")
        cv2.imwrite(strip_path, crop_bgr)

        strip_label_map[strip_stem] = label

    print(f"[TEST] Patchifying strips from: {tmp_strips_dir}")
    patches_dir = patchify_index_grouped(
        tmp_strips_dir,
        patch_h=BIG_PATCH_H,
        patch_w=BIG_PATCH_W,
        step_h=BIG_STEP_H,
        step_w=BIG_STEP_W,
        cover_edges=COVER_EDGES,
    )

    patch_paths = _list_images(patches_dir)
    print(f"[TEST] Patches found: {len(patch_paths)}")

    for p in patch_paths:
        parsed = _parse_patch_filename(p)
        if not parsed:
            print(f"   [WARN] Could not parse patch filename: {p}")
            continue

        base, _, _, _ = parsed
        label = strip_label_map.get(base)

        if label is None:
            print(f"   [WARN] No label found for strip base: {base}")
            continue

        img = cv2.imread(p)
        if img is None:
            print(f"   [WARN] Cannot read patch: {p}")
            continue

        out_name = os.path.basename(p)

        if label == "good":
            out_path = os.path.join(out_good_dir, out_name)
        else:
            out_path = os.path.join(out_anom_dir, out_name)

        ok = cv2.imwrite(out_path, img)
        if not ok:
            raise RuntimeError(f"Could not write patch: {out_path}")

    print(f"[TEST] DONE")
    print(f"   Good      -> {out_good_dir}")
    print(f"   Anomalous -> {out_anom_dir}")


# ============================================================
# MAIN
# ============================================================

def main():
    device = DEVICE
    if device == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA not available -> switching to CPU")
        device = "cpu"
    print("[INFO] Device:", device)

    r_detector = build_r_detector(YOLO_R_PATH, conf=CONF_THRES_R, device=device)
    print("[INFO] R-detector ready")

    reference_gray = None
    use_alignment_effective = USE_ALIGNMENT

    if USE_ALIGNMENT:
        reference_gray, reference_path = prepare_reference_from_first_train_image(TRAIN_RAW_DIR)

        if reference_gray is None:
            print("[ALIGN][WARN] Could not prepare reference from TRAIN_RAW_DIR. Alignment disabled.")
            use_alignment_effective = False
        else:
            print(f"[ALIGN] Common reference ready from: {reference_path}")
    else:
        print("[ALIGN] USE_ALIGNMENT=False")

    train_good_dir = os.path.join(DATASET_ROOT, "train", "good")
    test_good_dir  = os.path.join(DATASET_ROOT, "test", "good")
    test_anom_dir  = os.path.join(DATASET_ROOT, "test", "anomalous")

    os.makedirs(train_good_dir, exist_ok=True)

    process_split_train(
        split_name="train",
        raw_root=TRAIN_RAW_DIR,
        out_patches_dir=train_good_dir,
        r_detector=r_detector,
        use_alignment=use_alignment_effective,
        reference_gray=reference_gray,
    )

    if folder_has_files(TEST_RAW_DIR):
        os.makedirs(test_good_dir, exist_ok=True)
        os.makedirs(test_anom_dir, exist_ok=True)

        process_split_test(
            split_name="test",
            raw_root=TEST_RAW_DIR,
            out_good_dir=test_good_dir,
            out_anom_dir=test_anom_dir,
            r_detector=r_detector,
            use_alignment=use_alignment_effective,
            reference_gray=reference_gray,
        )
    else:
        print("[TEST] No test images found. Skipping test dataset preparation.")

    run_post_patch_augmentation(
        train_good_dir=train_good_dir,
        test_good_dir=test_good_dir,
        test_anom_dir=test_anom_dir,
    )


    print("\n✅ ALL DONE- SIDE WALL 1")
    print("Reference used :", "first image from TRAIN_RAW_DIR")
    print("Dataset root   :", DATASET_ROOT)
    print("Train good     :", train_good_dir)

    if folder_has_files(TEST_RAW_DIR):
        print("Test good      :", test_good_dir)
        print("Test anomalous :", test_anom_dir)
    else:
        print("Test Split : skipped")

if __name__ == "__main__":
    main()