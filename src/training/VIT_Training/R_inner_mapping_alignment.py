import os
import cv2
import uuid
import numpy as np
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction


def build_r_detector(model_path, conf=0.4, device="cuda"):
    return AutoDetectionModel.from_pretrained(
        model_type="yolov8",
        model_path=model_path,
        confidence_threshold=conf,
        device=device
    )


def _run_sahi_on_image(image_bgr, det_model, slice_h, slice_w,
                       overlap_h=0.0, overlap_w=0.0):
    tmp_path = os.path.join(os.getcwd(), f"__tmp_rdet_{uuid.uuid4().hex}.png")
    ok = cv2.imwrite(tmp_path, image_bgr)
    if not ok:
        raise RuntimeError(f"Failed to save temp image: {tmp_path}")

    try:
        result = get_sliced_prediction(
            tmp_path,
            det_model,
            slice_height=slice_h,
            slice_width=slice_w,
            overlap_height_ratio=overlap_h,
            overlap_width_ratio=overlap_w,
            auto_slice_resolution=False,
        )
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    return result


def detect_two_r_from_image(image_bgr, det_model, slice_h, slice_w):
    """
    Detect top 2 R markers from an image array.
    Returns list of:
        (x, y, w, h, cx, cy)
    sorted by y-center
    """
    if image_bgr is None:
        return []

    if image_bgr.ndim == 2:
        image_bgr = cv2.cvtColor(image_bgr, cv2.COLOR_GRAY2BGR)

    result = _run_sahi_on_image(
        image_bgr,
        det_model,
        slice_h,
        slice_w,
        overlap_h=0.0,
        overlap_w=0.0
    )

    r = []
    for ann in result.to_coco_annotations():
        if ann.get("category_id", None) != 0:
            continue

        x, y, w, h = map(float, ann["bbox"])
        cx = x + w / 2.0
        cy = y + h / 2.0
        r.append((x, y, w, h, cx, cy))

    r = sorted(r, key=lambda v: v[5])   # sort by y-center
    return r[:2]


def transform_point(M, px, py):
    p = np.array([px, py, 1], dtype=np.float32)
    q = M @ p
    return int(round(q[0])), int(round(q[1]))


def get_reference_r_band(reference_bgr, det_model, slice_h, slice_w):
    """
    Detect R on reference image ONCE and return fixed crop band info.

    Returns:
        dict with:
            status
            ref_r
            y1
            y2
            ref_h
            ref_w
    """
    if reference_bgr is None:
        return {
            "status": "fail",
            "reason": "reference_none"
        }

    if reference_bgr.ndim == 2:
        reference_bgr = cv2.cvtColor(reference_bgr, cv2.COLOR_GRAY2BGR)

    ref_r = detect_two_r_from_image(reference_bgr, det_model, slice_h, slice_w)
    if len(ref_r) < 2:
        return {
            "status": "fail",
            "reason": "reference_less_than_2_r",
            "ref_r": ref_r
        }

    ref_r = sorted(ref_r, key=lambda v: v[5])

    # use bbox top-y exactly like your old fixed-band crop script
    y1 = int(round(ref_r[0][1]))
    y2 = int(round(ref_r[1][1]))

    H, W = reference_bgr.shape[:2]
    y1 = max(0, min(y1, H - 1))
    y2 = max(0, min(y2, H - 1))

    if y2 <= y1:
        return {
            "status": "fail",
            "reason": "invalid_reference_crop_band",
            "ref_r": ref_r,
            "y1": y1,
            "y2": y2,
            "ref_h": H,
            "ref_w": W,
        }

    return {
        "status": "ok",
        "ref_r": ref_r,
        "y1": y1,
        "y2": y2,
        "ref_h": H,
        "ref_w": W,
    }


def crop_between_fixed_y(image_bgr, y1, y2, target_size=None):
    """
    Crop image using FIXED y1,y2 band.
    """
    if image_bgr is None:
        return None

    if image_bgr.ndim == 2:
        image_bgr = cv2.cvtColor(image_bgr, cv2.COLOR_GRAY2BGR)

    H, W = image_bgr.shape[:2]

    y1 = max(0, min(int(round(y1)), H - 1))
    y2 = max(0, min(int(round(y2)), H - 1))

    if y2 <= y1:
        return None

    crop = image_bgr[y1:y2, 0:W]
    if crop.size == 0:
        return None

    if target_size is not None:
        target_w, target_h = target_size
        crop = cv2.resize(crop, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

    return crop


def align_and_crop_to_reference_fixed_band(
    image_bgr,
    reference_bgr,
    det_model,
    slice_h,
    slice_w,
    target_size=None,
):
    """
    New behavior:
    1) detect R on source image
    2) detect R on reference image ONCE
    3) align source to reference using reference R centers
    4) crop aligned image using FIXED reference y1,y2
    5) resize final crop

    Returns:
        crop_resized_bgr, aligned_bgr, meta
    """
    if image_bgr is None:
        return None, None, {"status": "fail", "reason": "image_none"}
    if reference_bgr is None:
        return None, None, {"status": "fail", "reason": "reference_none"}

    if image_bgr.ndim == 2:
        image_bgr = cv2.cvtColor(image_bgr, cv2.COLOR_GRAY2BGR)
    if reference_bgr.ndim == 2:
        reference_bgr = cv2.cvtColor(reference_bgr, cv2.COLOR_GRAY2BGR)

    # detect source R
    raw_r = detect_two_r_from_image(image_bgr, det_model, slice_h, slice_w)
    if len(raw_r) < 2:
        return None, None, {
            "status": "fail",
            "reason": "source_less_than_2_r",
            "raw_r": raw_r
        }

    # detect reference R ONCE + fixed crop band
    ref_info = get_reference_r_band(reference_bgr, det_model, slice_h, slice_w)
    if ref_info["status"] != "ok":
        return None, None, ref_info

    ref_r = ref_info["ref_r"]
    fixed_y1 = ref_info["y1"]
    fixed_y2 = ref_info["y2"]

    ref_pts = np.array([
        [ref_r[0][4], ref_r[0][5]],
        [ref_r[1][4], ref_r[1][5]],
    ], dtype=np.float32)

    src_pts = np.array([
        [raw_r[0][4], raw_r[0][5]],
        [raw_r[1][4], raw_r[1][5]],
    ], dtype=np.float32)

    M, _ = cv2.estimateAffinePartial2D(src_pts, ref_pts)
    if M is None:
        return None, None, {
            "status": "fail",
            "reason": "affine_estimation_failed",
            "raw_r": raw_r,
            "ref_r": ref_r,
        }

    # align onto reference canvas size
    ref_h, ref_w = reference_bgr.shape[:2]
    aligned_bgr = cv2.warpAffine(
        image_bgr,
        M,
        (ref_w, ref_h),
        flags=cv2.INTER_LINEAR,
        borderValue=(0, 0, 0),
    )

    # FIXED crop from reference band
    crop_bgr = crop_between_fixed_y(
        aligned_bgr,
        fixed_y1,
        fixed_y2,
        target_size=target_size,
    )

    if crop_bgr is None:
        return None, aligned_bgr, {
            "status": "fail",
            "reason": "crop_failed_fixed_reference_band",
            "fixed_y1": fixed_y1,
            "fixed_y2": fixed_y2,
        }

    aligned_r = []
    for x, y, w, h, cx, cy in raw_r:
        nx, ny = transform_point(M, x, y)
        ncx, ncy = transform_point(M, cx, cy)
        aligned_r.append((nx, ny, ncx, ncy))

    meta = {
        "status": "ok",
        "raw_r": raw_r,
        "ref_r": ref_r,
        "aligned_r": aligned_r,
        "fixed_crop_y1": fixed_y1,
        "fixed_crop_y2": fixed_y2,
        "ref_h": ref_h,
        "ref_w": ref_w,
        "final_h": int(crop_bgr.shape[0]),
        "final_w": int(crop_bgr.shape[1]),
    }
    return crop_bgr, aligned_bgr, meta