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

def _run_sahi_on_image(image_bgr, det_model, slice_h, slice_w):
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
            overlap_height_ratio=0,
            overlap_width_ratio=0,
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
    """
    if image_bgr is None:
        return []

    if image_bgr.ndim == 2:
        image_bgr = cv2.cvtColor(image_bgr, cv2.COLOR_GRAY2BGR)

    result = _run_sahi_on_image(image_bgr, det_model, slice_h, slice_w)

    r = []
    for ann in result.to_coco_annotations():
        if ann.get("category_id", None) != 0:
            continue

        x, y, w, h = map(float, ann["bbox"])
        cx = x + w / 2.0
        cy = y + h / 2.0
        r.append((x, y, w, h, cx, cy))

    r = sorted(r, key=lambda v: v[5])  # sort by y-center
    return r[:2]


def transform_point(M, px, py):
    p = np.array([px, py, 1], dtype=np.float32)
    q = M @ p
    return int(round(q[0])), int(round(q[1]))


def detect_and_crop_gray(pre_img_gray, det_model, slice_h, slice_w):
    """
    Crop vertically between first and last R markers on aligned grayscale image.
    Returns:
        crop_bgr, top_offset, detections
    """
    if pre_img_gray is None:
        return None, None, []

    if pre_img_gray.ndim == 3:
        pre_img_gray = cv2.cvtColor(pre_img_gray, cv2.COLOR_BGR2GRAY)

    tmp_path = os.path.join(os.getcwd(), f"__tmp_crop_{uuid.uuid4().hex}.png")
    ok = cv2.imwrite(tmp_path, pre_img_gray)
    if not ok:
        raise RuntimeError(f"Failed to save temp image: {tmp_path}")

    try:
        result = get_sliced_prediction(
            tmp_path,
            det_model,
            slice_height=slice_h,
            slice_width=slice_w,
            overlap_height_ratio=0.2,
            overlap_width_ratio=0.2,
            auto_slice_resolution=False,
        )
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    detections = []
    for ann in result.to_coco_annotations():
        if ann.get("category_id", None) != 0:
            continue
        if "bbox" not in ann or len(ann["bbox"]) != 4:
            continue

        x, y, w, h = [int(v) for v in ann["bbox"]]
        detections.append((x, y, w, h))

    if len(detections) < 2:
        return None, None, detections

    detections = sorted(detections, key=lambda b: b[1])

    H, W = pre_img_gray.shape[:2]
    y1 = max(0, min(detections[0][1], H - 1))
    y2 = max(0, min(detections[-1][1], H - 1))

    if y2 <= y1:
        return None, None, detections

    crop_gray = pre_img_gray[y1:y2, :]
    if crop_gray.size == 0:
        return None, None, detections

    crop_bgr = cv2.cvtColor(crop_gray, cv2.COLOR_GRAY2BGR)
    return crop_bgr, y1, detections


def align_and_crop_to_reference(
    image_bgr,
    reference_bgr,
    det_model,
    slice_h,
    slice_w,
    target_size=None,
):
    """
    Exact standalone-script behavior:
    1) detect R on polarized current image
    2) detect R on polarized reference image
    3) align current to reference
    4) detect R again on aligned grayscale image
    5) crop between R markers
    6) resize final crop

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

    # detect on source
    raw_r = detect_two_r_from_image(image_bgr, det_model, slice_h, slice_w)
    if len(raw_r) < 2:
        return None, None, {"status": "fail", "reason": "source_less_than_2_r"}

    # detect on reference
    ref_r = detect_two_r_from_image(reference_bgr, det_model, slice_h, slice_w)
    if len(ref_r) < 2:
        return None, None, {"status": "fail", "reason": "reference_less_than_2_r"}

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
        return None, None, {"status": "fail", "reason": "affine_estimation_failed"}

    aligned_bgr = cv2.warpAffine(
        image_bgr,
        M,
        (image_bgr.shape[1], image_bgr.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderValue=(0, 0, 0),
    )

    gray_aligned = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2GRAY)
    crop_bgr, top_offset, crop_r = detect_and_crop_gray(
        gray_aligned,
        det_model,
        slice_h,
        slice_w,
    )

    if crop_bgr is None:
        return None, aligned_bgr, {"status": "fail", "reason": "crop_failed_after_alignment"}

    if target_size is not None:
        target_w, target_h = target_size
        crop_bgr = cv2.resize(
            crop_bgr,
            (target_w, target_h),
            interpolation=cv2.INTER_LINEAR,
        )

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
        "crop_top_offset": top_offset,
        "crop_r_detections": crop_r,
        "final_h": int(crop_bgr.shape[0]),
        "final_w": int(crop_bgr.shape[1]),
    }
    return crop_bgr, aligned_bgr, meta