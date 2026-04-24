import os
import cv2
import math
import numpy as np
import onnxruntime as ort


# =========================================================
# DEFAULTS
# =========================================================
IMG_SIZE = 640
DEFAULT_CONF_THRES = 0.3
DEFAULT_IOU_THRES = 0.45


# =========================================================
# BUILD DETECTOR
# =========================================================
def build_r_detector(model_path, conf=0.4, device="cuda"):
    """
    Same public interface as current build_r_detector(...)

    Returns a small dict-like detector object that carries:
      - ONNX Runtime session
      - input tensor name
      - thresholds
      - device/providers metadata
    """
    providers = []

    dev = str(device).lower() if device is not None else "cpu"
    if dev.startswith("cuda"):
        providers.append("CUDAExecutionProvider")
    providers.append("CPUExecutionProvider")

    sess = ort.InferenceSession(model_path, providers=providers)
    input_name = sess.get_inputs()[0].name

    det_model = {
        "session": sess,
        "input_name": input_name,
        "conf": float(conf if conf is not None else DEFAULT_CONF_THRES),
        "iou": float(DEFAULT_IOU_THRES),
        "img_size": int(IMG_SIZE),
        "providers": providers,
        "model_path": model_path,
        "device": device,
    }

    print(f"[ONNX R] loaded model: {model_path}")
    print(f"[ONNX R] providers   : {sess.get_providers()}")
    print(f"[ONNX R] conf        : {det_model['conf']}")
    return det_model


# =========================================================
# HELPERS
# =========================================================
def _ensure_bgr(image_bgr):
    if image_bgr is None:
        return None
    if image_bgr.ndim == 2:
        return cv2.cvtColor(image_bgr, cv2.COLOR_GRAY2BGR)
    return image_bgr


def _preprocess_patch(img_bgr, img_size):
    x = cv2.resize(img_bgr, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
    x = x.astype(np.float32) / 255.0
    x = np.transpose(x, (2, 0, 1))[None, ...]
    return x


def _apply_nms_xywh(boxes_xywh, scores, conf_thres, iou_thres):
    if not boxes_xywh:
        return []

    idxs = cv2.dnn.NMSBoxes(boxes_xywh, scores, conf_thres, iou_thres)
    if idxs is None or len(idxs) == 0:
        return []

    idxs = np.array(idxs).reshape(-1).tolist()
    return idxs


def _run_patch_inference(patch_bgr, off_x, off_y, det_model):
    session = det_model["session"]
    input_name = det_model["input_name"]
    img_size = det_model["img_size"]
    conf_thres = det_model["conf"]
    iou_thres = det_model["iou"]

    inp = _preprocess_patch(patch_bgr, img_size)
    outputs = session.run(None, {input_name: inp})[0]

    # Common export shapes:
    #   [1, N, D] or [N, D]
    if outputs.ndim == 3:
        outputs = outputs[0]

    patch_h, patch_w = patch_bgr.shape[:2]
    sx = patch_w / float(img_size)
    sy = patch_h / float(img_size)

    boxes_xywh = []
    scores = []
    candidates = []

    for det in outputs:
        if len(det) < 5:
            continue

        x1, y1, x2, y2, conf = det[:5]

        if float(conf) < conf_thres:
            continue

        x1 = float(x1) * sx + off_x
        y1 = float(y1) * sy + off_y
        x2 = float(x2) * sx + off_x
        y2 = float(y2) * sy + off_y

        w = max(0.0, x2 - x1)
        h = max(0.0, y2 - y1)
        if w <= 1 or h <= 1:
            continue

        boxes_xywh.append([int(round(x1)), int(round(y1)), int(round(w)), int(round(h))])
        scores.append(float(conf))
        candidates.append([float(x1), float(y1), float(x2), float(y2), float(conf)])

    keep = _apply_nms_xywh(boxes_xywh, scores, conf_thres, iou_thres)
    return [candidates[i] for i in keep]


def _global_nms_xyxy(dets_xyxy_conf, conf_thres, iou_thres):
    if not dets_xyxy_conf:
        return []

    boxes_xywh = []
    scores = []
    for x1, y1, x2, y2, conf in dets_xyxy_conf:
        boxes_xywh.append([
            int(round(x1)),
            int(round(y1)),
            int(round(max(0.0, x2 - x1))),
            int(round(max(0.0, y2 - y1))),
        ])
        scores.append(float(conf))

    keep = _apply_nms_xywh(boxes_xywh, scores, conf_thres, iou_thres)
    return [dets_xyxy_conf[i] for i in keep]


def _run_onnx_on_image(image_bgr, det_model, slice_h, slice_w):
    """
    Slice the image exactly from the provided slice_h/slice_w grid.
    This preserves compatibility with the current pipeline config, which
    already passes SLICE_H and SLICE_W into align_and_crop_to_reference(...).
    """
    if image_bgr is None:
        raise RuntimeError("image_bgr is None")

    image_bgr = _ensure_bgr(image_bgr)
    H, W = image_bgr.shape[:2]

    rows = math.ceil(H / slice_h)
    cols = math.ceil(W / slice_w)
    expected_slices = rows * cols
    print(f"[DEBUG] ONNX input shape: {(H, W)} | slice=({slice_h}, {slice_w}) | expected_slices={expected_slices}")

    all_dets = []

    for r in range(rows):
        y0 = r * slice_h
        y1 = min(H, y0 + slice_h)

        for c in range(cols):
            x0 = c * slice_w
            x1 = min(W, x0 + slice_w)

            patch = image_bgr[y0:y1, x0:x1]
            if patch.size == 0:
                continue

            dets = _run_patch_inference(patch, x0, y0, det_model)
            if dets:
                all_dets.extend(dets)

    # one more global NMS across all slice detections
    all_dets = _global_nms_xyxy(all_dets, det_model["conf"], det_model["iou"])
    return all_dets


def transform_point(M, px, py):
    p = np.array([px, py, 1.0], dtype=np.float32)
    q = M @ p
    return int(round(float(q[0]))), int(round(float(q[1])))


def _det_to_tuple_xywh(det):
    x1, y1, x2, y2, _conf = det
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    cx = x1 + w / 2.0
    cy = y1 + h / 2.0
    return (float(x1), float(y1), float(w), float(h), float(cx), float(cy))


# =========================================================
# PUBLIC DETECTION HELPERS
# =========================================================
def detect_two_r_from_image(image_bgr, det_model, slice_h, slice_w):
    """
    Same return style as current implementation:
        [(x, y, w, h, cx, cy), ...]
    sorted by y-center, top two only.
    """
    if image_bgr is None:
        return []

    image_bgr = _ensure_bgr(image_bgr)

    dets = _run_onnx_on_image(image_bgr, det_model, slice_h, slice_w)
    if not dets:
        return []

    r = [_det_to_tuple_xywh(d) for d in dets]
    r = sorted(r, key=lambda v: v[5])  # by cy
    return r[:2]


def get_reference_r_points(reference_bgr, det_model, slice_h, slice_w):
    ref_r = detect_two_r_from_image(reference_bgr, det_model, slice_h, slice_w)
    if len(ref_r) < 2:
        return None
    return ref_r


# =========================================================
# OPTIONAL LEGACY COMPAT HELPER
# =========================================================
def detect_and_crop_gray(pre_img_gray, det_model, slice_h, slice_w):
    """
    Kept for interface compatibility with the current module.
    Detect on grayscale image converted to BGR, crop vertically between
    top two R detections.
    """
    if pre_img_gray is None:
        return None, None, []

    if pre_img_gray.ndim == 3:
        pre_img_gray = cv2.cvtColor(pre_img_gray, cv2.COLOR_BGR2GRAY)

    gray_bgr = cv2.cvtColor(pre_img_gray, cv2.COLOR_GRAY2BGR)
    detections = detect_two_r_from_image(gray_bgr, det_model, slice_h, slice_w)

    if len(detections) < 2:
        return None, None, detections

    H, W = pre_img_gray.shape[:2]
    y1 = max(0, min(int(round(detections[0][1])), H - 1))
    y2 = max(0, min(int(round(detections[1][1])), H - 1))

    if y2 <= y1:
        return None, None, detections

    crop_gray = pre_img_gray[y1:y2, :]
    if crop_gray.size == 0:
        return None, None, detections

    crop_bgr = cv2.cvtColor(crop_gray, cv2.COLOR_GRAY2BGR)
    return crop_bgr, y1, detections


# =========================================================
# MAIN ALIGN + CROP
# =========================================================


def align_and_crop_to_reference(
    image_bgr,
    reference_bgr,
    det_model,
    slice_h,
    slice_w,
    target_size=None,
    reference_r=None,
):
    """
    Same public contract as the current R_Detection_align_crop.align_and_crop_to_reference(...)

    Important:
    - Source R is detected using ONNX.
    - Reference geometry comes from reference_r if supplied.
    - Crop band follows reference_y1/reference_y2 exactly, like current code.
    """
    if image_bgr is None:
        return None, None, {"status": "fail", "reason": "image_none"}
    if reference_bgr is None:
        return None, None, {"status": "fail", "reason": "reference_none"}
    if reference_r is not None:
        if not isinstance(reference_r, (list, tuple)) or len(reference_r) < 2:
            return None, None, {
                "status": "fail",
                "reason": f"bad_reference_r_type_or_length: type={type(reference_r)}, value={reference_r}"
            }

    image_bgr = _ensure_bgr(image_bgr)
    reference_bgr = _ensure_bgr(reference_bgr)

    # Detect R on source
    raw_r = detect_two_r_from_image(image_bgr, det_model, slice_h, slice_w)
    if len(raw_r) < 2:
        return None, None, {"status": "fail", "reason": "source_less_than_2_r"}

    # Use saved reference_r when available
    if reference_r is None:
        ref_r = detect_two_r_from_image(reference_bgr, det_model, slice_h, slice_w)
        if len(ref_r) < 2:
            return None, None, {"status": "fail", "reason": "reference_less_than_2_r"}
    else:
        ref_r = reference_r

    ref_pts = np.array(
        [
            [ref_r[0][4], ref_r[0][5]],
            [ref_r[1][4], ref_r[1][5]],
        ],
        dtype=np.float32,
    )

    src_pts = np.array(
        [
            [raw_r[0][4], raw_r[0][5]],
            [raw_r[1][4], raw_r[1][5]],
        ],
        dtype=np.float32,
    )

    M, _ = cv2.estimateAffinePartial2D(src_pts, ref_pts)
    if M is None:
        return None, None, {"status": "fail", "reason": "affine_estimation_failed"}

    # Keep the same reference-band crop logic as current code
    ref_h, ref_w = reference_bgr.shape[:2]
    y1 = int(round(ref_r[0][1]))
    y2 = int(round(ref_r[1][1]))

    y1 = max(0, min(y1, ref_h - 1))
    y2 = max(0, min(y2, ref_h - 1))

    if y2 <= y1:
        return None, None, {"status": "fail", "reason": "invalid_crop_band"}

    band_h = y2 - y1
    band_w = ref_w

    # ROI-only warp, same strategy as current module
    Minv = cv2.invertAffineTransform(M)

    dst_corners = np.array(
        [
            [0, y1],
            [ref_w - 1, y1],
            [0, y2 - 1],
            [ref_w - 1, y2 - 1],
        ],
        dtype=np.float32,
    ).reshape(-1, 1, 2)

    src_corners = cv2.transform(dst_corners, Minv).reshape(-1, 2)

    margin = 20
    src_h, src_w = image_bgr.shape[:2]

    sx0 = max(0, int(np.floor(src_corners[:, 0].min())) - margin)
    sx1 = min(src_w, int(np.ceil(src_corners[:, 0].max())) + margin)
    sy0 = max(0, int(np.floor(src_corners[:, 1].min())) - margin)
    sy1 = min(src_h, int(np.ceil(src_corners[:, 1].max())) + margin)

    if sx1 <= sx0 or sy1 <= sy0:
        return None, None, {
            "status": "fail",
            "reason": "invalid_source_roi",
            "src_corners": src_corners.tolist(),
        }

    src_roi = image_bgr[sy0:sy1, sx0:sx1]

    if src_roi.shape[0] >= 32767 or src_roi.shape[1] >= 32767:
        return None, None, {
            "status": "fail",
            "reason": "source_roi_too_large_for_opencv",
            "src_roi_shape": list(src_roi.shape),
        }

    if band_h >= 32767 or band_w >= 32767:
        return None, None, {
            "status": "fail",
            "reason": "destination_band_too_large_for_opencv",
            "band_h": int(band_h),
            "band_w": int(band_w),
        }

    A = M[:, :2]
    t = M[:, 2]
    offset = np.array([sx0, sy0], dtype=np.float32)

    t_roi = A @ offset + t
    M_roi = np.hstack([A, t_roi.reshape(2, 1)]).astype(np.float32)

    # output y=0 should correspond to full-image y=y1
    M_roi[1, 2] -= y1

    crop_bgr = cv2.warpAffine(
        src_roi,
        M_roi,
        (band_w, band_h),
        flags=cv2.INTER_LINEAR,
        borderValue=(0, 0, 0),
    )

    if crop_bgr is None or crop_bgr.size == 0:
        return None, None, {"status": "fail", "reason": "warp_crop_failed"}

    if target_size is not None:
        target_w, target_h = target_size
        crop_bgr = cv2.resize(crop_bgr, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

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
        "fixed_crop_y1": int(y1),
        "fixed_crop_y2": int(y2),
        "src_roi": [int(sx0), int(sy0), int(sx1), int(sy1)],
        "warp_mode": "roi_band_only",
        "final_h": int(crop_bgr.shape[0]),
        "final_w": int(crop_bgr.shape[1]),
    }

    aligned_bgr = None
    return crop_bgr, aligned_bgr, meta