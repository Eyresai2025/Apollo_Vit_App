import cv2
import math
import numpy as np
import onnxruntime as ort


# =========================================================
# DEFAULTS
# =========================================================
IMG_SIZE = 640
DEFAULT_CONF_THRES = 0.4
DEFAULT_IOU_THRES = 0.45


# =========================================================
# BUILD DETECTOR
# =========================================================
def build_r_detector(model_path, conf=0.4, device="cuda"):
    """
    ONNX-based replacement for the old SAHI/Ultralytics detector.

    Returns a dict-style detector object.
    """
    providers = []
    dev = str(device).lower() if device is not None else "cpu"
    if dev.startswith("cuda"):
        providers.append("CUDAExecutionProvider")
    providers.append("CPUExecutionProvider")

    session = ort.InferenceSession(model_path, providers=providers)
    input_name = session.get_inputs()[0].name

    det_model = {
        "session": session,
        "input_name": input_name,
        "conf": float(conf if conf is not None else DEFAULT_CONF_THRES),
        "iou": float(DEFAULT_IOU_THRES),
        "img_size": int(IMG_SIZE),
        "providers": providers,
        "model_path": model_path,
        "device": device,
    }

    print(f"[ONNX R] loaded model: {model_path}")
    print(f"[ONNX R] providers   : {session.get_providers()}")
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

    return np.array(idxs).reshape(-1).tolist()


def _run_patch_inference(patch_bgr, off_x, off_y, det_model):
    session = det_model["session"]
    input_name = det_model["input_name"]
    img_size = det_model["img_size"]
    conf_thres = det_model["conf"]
    iou_thres = det_model["iou"]

    inp = _preprocess_patch(patch_bgr, img_size)
    outputs = session.run(None, {input_name: inp})[0]

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
        conf = float(conf)

        if conf < conf_thres:
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
        scores.append(conf)
        candidates.append([float(x1), float(y1), float(x2), float(y2), conf])

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

    return _global_nms_xyxy(all_dets, det_model["conf"], det_model["iou"])


def _det_to_tuple_xywh(det):
    x1, y1, x2, y2, _conf = det
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    cx = x1 + w / 2.0
    cy = y1 + h / 2.0
    return (float(x1), float(y1), float(w), float(h), float(cx), float(cy))


def transform_point(M, px, py):
    p = np.array([px, py, 1.0], dtype=np.float32)
    q = M @ p
    return int(round(float(q[0]))), int(round(float(q[1])))


# =========================================================
# R DETECTION
# =========================================================
def detect_two_r_from_image(image_bgr, det_model, slice_h, slice_w):
    """
    Match the old return style:
        [(x, y, w, h, cx, cy), ...]
    sorted by y-center.
    """
    if image_bgr is None:
        return []

    image_bgr = _ensure_bgr(image_bgr)
    dets = _run_onnx_on_image(image_bgr, det_model, slice_h, slice_w)

    if not dets:
        return []

    r = [_det_to_tuple_xywh(d) for d in dets]
    r = sorted(r, key=lambda v: v[5])  # by cy

    if len(r) >= 2:
        return r[:2]
    return []


def get_reference_r_points(reference_bgr, det_model, slice_h, slice_w):
    ref_r = detect_two_r_from_image(reference_bgr, det_model, slice_h, slice_w)
    if len(ref_r) < 2:
        return None
    return ref_r


# =========================================================
# FIXED-BAND CROPPING
# =========================================================
def crop_between_fixed_y(image_bgr, y1, y2, target_size=None):
    """
    Crop image using FIXED y1,y2 band.
    """
    if image_bgr is None:
        return None

    image_bgr = _ensure_bgr(image_bgr)
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


def crop_from_reference_band_info(image_bgr, ref_info, target_size=None):
    """
    Use ONLY the precomputed reference band (y1, y2) to crop.
    No incoming-image R detection is done here.
    """
    if image_bgr is None:
        return None, None, {"status": "fail", "reason": "image_none"}

    image_bgr = _ensure_bgr(image_bgr)

    if ref_info is None:
        return None, None, {"status": "fail", "reason": "reference_band_info_none"}

    if ref_info.get("status") != "ok":
        return None, None, ref_info

    y1 = ref_info["y1"]
    y2 = ref_info["y2"]

    crop_bgr = crop_between_fixed_y(image_bgr, y1, y2, target_size=target_size)

    if crop_bgr is None:
        return None, image_bgr, {
            "status": "fail",
            "reason": "crop_failed_reference_band_only",
            "fixed_y1": y1,
            "fixed_y2": y2,
        }

    meta = {
        "status": "ok",
        "mode": "reference_band_only",
        "fixed_crop_y1": int(y1),
        "fixed_crop_y2": int(y2),
        "ref_h": int(ref_info.get("ref_h", image_bgr.shape[0])),
        "ref_w": int(ref_info.get("ref_w", image_bgr.shape[1])),
        "final_h": int(crop_bgr.shape[0]),
        "final_w": int(crop_bgr.shape[1]),
    }
    return crop_bgr, image_bgr.copy(), meta


def get_reference_r_band(reference_bgr, det_model, slice_h, slice_w):
    """
    Detect R on reference image ONCE and return fixed crop band info.
    """
    if reference_bgr is None:
        return {"status": "fail", "reason": "reference_none"}

    reference_bgr = _ensure_bgr(reference_bgr)

    ref_r = detect_two_r_from_image(reference_bgr, det_model, slice_h, slice_w)
    if len(ref_r) < 2:
        return {
            "status": "fail",
            "reason": "reference_less_than_2_r",
            "ref_r": ref_r,
        }

    ref_r = sorted(ref_r, key=lambda v: v[5])

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


# =========================================================
# OPTIONAL ALIGNMENT TO REFERENCE + FIXED BAND
# =========================================================
def align_and_crop_to_reference_fixed_band(
    image_bgr,
    reference_bgr,
    det_model,
    slice_h,
    slice_w,
    target_size=None,
    ref_info=None,
):
    """
    1) detect R on source image
    2) use precomputed/detected ref_info
    3) align source to reference using reference R centers
    4) crop aligned image using fixed reference y1,y2
    """
    if image_bgr is None:
        return None, None, {"status": "fail", "reason": "image_none"}
    if reference_bgr is None:
        return None, None, {"status": "fail", "reason": "reference_none"}

    image_bgr = _ensure_bgr(image_bgr)
    reference_bgr = _ensure_bgr(reference_bgr)

    raw_r = detect_two_r_from_image(image_bgr, det_model, slice_h, slice_w)
    if len(raw_r) < 2:
        return None, None, {
            "status": "fail",
            "reason": "source_less_than_2_r",
            "raw_r": raw_r,
        }

    if ref_info is None:
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

    ref_h, ref_w = reference_bgr.shape[:2]
    aligned_bgr = cv2.warpAffine(
        image_bgr,
        M,
        (ref_w, ref_h),
        flags=cv2.INTER_LINEAR,
        borderValue=(0, 0, 0),
    )

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