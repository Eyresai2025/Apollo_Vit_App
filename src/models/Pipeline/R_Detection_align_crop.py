import os
import cv2
import math
import json
import numpy as np
from ultralytics import YOLO

# =========================================================
# DEFAULTS
# =========================================================
IMG_SIZE = 640
DEFAULT_CONF_THRES = 0.3
DEFAULT_IOU_THRES = 0.45

# Crop band
RAW_R_CROP_MARGIN_Y = 0

# =========================================================
# CROP-LEVEL ANCHOR ALIGNMENT CONFIG
# This runs AFTER raw R crop + resize to 2000x10000.
# =========================================================
CROP_ANCHOR_PATCH_H = 2500
CROP_ANCHOR_PATCH_W = 2000
CROP_ANCHOR_STEP_H = 2000
CROP_ANCHOR_STEP_W = 1500

CROP_ANCHOR_CONF = 0.10
CROP_ANCHOR_IOU = 0.50
CROP_ANCHOR_KEEP_CLASS = None
CROP_ANCHOR_SELECT_MODE = "highest_conf"

CROP_ANCHOR_INTERPOLATION = cv2.INTER_LINEAR


# =========================================================
# BUILD DETECTOR
# =========================================================
def build_r_detector(model_path, conf=0.25, device="cuda"):
    dev = str(device).lower() if device is not None else "cpu"

    model = YOLO(model_path)

    det_model = {
        "model": model,
        "conf": float(conf if conf is not None else DEFAULT_CONF_THRES),
        "iou": float(DEFAULT_IOU_THRES),
        "img_size": int(IMG_SIZE),
        "model_path": model_path,
        "device": dev,
    }

    print(f"[ONNX R] loaded model: {model_path}")
    print(f"[ONNX R] device      : {dev}")
    print(f"[ONNX R] conf        : {det_model['conf']}")

    return det_model

# =========================================================
# BASIC HELPERS
# =========================================================
def _ensure_bgr(image_bgr):
    if image_bgr is None:
        return None

    if image_bgr.ndim == 2:
        return cv2.cvtColor(image_bgr, cv2.COLOR_GRAY2BGR)

    if image_bgr.ndim == 3 and image_bgr.shape[2] == 1:
        return cv2.cvtColor(image_bgr[:, :, 0], cv2.COLOR_GRAY2BGR)

    return image_bgr


def _apply_nms_xywh(boxes_xywh, scores, conf_thres, iou_thres):
    if not boxes_xywh:
        return []

    idxs = cv2.dnn.NMSBoxes(
        boxes_xywh,
        scores,
        float(conf_thres),
        float(iou_thres),
    )

    if idxs is None or len(idxs) == 0:
        return []

    idxs = np.array(idxs).reshape(-1).tolist()
    return idxs


def _det_to_tuple_xywh(det):
    """
    Converts [x1, y1, x2, y2, conf] to:
        (x, y, w, h, cx, cy)
    """
    x1, y1, x2, y2, _conf = det

    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)

    cx = x1 + w / 2.0
    cy = y1 + h / 2.0

    return (
        float(x1),
        float(y1),
        float(w),
        float(h),
        float(cx),
        float(cy),
    )


# =========================================================
# FULL IMAGE R DETECTION USING SLICES
# =========================================================
def _run_patch_inference(patch_bgr, off_x, off_y, det_model):
    model = det_model["model"]
    img_size = det_model["img_size"]
    conf_thres = det_model["conf"]
    iou_thres = det_model["iou"]
    device = det_model["device"]

    if patch_bgr is None or patch_bgr.size == 0:
        return []

    pred_device = 0 if str(device).startswith("cuda") else "cpu"

    results = model.predict(
        source=patch_bgr,
        imgsz=img_size,
        conf=conf_thres,
        iou=iou_thres,
        device=pred_device,
        verbose=False,
        save=False,
    )

    if not results:
        return []

    r = results[0]

    if r.boxes is None or r.boxes.xyxy is None or len(r.boxes) == 0:
        return []

    boxes = r.boxes.xyxy.detach().cpu().numpy()
    confs = r.boxes.conf.detach().cpu().numpy()

    candidates = []

    for box, conf in zip(boxes, confs):
        x1, y1, x2, y2 = box.tolist()

        x1 = float(x1) + float(off_x)
        y1 = float(y1) + float(off_y)
        x2 = float(x2) + float(off_x)
        y2 = float(y2) + float(off_y)

        w = max(0.0, x2 - x1)
        h = max(0.0, y2 - y1)

        if w <= 1 or h <= 1:
            continue

        candidates.append([
            float(x1),
            float(y1),
            float(x2),
            float(y2),
            float(conf),
        ])

    if not candidates:
        return []

    boxes_xywh = []
    scores = []

    for x1, y1, x2, y2, conf in candidates:
        boxes_xywh.append([
            int(round(x1)),
            int(round(y1)),
            int(round(max(0.0, x2 - x1))),
            int(round(max(0.0, y2 - y1))),
        ])
        scores.append(float(conf))

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
    Runs sliced R detection on the full raw/preprocessed image.
    """
    if image_bgr is None:
        raise RuntimeError("image_bgr is None")

    image_bgr = _ensure_bgr(image_bgr)

    H, W = image_bgr.shape[:2]

    rows = math.ceil(H / slice_h)
    cols = math.ceil(W / slice_w)
    expected_slices = rows * cols

    print(
        f"[DEBUG] ONNX input shape: {(H, W)} | "
        f"slice=({slice_h}, {slice_w}) | "
        f"expected_slices={expected_slices}"
    )

    all_dets = []

    for rr in range(rows):
        y0 = rr * slice_h
        y1 = min(H, y0 + slice_h)

        for cc in range(cols):
            x0 = cc * slice_w
            x1 = min(W, x0 + slice_w)

            patch = image_bgr[y0:y1, x0:x1]

            if patch.size == 0:
                continue

            dets = _run_patch_inference(
                patch_bgr=patch,
                off_x=x0,
                off_y=y0,
                det_model=det_model,
            )

            if dets:
                all_dets.extend(dets)

    all_dets = _global_nms_xyxy(
        all_dets,
        det_model["conf"],
        det_model["iou"],
    )

    return all_dets


def detect_r_candidates_from_image(image_bgr, det_model, slice_h, slice_w):
    """
    Detect all R candidates.

    Returns:
        [(x, y, w, h, cx, cy), ...] sorted by y-center
    """
    if image_bgr is None:
        return []

    image_bgr = _ensure_bgr(image_bgr)

    dets = _run_onnx_on_image(
        image_bgr=image_bgr,
        det_model=det_model,
        slice_h=slice_h,
        slice_w=slice_w,
    )

    if not dets:
        return []

    r = [_det_to_tuple_xywh(d) for d in dets]
    r = sorted(r, key=lambda v: v[5])

    return r


def detect_two_r_from_image(image_bgr, det_model, slice_h, slice_w):
    """
    Backward-compatible helper.

    Returns top two R candidates sorted by y-center.
    """
    candidates = detect_r_candidates_from_image(
        image_bgr=image_bgr,
        det_model=det_model,
        slice_h=slice_h,
        slice_w=slice_w,
    )

    return candidates[:2]


def get_reference_r_points(reference_bgr, det_model, slice_h, slice_w):
    """
    Used during calibration to save reference_r.pt.
    Kept for compatibility with your existing calibration artifact flow.
    """
    ref_r = detect_two_r_from_image(
        image_bgr=reference_bgr,
        det_model=det_model,
        slice_h=slice_h,
        slice_w=slice_w,
    )

    if len(ref_r) < 2:
        return None

    return ref_r


def draw_crop_anchor_debug(crop_bgr, cur_box, ref_box, save_path):
    if save_path is None:
        return

    vis = crop_bgr.copy()

    if vis.ndim == 2:
        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

    # Current anchor in red
    if cur_box is not None:
        x1, y1, x2, y2, conf, cls = cur_box

        cv2.rectangle(
            vis,
            (int(x1), int(y1)),
            (int(x2), int(y2)),
            (0, 0, 255),
            2,
        )

        cv2.putText(
            vis,
            f"CUR {float(conf):.2f}",
            (int(x1), max(20, int(y1) - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
        )

    # Reference anchor in green
    if ref_box is not None:
        x1, y1, x2, y2, conf, cls = ref_box

        cv2.rectangle(
            vis,
            (int(x1), int(y1)),
            (int(x2), int(y2)),
            (0, 255, 0),
            2,
        )

        cv2.putText(
            vis,
            "REF",
            (int(x1), max(45, int(y1) - 28)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, vis)

def project_raw_r_box_to_resized_crop(
    raw_r_box,
    crop_y1,
    crop_y2,
    src_w,
    resized_w,
    resized_h,
):
    """
    Project already-detected raw R bbox into the resized crop coordinate system.

    raw_r_box format:
        (x, y, w, h, cx, cy)

    Returns:
        [x1, y1, x2, y2, conf, cls]
    """
    x, y, w, h, cx, cy = raw_r_box

    raw_crop_h = max(float(crop_y2 - crop_y1), 1.0)

    scale_x = float(resized_w) / max(float(src_w), 1.0)
    scale_y = float(resized_h) / raw_crop_h

    x1 = float(x) * scale_x
    y1 = (float(y) - float(crop_y1)) * scale_y
    x2 = (float(x) + float(w)) * scale_x
    y2 = (float(y) + float(h) - float(crop_y1)) * scale_y

    x1 = max(0.0, min(float(resized_w - 1), x1))
    y1 = max(0.0, min(float(resized_h - 1), y1))
    x2 = max(0.0, min(float(resized_w), x2))
    y2 = max(0.0, min(float(resized_h), y2))

    return [
        float(x1),
        float(y1),
        float(x2),
        float(y2),
        1.0,
        0,
    ]


def apply_crop_anchor_alignment(
    crop_bgr,
    crop_anchor_ref_path,
    current_anchor_box,
    crop_anchor_debug_path=None,
    debug_name="",
):
    """
    AUTO reference logic using projected raw-R anchor.

    If crop_anchor_ref_path does not exist:
        current crop becomes reference.

    If crop_anchor_ref_path exists:
        current crop is shifted to saved reference anchor.

    No second YOLO detection is done on the resized crop.
    """
    if crop_bgr is None:
        raise RuntimeError("[CROP_ANCHOR] crop_bgr is None")

    if crop_anchor_ref_path is None:
        raise RuntimeError("[CROP_ANCHOR] crop_anchor_ref_path is None")

    if current_anchor_box is None:
        raise RuntimeError("[CROP_ANCHOR] current_anchor_box is None")

    crop_bgr = _ensure_bgr(crop_bgr)

    x1, y1, x2, y2, conf, cls = current_anchor_box

    # =====================================================
    # CASE 1: First calibration image creates reference
    # =====================================================
    if not os.path.isfile(crop_anchor_ref_path):
        os.makedirs(os.path.dirname(crop_anchor_ref_path), exist_ok=True)

        ref_obj = {
            "anchor_box": [
                float(x1),
                float(y1),
                float(x2),
                float(y2),
                float(conf),
                int(cls),
            ],
            "anchor_x1": float(x1),
            "anchor_y1": float(y1),
            "anchor_w": float(x2 - x1),
            "anchor_h": float(y2 - y1),
            "crop_w": int(crop_bgr.shape[1]),
            "crop_h": int(crop_bgr.shape[0]),
            "anchor_source": "projected_raw_r_box",
            "debug_name": debug_name,
        }

        with open(crop_anchor_ref_path, "w", encoding="utf-8") as f:
            json.dump(ref_obj, f, indent=2)

        print(
            f"[CROP_ANCHOR][AUTO REF CREATED] {crop_anchor_ref_path} | "
            f"x1={x1:.1f}, y1={y1:.1f} | "
            f"source=projected_raw_r_box | debug_name={debug_name}"
        )

        draw_crop_anchor_debug(
            crop_bgr=crop_bgr,
            cur_box=current_anchor_box,
            ref_box=current_anchor_box,
            save_path=crop_anchor_debug_path,
        )

        return crop_bgr, {
            "enabled": True,
            "created_reference": True,
            "anchor_source": "projected_raw_r_box",
            "anchor_ref_path": crop_anchor_ref_path,
            "current_box": [
                float(x1),
                float(y1),
                float(x2),
                float(y2),
                float(conf),
                int(cls),
            ],
            "dx": 0.0,
            "dy": 0.0,
        }

    # =====================================================
    # CASE 2: Align to saved reference
    # =====================================================
    with open(crop_anchor_ref_path, "r", encoding="utf-8") as f:
        ref_obj = json.load(f)

    ref_box = ref_obj["anchor_box"]
    ref_x1 = float(ref_obj["anchor_x1"])
    ref_y1 = float(ref_obj["anchor_y1"])

    dx = ref_x1 - float(x1)
    dy = ref_y1 - float(y1)

    M = np.float32([
        [1, 0, dx],
        [0, 1, dy],
    ])

    aligned_crop = cv2.warpAffine(
        crop_bgr,
        M,
        (crop_bgr.shape[1], crop_bgr.shape[0]),
        flags=CROP_ANCHOR_INTERPOLATION,
        borderMode=cv2.BORDER_REFLECT_101,
    )

    print(
        f"[CROP_ANCHOR][AUTO ALIGN][{debug_name}] "
        f"cur=({x1:.1f},{y1:.1f}) ref=({ref_x1:.1f},{ref_y1:.1f}) "
        f"dx={dx:.1f}, dy={dy:.1f} | source=projected_raw_r_box"
    )

    shifted_cur_box = [
        ref_x1,
        ref_y1,
        ref_x1 + float(x2 - x1),
        ref_y1 + float(y2 - y1),
        float(conf),
        int(cls),
    ]

    draw_crop_anchor_debug(
        crop_bgr=aligned_crop,
        cur_box=shifted_cur_box,
        ref_box=ref_box,
        save_path=crop_anchor_debug_path,
    )

    return aligned_crop, {
        "enabled": True,
        "created_reference": False,
        "anchor_source": "projected_raw_r_box",
        "anchor_ref_path": crop_anchor_ref_path,
        "reference_box": ref_box,
        "current_box_before_shift": [
            float(x1),
            float(y1),
            float(x2),
            float(y2),
            float(conf),
            int(cls),
        ],
        "current_box_after_shift": shifted_cur_box,
        "dx": float(dx),
        "dy": float(dy),
    }


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

    # crop-anchor alignment
    enable_crop_anchor_align=False,
    crop_anchor_ref_path=None,
    create_crop_anchor_reference=False,
    crop_anchor_debug_path=None,
    debug_name="",
):
    """
    NEW FLOW:

    1. Detect R on raw/current image.
    2. Crop raw/current image directly between detected R1 and R2.
    3. Resize crop to target_size, normally 2000x10000.
    4. Run crop-anchor alignment on the resized crop.
    5. Return final crop.

    No full-image affine alignment.
    No chunked alignment.

    reference_bgr and reference_r are accepted only to keep your existing
    pipeline call compatible.
    """

    if image_bgr is None:
        return None, None, {
            "status": "fail",
            "reason": "image_none",
        }

    image_bgr = _ensure_bgr(image_bgr)

    # =====================================================
    # 1. Detect R on raw/current image
    # =====================================================
    raw_r = detect_two_r_from_image(
        image_bgr=image_bgr,
        det_model=det_model,
        slice_h=slice_h,
        slice_w=slice_w,
    )

    print("\n[RAW_R_CROP] ===============================")
    print("[RAW_R_CROP] detected R count:", len(raw_r))

    for idx, rdet in enumerate(raw_r):
        x, y, w, h, cx, cy = rdet
        print(
            f"[RAW_R_CROP] R{idx}: "
            f"x={x:.1f}, y={y:.1f}, w={w:.1f}, h={h:.1f}, "
            f"cx={cx:.1f}, cy={cy:.1f}"
        )

    if len(raw_r) < 2:
        return None, None, {
            "status": "fail",
            "reason": "source_less_than_2_r",
            "raw_r": raw_r,
        }

    raw_r = sorted(raw_r, key=lambda v: v[5])[:2]

    raw_gap = abs(float(raw_r[1][5]) - float(raw_r[0][5]))

    ref_gap = None
    gap_ratio = None

    if reference_r is not None and len(reference_r) >= 2:
        ref_r_sorted = sorted(reference_r, key=lambda v: v[5])[:2]
        ref_gap = abs(float(ref_r_sorted[1][5]) - float(ref_r_sorted[0][5]))
        gap_ratio = raw_gap / max(ref_gap, 1e-6)

        print(
            f"[R_SCALE_CHECK] raw_gap={raw_gap:.2f} | "
            f"ref_gap={ref_gap:.2f} | raw/ref={gap_ratio:.5f}"
        )

        if gap_ratio < 0.97 or gap_ratio > 1.03:
            print(
                "[R_SCALE_CHECK][WARN] R-to-R crop height differs from calibration. "
                "Patch scale mismatch may happen."
            )
    else:
        print(f"[R_SCALE_CHECK] raw_gap={raw_gap:.2f} | reference_r not available")

    # =====================================================
    # 2. Crop raw image between R1 and R2
    # =====================================================
    H, W = image_bgr.shape[:2]

    crop_y1 = int(round(raw_r[0][1])) - int(RAW_R_CROP_MARGIN_Y)
    crop_y2 = int(round(raw_r[1][1])) + int(RAW_R_CROP_MARGIN_Y)

    crop_y1 = max(0, min(crop_y1, H - 1))
    crop_y2 = max(0, min(crop_y2, H))

    if crop_y2 <= crop_y1:
        return None, None, {
            "status": "fail",
            "reason": "invalid_raw_r_crop_band",
            "crop_y1": crop_y1,
            "crop_y2": crop_y2,
            "raw_r": raw_r,
        }

    crop_bgr = image_bgr[crop_y1:crop_y2, :]

    if crop_bgr is None or crop_bgr.size == 0:
        return None, None, {
            "status": "fail",
            "reason": "empty_raw_r_crop",
            "crop_y1": crop_y1,
            "crop_y2": crop_y2,
        }

    print(
        f"[RAW_R_CROP] crop_y1={crop_y1} | crop_y2={crop_y2} | "
        f"before_resize_shape={crop_bgr.shape}"
    )

    # =====================================================
    # 3. Resize to final crop size: usually 2000x10000
    # =====================================================
    if target_size is not None:
        target_w, target_h = target_size

        crop_bgr = cv2.resize(
            crop_bgr,
            (int(target_w), int(target_h)),
            interpolation=cv2.INTER_LINEAR,
        )

    print(f"[RAW_R_CROP] after_resize_shape={crop_bgr.shape}")

    # =====================================================
    # 4. Crop-anchor alignment on resized crop
    # =====================================================
    crop_anchor_meta = {
        "enabled": False,
        "created_reference": False,
    }

    if crop_anchor_ref_path is not None:
        current_anchor_box = project_raw_r_box_to_resized_crop(
            raw_r_box=raw_r[0],
            crop_y1=crop_y1,
            crop_y2=crop_y2,
            src_w=W,
            resized_w=crop_bgr.shape[1],
            resized_h=crop_bgr.shape[0],
        )

        print(
            f"[CROP_ANCHOR] projected raw-R anchor | "
            f"x1={current_anchor_box[0]:.1f}, "
            f"y1={current_anchor_box[1]:.1f}, "
            f"x2={current_anchor_box[2]:.1f}, "
            f"y2={current_anchor_box[3]:.1f}"
        )

        crop_bgr, crop_anchor_meta = apply_crop_anchor_alignment(
            crop_bgr=crop_bgr,
            crop_anchor_ref_path=crop_anchor_ref_path,
            current_anchor_box=current_anchor_box,
            crop_anchor_debug_path=crop_anchor_debug_path,
            debug_name=debug_name,
        )

    # =====================================================
    # 5. Metadata
    # =====================================================
    meta = {
        "status": "ok",
        "alignment_mode": "raw_r_crop_then_crop_anchor",

        "raw_r": raw_r,

        "crop_y1": int(crop_y1),
        "crop_y2": int(crop_y2),
        "crop_h_before_resize": int(crop_y2 - crop_y1),
        "crop_w_before_resize": int(W),

        "target_size": list(target_size) if target_size is not None else None,
        "final_h": int(crop_bgr.shape[0]),
        "final_w": int(crop_bgr.shape[1]),

        "crop_anchor": crop_anchor_meta,
    }

    # aligned_bgr/debug image can be final crop itself
    return crop_bgr, crop_bgr.copy(), meta