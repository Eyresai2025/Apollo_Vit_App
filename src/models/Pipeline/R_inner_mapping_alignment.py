import os
import json
import cv2
import numpy as np

from src.models.Pipeline import inner_bead_tread_alignment as xalign


# ============================================================
# OFFSET RATIO LOADING
# ============================================================

def load_offset_ratio_from_json(json_path, side_name):
    """
    Supports both formats:

    Format 1:
        {
            "tread": 0.045,
            "innerwall": 0.012,
            "bead": 0.018
        }

    Format 2:
        {
            "final_offsets": {
                "tread": {"offset_ratio": 0.045},
                "innerwall": {"offset_ratio": 0.012},
                "bead": {"offset_ratio": 0.018}
            }
        }
    """
    if not json_path or not os.path.exists(json_path):
        raise RuntimeError(f"Offset JSON not found: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    side_name = side_name.lower().strip()

    if "final_offsets" in data:
        final_offsets = data["final_offsets"]
        if side_name not in final_offsets:
            raise KeyError(f"'{side_name}' missing in final_offsets of {json_path}")
        return float(final_offsets[side_name]["offset_ratio"])

    if side_name in data:
        return float(data[side_name])

    raise KeyError(f"Offset ratio for '{side_name}' not found in {json_path}")


# ============================================================
# R ANCHOR FROM SIDEWALL CROP META
# ============================================================

def extract_sidewall_r_anchor_from_meta(crop_meta):
    """
    crop_meta comes from R_detection_onnx_align_crop.align_and_crop_to_reference().
    That meta contains crop_y1 and crop_y2, which are the raw sidewall R1/R2 crop limits.
    """
    if crop_meta is None:
        raise RuntimeError("crop_meta is None. Cannot extract sidewall R anchor.")

    if "crop_y1" not in crop_meta or "crop_y2" not in crop_meta:
        raise RuntimeError(f"crop_meta missing crop_y1/crop_y2: {crop_meta}")

    r1 = int(crop_meta["crop_y1"])
    r2 = int(crop_meta["crop_y2"])

    if r2 <= r1:
        raise RuntimeError(f"Invalid sidewall R anchor: R1={r1}, R2={r2}")

    return {
        "r1_top_y": r1,
        "r2_top_y": r2,
        "one_rev_height": int(r2 - r1),
    }


# ============================================================
# OFFSET CROP
# ============================================================

def calculate_offset_crop_window(r1_top_y, r2_top_y, offset_ratio, image_height, side_name):
    r1_top_y = int(r1_top_y)
    r2_top_y = int(r2_top_y)

    if r2_top_y <= r1_top_y:
        raise RuntimeError(f"[{side_name}] Invalid R1/R2: R1={r1_top_y}, R2={r2_top_y}")

    one_rev_height = r2_top_y - r1_top_y

    start_y = int(round(r1_top_y + float(offset_ratio) * one_rev_height))
    end_y = start_y + one_rev_height

    if start_y < 0:
        raise RuntimeError(
            f"[{side_name}] crop start is negative: {start_y}. "
            f"Check offset ratio or capture extra lines."
        )

    if end_y > image_height:
        raise RuntimeError(
            f"[{side_name}] crop end exceeds image height: "
            f"end_y={end_y}, image_height={image_height}. "
            f"Check offset ratio or capture extra lines."
        )

    return start_y, end_y, one_rev_height


def crop_resize_by_sidewall_anchor(pre_bgr, side_name, sidewall_r_anchor, offset_ratio, target_size):
    """
    pre_bgr is already polarized image.
    This function only performs Y crop using sidewall R1/R2 + offset ratio,
    then resizes to target_size.
    """
    if pre_bgr is None:
        raise RuntimeError(f"[{side_name}] pre_bgr is None")

    r1 = int(sidewall_r_anchor["r1_top_y"])
    r2 = int(sidewall_r_anchor["r2_top_y"])

    start_y, end_y, one_rev_height = calculate_offset_crop_window(
        r1_top_y=r1,
        r2_top_y=r2,
        offset_ratio=offset_ratio,
        image_height=pre_bgr.shape[0],
        side_name=side_name,
    )

    crop_bgr = pre_bgr[start_y:end_y, :].copy()

    target_w, target_h = target_size
    crop_bgr = cv2.resize(
        crop_bgr,
        (int(target_w), int(target_h)),
        interpolation=cv2.INTER_AREA,
    )

    crop_meta = {
        "status": "ok",
        "mode": "sidewall_r_anchor_offset_crop",
        "side": side_name,
        "r1_top_y": int(r1),
        "r2_top_y": int(r2),
        "one_rev_height": int(one_rev_height),
        "offset_ratio": float(offset_ratio),
        "crop_start_y": int(start_y),
        "crop_end_y": int(end_y),
        "before_resize_shape": list(pre_bgr[start_y:end_y, :].shape),
        "after_resize_shape": list(crop_bgr.shape),
    }

    print(
        f"[{side_name.upper()} OFFSET CROP] "
        f"R1={r1} | R2={r2} | one_rev={one_rev_height} | "
        f"offset={offset_ratio:.6f} | crop={start_y}:{end_y} | "
        f"resized={crop_bgr.shape}"
    )

    return crop_bgr, crop_meta


# ============================================================
# X ALIGNMENT
# ============================================================

def configure_xalign_for_side(side_name):
    side_name = side_name.lower().strip()

    xalign.SIDE_NAME = side_name

    # Your latest working tread setting
    if side_name == "tread":
        xalign.DETECTION_METHOD_BY_SIDE["tread"] = "texture"
        xalign.TREAD_SEARCH_X_RATIO = (0.20, 0.92)
        xalign.TEXTURE_REFINE_WITH_SOBEL_X = False

    # For bead/innerwall start with foreground.
    # If they fail later, change their method to texture.
    if side_name == "innerwall":
        xalign.DETECTION_METHOD_BY_SIDE["innerwall"] = "foreground"

    if side_name == "bead":
        xalign.DETECTION_METHOD_BY_SIDE["bead"] = "foreground"

# ============================================================
# TREAD PROFILE X ALIGNMENT
# ============================================================

TREAD_PROFILE_MAX_SHIFT_PX = 350
TREAD_PROFILE_MIN_NCC_SCORE = 0.35
TREAD_PROFILE_ROW_ROI_RATIO = (0.05, 0.95)
TREAD_PROFILE_REMOVE_BRIGHT_ROWS = True
TREAD_PROFILE_BRIGHT_ROW_PERCENTILE = 97.5
TREAD_PROFILE_BORDER_VALUE = 0


def _profile_to_gray_float32(img):
    if img.ndim == 2:
        gray = img
    elif img.ndim == 3 and img.shape[2] == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif img.ndim == 3 and img.shape[2] == 4:
        gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    else:
        raise ValueError(f"Unsupported image shape: {img.shape}")

    return gray.astype(np.float32)


def _z_norm_1d(x):
    x = x.astype(np.float32)
    m = float(np.mean(x))
    s = float(np.std(x))

    if s < 1e-6:
        return x * 0.0

    return (x - m) / s


def build_tread_x_profile_signature(img):
    """
    Builds 1D X signature for tread profile alignment.

    Used only for X alignment calculation.
    The output aligned image is not preprocessed.
    """
    gray = _profile_to_gray_float32(img)
    h, w = gray.shape[:2]

    y1 = int(h * TREAD_PROFILE_ROW_ROI_RATIO[0])
    y2 = int(h * TREAD_PROFILE_ROW_ROI_RATIO[1])

    y1 = max(0, min(y1, h - 1))
    y2 = max(y1 + 1, min(y2, h))

    roi = gray[y1:y2, :].copy()
    rows_before = roi.shape[0]

    # Remove very bright horizontal tape rows
    if TREAD_PROFILE_REMOVE_BRIGHT_ROWS and roi.shape[0] > 50:
        row_median = np.median(roi, axis=1)
        cut = np.percentile(row_median, TREAD_PROFILE_BRIGHT_ROW_PERCENTILE)
        valid_rows = row_median < cut

        if np.sum(valid_rows) > 0.50 * len(valid_rows):
            roi = roi[valid_rows, :]

    rows_after = roi.shape[0]

    # Row-wise normalization
    row_med = np.median(roi, axis=1, keepdims=True)
    row_mad = np.median(np.abs(roi - row_med), axis=1, keepdims=True)
    row_mad[row_mad < 1.0] = 1.0

    roi_norm = (roi - row_med) / row_mad
    roi_norm = np.clip(roi_norm, -5.0, 5.0).astype(np.float32)

    intensity_profile = np.median(roi_norm, axis=0).astype(np.float32)

    sobel_x = cv2.Sobel(
        roi_norm,
        cv2.CV_32F,
        dx=1,
        dy=0,
        ksize=3,
    )

    sobel_x = np.abs(sobel_x)
    edge_profile = np.median(sobel_x, axis=0).astype(np.float32)

    intensity_profile = cv2.GaussianBlur(
        intensity_profile.reshape(1, -1),
        (21, 1),
        0,
    ).flatten()

    edge_profile = cv2.GaussianBlur(
        edge_profile.reshape(1, -1),
        (21, 1),
        0,
    ).flatten()

    intensity_profile = _z_norm_1d(intensity_profile)
    edge_profile = _z_norm_1d(edge_profile)

    signature = 0.35 * intensity_profile + 0.65 * edge_profile
    signature = _z_norm_1d(signature)

    meta = {
        "image_height": int(h),
        "image_width": int(w),
        "roi_y1": int(y1),
        "roi_y2": int(y2),
        "rows_before_tape_filter": int(rows_before),
        "rows_after_tape_filter": int(rows_after),
        "remove_bright_tape_rows": bool(TREAD_PROFILE_REMOVE_BRIGHT_ROWS),
    }

    return signature.astype(np.float32), meta


def _normalized_cross_correlation(a, b):
    a = a.astype(np.float32)
    b = b.astype(np.float32)

    if len(a) < 20 or len(b) < 20:
        return -1.0

    a = a - float(np.mean(a))
    b = b - float(np.mean(b))

    denom = float(np.linalg.norm(a) * np.linalg.norm(b))

    if denom < 1e-6:
        return -1.0

    return float(np.dot(a, b) / denom)


def find_tread_best_shift_x(ref_sig, cur_sig, max_shift_px):
    ref_sig = ref_sig.astype(np.float32)
    cur_sig = cur_sig.astype(np.float32)

    w = min(len(ref_sig), len(cur_sig))
    ref_sig = ref_sig[:w]
    cur_sig = cur_sig[:w]

    max_shift_px = int(min(max_shift_px, w // 3))

    best_shift = 0
    best_score = -999.0

    for shift in range(-max_shift_px, max_shift_px + 1):
        if shift > 0:
            ref_part = ref_sig[shift:]
            cur_part = cur_sig[:w - shift]
        elif shift < 0:
            s = -shift
            ref_part = ref_sig[:w - s]
            cur_part = cur_sig[s:]
        else:
            ref_part = ref_sig
            cur_part = cur_sig

        score = _normalized_cross_correlation(ref_part, cur_part)

        if score > best_score:
            best_score = score
            best_shift = shift

    return int(best_shift), float(best_score)


def shift_image_x_same_size_profile(img, shift_x, border_value=0):
    h, w = img.shape[:2]

    M = np.float32([
        [1, 0, int(shift_x)],
        [0, 1, 0],
    ])

    if img.ndim == 3:
        border = tuple([border_value] * img.shape[2])
    else:
        border = border_value

    aligned = cv2.warpAffine(
        img,
        M,
        (w, h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border,
    )

    return aligned


def _profile_to_uint8_display(img):
    if img.ndim == 3 and img.dtype == np.uint8:
        return img.copy()

    gray = _profile_to_gray_float32(img)

    p1 = np.percentile(gray, 1)
    p99 = np.percentile(gray, 99)

    if p99 <= p1:
        p1 = float(gray.min())
        p99 = float(gray.max())

    if p99 <= p1:
        out = np.zeros(gray.shape, dtype=np.uint8)
    else:
        out = (gray - p1) / (p99 - p1)
        out = np.clip(out, 0, 1)
        out = (out * 255).astype(np.uint8)

    return cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)


def save_tread_profile_debug(ref_img, cur_img, aligned_img, shift_x, score, debug_save_path):
    ref_vis = _profile_to_uint8_display(ref_img)
    cur_vis = _profile_to_uint8_display(cur_img)
    aligned_vis = _profile_to_uint8_display(aligned_img)

    h = min(ref_vis.shape[0], cur_vis.shape[0], aligned_vis.shape[0])
    w = min(ref_vis.shape[1], cur_vis.shape[1], aligned_vis.shape[1])

    ref_vis = ref_vis[:h, :w]
    cur_vis = cur_vis[:h, :w]
    aligned_vis = aligned_vis[:h, :w]

    cx = w // 2

    cv2.line(ref_vis, (cx, 0), (cx, h - 1), (0, 255, 0), 2)
    cv2.line(cur_vis, (cx, 0), (cx, h - 1), (0, 0, 255), 2)
    cv2.line(aligned_vis, (cx, 0), (cx, h - 1), (0, 255, 255), 2)

    cv2.putText(ref_vis, "REFERENCE", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    cv2.putText(cur_vis, f"BEFORE shift={shift_x}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
    cv2.putText(aligned_vis, f"AFTER score={score:.3f}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

    debug = np.hstack([ref_vis, cur_vis, aligned_vis])

    os.makedirs(os.path.dirname(debug_save_path), exist_ok=True)
    cv2.imwrite(debug_save_path, debug)


def apply_tread_profile_x_alignment(
    crop_bgr,
    artifacts_dir,
    create_reference_if_missing=True,
    debug_save_path=None,
):
    """
    Calibration:
        first tread crop creates profile reference.

    Inference:
        incoming tread crop aligns to saved profile reference.
    """
    os.makedirs(artifacts_dir, exist_ok=True)

    ref_sig_path = os.path.join(artifacts_dir, "tread_x_reference_signature.npy")
    ref_meta_path = os.path.join(artifacts_dir, "tread_x_reference_signature_meta.json")
    ref_crop_path = os.path.join(artifacts_dir, "tread_x_reference_crop.png")

    # --------------------------------------------------------
    # Calibration: create reference
    # --------------------------------------------------------
    if create_reference_if_missing and not os.path.exists(ref_sig_path):
        ref_sig, ref_meta = build_tread_x_profile_signature(crop_bgr)

        np.save(ref_sig_path, ref_sig)

        with open(ref_meta_path, "w", encoding="utf-8") as f:
            json.dump(ref_meta, f, indent=4)

        cv2.imwrite(ref_crop_path, crop_bgr)

        meta = {
            "status": "ok",
            "mode": "created_tread_profile_reference",
            "side": "tread",
            "final_shift_x": 0,
            "ncc_score": 1.0,
            "ref_signature_path": ref_sig_path,
            "ref_meta_path": ref_meta_path,
            "ref_crop_path": ref_crop_path,
            "reference_meta": ref_meta,
        }

        print(f"[TREAD PROFILE XALIGN] Created reference signature: {ref_sig_path}")
        print(f"[TREAD PROFILE XALIGN] Saved reference crop      : {ref_crop_path}")

        return crop_bgr, meta

    # --------------------------------------------------------
    # Inference: load reference and align
    # --------------------------------------------------------
    if not os.path.exists(ref_sig_path):
        raise RuntimeError(
            f"[tread] Tread profile reference missing: {ref_sig_path}. "
            f"Run tread calibration first."
        )

    if not os.path.exists(ref_crop_path):
        raise RuntimeError(
            f"[tread] Tread profile reference crop missing: {ref_crop_path}. "
            f"Run tread calibration first."
        )

    ref_sig = np.load(ref_sig_path)
    ref_crop = cv2.imread(ref_crop_path, cv2.IMREAD_UNCHANGED)

    if ref_crop is None:
        raise RuntimeError(f"[tread] Cannot read reference crop: {ref_crop_path}")

    cur_sig, cur_meta = build_tread_x_profile_signature(crop_bgr)

    shift_x, score = find_tread_best_shift_x(
        ref_sig=ref_sig,
        cur_sig=cur_sig,
        max_shift_px=TREAD_PROFILE_MAX_SHIFT_PX,
    )

    aligned_bgr = shift_image_x_same_size_profile(
        crop_bgr,
        shift_x=shift_x,
        border_value=TREAD_PROFILE_BORDER_VALUE,
    )

    warnings = []

    if score < TREAD_PROFILE_MIN_NCC_SCORE:
        warnings.append(
            f"LOW_NCC_SCORE {score:.3f} < {TREAD_PROFILE_MIN_NCC_SCORE:.3f}"
        )

    if debug_save_path:
        try:
            save_tread_profile_debug(
                ref_img=ref_crop,
                cur_img=crop_bgr,
                aligned_img=aligned_bgr,
                shift_x=shift_x,
                score=score,
                debug_save_path=debug_save_path,
            )
        except Exception as e:
            print(f"[TREAD PROFILE XALIGN][WARN] debug save failed: {e}")

    meta = {
        "status": "ok",
        "mode": "tread_profile_x_alignment",
        "side": "tread",
        "final_shift_x": int(shift_x),
        "ncc_score": float(score),
        "warnings": warnings,
        "ref_signature_path": ref_sig_path,
        "ref_crop_path": ref_crop_path,
        "current_signature_meta": cur_meta,
    }

    print(
        f"[TREAD PROFILE XALIGN] shift_x={shift_x} | "
        f"score={score:.4f} | warnings={warnings}"
    )

    return aligned_bgr, meta


def apply_x_alignment_with_reference(
    crop_bgr,
    side_name,
    artifacts_dir,
    create_reference_if_missing=True,
    debug_save_path=None,
):
    """
    Creates/loads X-edge reference for tread/inner/bead.

    During calibration:
        first crop creates reference edges.

    During inference:
        saved reference edges are loaded and current image is shifted in X.
    """

    side_name = side_name.lower().strip()

    # --------------------------------------------------------
    # Tread uses profile-based X alignment.
    # This avoids unreliable left/right edge bbox detection.
    # --------------------------------------------------------
    if side_name == "tread":
        return apply_tread_profile_x_alignment(
            crop_bgr=crop_bgr,
            artifacts_dir=artifacts_dir,
            create_reference_if_missing=create_reference_if_missing,
            debug_save_path=debug_save_path,
        )
    
    os.makedirs(artifacts_dir, exist_ok=True)

    configure_xalign_for_side(side_name)

    ref_edges_path = os.path.join(artifacts_dir, f"{side_name}_x_ref_edges.json")

    if create_reference_if_missing and not os.path.exists(ref_edges_path):
        ref_edges = xalign.detect_tire_edges(
            crop_bgr,
            side_name=f"{side_name}_x_reference",
        )

        # --------------------------------------------------------
        # Save reference edge JSON
        # This is used during production/inference alignment.
        # --------------------------------------------------------
        with open(ref_edges_path, "w", encoding="utf-8") as f:
            json.dump(ref_edges, f, indent=4)

        # --------------------------------------------------------
        # Save reference cropped image
        # This is for visual validation/debug.
        # --------------------------------------------------------
        ref_crop_path = os.path.join(
            artifacts_dir,
            f"{side_name}_x_reference_crop.png"
        )
        cv2.imwrite(ref_crop_path, crop_bgr)

        # --------------------------------------------------------
        # Save reference bbox overlay
        # This lets you confirm whether the detected edge is correct.
        # --------------------------------------------------------
        ref_bbox_path = os.path.join(
            artifacts_dir,
            f"{side_name}_x_reference_bbox.png"
        )

        try:
            ref_overlay = xalign.draw_bbox_overlay(
                crop_bgr,
                edges=ref_edges,
                ref_edges=None,
                shift_x=0,
                title=f"{side_name.upper()} X REFERENCE",
            )
            cv2.imwrite(ref_bbox_path, ref_overlay)
        except Exception as e:
            print(f"[{side_name.upper()} XALIGN][WARN] reference bbox save failed: {e}")
            ref_bbox_path = None

        x_meta = {
            "status": "ok",
            "mode": "created_x_reference",
            "side": side_name,
            "ref_edges_path": ref_edges_path,
            "ref_crop_path": ref_crop_path,
            "ref_bbox_path": ref_bbox_path,
            "ref_edges": ref_edges,
            "final_shift_x": 0,
        }

        print(f"[{side_name.upper()} XALIGN] Created reference edges: {ref_edges_path}")
        print(f"[{side_name.upper()} XALIGN] Saved reference crop : {ref_crop_path}")
        print(f"[{side_name.upper()} XALIGN] Saved reference bbox : {ref_bbox_path}")

        return crop_bgr, x_meta

    if not os.path.exists(ref_edges_path):
        raise RuntimeError(
            f"[{side_name}] X reference edges not found: {ref_edges_path}. "
            f"Run calibration first."
        )

    with open(ref_edges_path, "r", encoding="utf-8") as f:
        ref_edges = json.load(f)

    aligned_bgr, x_info = xalign.align_to_reference_edges(
        img=crop_bgr,
        ref_edges=ref_edges,
        side_name=side_name,
    )

    x_info["ref_edges_path"] = ref_edges_path

    if debug_save_path:
        try:
            os.makedirs(os.path.dirname(debug_save_path), exist_ok=True)
            overlay = xalign.draw_bbox_overlay(
                crop_bgr,
                edges=x_info["current_detector_info"],
                ref_edges=ref_edges,
                shift_x=x_info["final_shift_x"],
                title=f"{side_name.upper()} BEFORE XALIGN",
            )
            cv2.imwrite(debug_save_path, overlay)
        except Exception as e:
            print(f"[{side_name.upper()} XALIGN][WARN] debug save failed: {e}")

    print(
        f"[{side_name.upper()} XALIGN] "
        f"shift_x={x_info['final_shift_x']} | "
        f"before_LR=({x_info['before_left_x']},{x_info['before_right_x']}) | "
        f"ref_LR=({x_info['ref_left_x']},{x_info['ref_right_x']})"
    )

    return aligned_bgr, x_info


# ============================================================
# MAIN HELPER USED BY TREAD / INNER / BEAD PIPELINES
# ============================================================

def crop_resize_xalign_non_r_side(
    pre_bgr,
    side_name,
    sidewall_r_anchor,
    offset_ratio,
    target_size,
    artifacts_dir,
    create_x_reference_if_missing=True,
    debug_save_path=None,
):
    crop_bgr, crop_meta = crop_resize_by_sidewall_anchor(
        pre_bgr=pre_bgr,
        side_name=side_name,
        sidewall_r_anchor=sidewall_r_anchor,
        offset_ratio=offset_ratio,
        target_size=target_size,
    )

    aligned_bgr, x_meta = apply_x_alignment_with_reference(
        crop_bgr=crop_bgr,
        side_name=side_name,
        artifacts_dir=artifacts_dir,
        create_reference_if_missing=create_x_reference_if_missing,
        debug_save_path=debug_save_path,
    )

    meta = {
        "status": "ok",
        "side": side_name,
        "crop_meta": crop_meta,
        "x_align_meta": x_meta,
    }

    return aligned_bgr, meta