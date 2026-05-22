import os
import csv
import cv2
import numpy as np
from pathlib import Path


# ============================================================
# USER SETTINGS
# ============================================================

SIDE_NAME = "tread"   # tread / bead / innerwall

INPUT_DIR = r"C:\Users\eyres\Desktop\Apollo\Apollo_Data\Final_Test\bad\254901432\Good\R_Crop_Output\crop"
OUTPUT_DIR = r"C:\Users\eyres\Desktop\Apollo\Apollo_Data\Final_Test\bad\254901432\Good\R_Crop_Output\crop\x_aligned_first_ref"

# If None, first image in INPUT_DIR is used as reference.
REFERENCE_IMAGE_PATH = None

SAVE_DEBUG_BBOX = True

# If True, images with bad edge validation are skipped.
# If False, warning is printed but alignment is still saved.
STRICT_VALIDATION = False

IMAGE_EXTENSIONS = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]


# ============================================================
# COMMON EDGE DETECTION SETTINGS
# ============================================================

# Use middle portion of image height for edge detection.
# Debug bbox is still drawn on full height.
ROW_ROI_RATIO = (0.10, 0.90)

# Validation limits.
MAX_WIDTH_ERROR_PX = 250
MAX_SHIFT_DISAGREEMENT_PX = 150

# Border color after X shift.
BORDER_VALUE = 0


# ============================================================
# METHOD SELECTION
# ============================================================
# For tread, texture method is better because side strips can confuse
# foreground-based detection.
#
# For bead/innerwall, foreground may still work. If they fail, try "texture".
# Options:
#   "texture"
#   "foreground"
# ============================================================

DETECTION_METHOD_BY_SIDE = {
    "tread": "texture",
    "bead": "foreground",
    "innerwall": "foreground",
}


# ============================================================
# FOREGROUND METHOD SETTINGS
# Used for bead / innerwall, or tread if selected.
# ============================================================

BACKGROUND_STRIP_RATIO = 0.05
BACKGROUND_SIGMA_K = 2.0

MIN_COLUMN_OCCUPANCY = 0.05

# For full resolution 2048/4096 width images, 40-100 is safer.
MIN_COMPONENT_WIDTH_PX = 80

# "largest"   = select widest foreground region
# "outermost" = select first foreground region to last foreground region
COMPONENT_SELECT_MODE = "largest"

OCCUPANCY_SMOOTH_KERNEL = 9


# ============================================================
# TREAD TEXTURE METHOD SETTINGS
# ============================================================
# Tread has horizontal grooves, so Sobel-Y texture energy works better.
# This only affects edge detection. Output image remains raw shifted image.
# ============================================================

TEXTURE_SMOOTH_KERNEL = 15

# Lower = wider bbox, higher = tighter bbox.
TEXTURE_THRESHOLD_RATIO = 0.08

# Minimum texture run width as fraction of image width.
TEXTURE_MIN_RUN_WIDTH_RATIO = 0.02

# For tread image, ignore left-side non-tread strip.
# If left bbox is still too far left, increase 0.14 -> 0.18 -> 0.22.
# If left bbox becomes too far right, decrease 0.14 -> 0.10.
TREAD_SEARCH_X_RATIO = (0.1,0.9)                       #(0.2, 0.92)

# First keep this False. Sobel-X refinement can snap back to the wrong side strip.
TEXTURE_REFINE_WITH_SOBEL_X = False


# ============================================================
# SOBEL REFINEMENT SETTINGS
# ============================================================

REFINE_WINDOW_PX = 35
SOBEL_X_SMOOTH_KERNEL = 15
EDGE_THRESHOLD_RATIO = 0.20


# ============================================================
# BASIC HELPERS
# ============================================================

def ensure_odd(v: int) -> int:
    v = int(v)
    if v < 3:
        v = 3
    if v % 2 == 0:
        v += 1
    return v


def list_images(folder: str):
    return sorted([
        f for f in os.listdir(folder)
        if Path(f).suffix.lower() in IMAGE_EXTENSIONS
    ])


def read_image(path: str):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise RuntimeError(f"Could not read image: {path}")
    return img


def save_image(path: str, img: np.ndarray):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ok = cv2.imwrite(path, img)
    if not ok:
        raise RuntimeError(f"Failed to save image: {path}")


def to_gray_float32(img: np.ndarray) -> np.ndarray:
    """
    Grayscale conversion is used only for edge detection.
    Final output image is not preprocessed.
    """
    if img.ndim == 2:
        gray = img
    elif img.ndim == 3 and img.shape[2] == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif img.ndim == 3 and img.shape[2] == 4:
        gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    else:
        raise ValueError(f"Unsupported image shape: {img.shape}")

    return gray.astype(np.float32)


def get_roi(gray: np.ndarray):
    h, w = gray.shape[:2]

    y1 = int(h * ROW_ROI_RATIO[0])
    y2 = int(h * ROW_ROI_RATIO[1])

    y1 = max(0, min(y1, h - 1))
    y2 = max(y1 + 1, min(y2, h))

    return gray[y1:y2, :], y1, y2


# ============================================================
# RUN / COMPONENT HELPERS
# ============================================================

def find_runs(mask_1d: np.ndarray):
    """
    Returns continuous True runs: [(start, end), ...]
    end is inclusive.
    """
    runs = []
    in_run = False
    start = 0

    for i, v in enumerate(mask_1d):
        if bool(v) and not in_run:
            in_run = True
            start = i
        elif not bool(v) and in_run:
            runs.append((start, i - 1))
            in_run = False

    if in_run:
        runs.append((start, len(mask_1d) - 1))

    return runs


def select_component_from_runs(runs, image_width: int, mode: str):
    """
    Selects the region that should be treated as the tire/tread body.

    mode="largest":
        selects widest run.

    mode="outermost":
        selects first run start and last run end.
    """
    if not runs:
        raise RuntimeError("No runs available for component selection")

    mode = mode.lower().strip()

    if mode == "largest":
        selected = max(runs, key=lambda r: r[1] - r[0] + 1)
        return int(selected[0]), int(selected[1]), selected

    if mode == "outermost":
        selected = (runs[0][0], runs[-1][1])
        return int(runs[0][0]), int(runs[-1][1]), selected

    raise ValueError("COMPONENT_SELECT_MODE must be 'largest' or 'outermost'")


# ============================================================
# SOBEL-X REFINEMENT
# ============================================================

def get_sobel_x_profile(gray_roi: np.ndarray) -> np.ndarray:
    """
    Sobel X detects vertical edges.
    Used only to refine edge position.
    """
    sobel_x = cv2.Sobel(
        gray_roi,
        cv2.CV_32F,
        dx=1,
        dy=0,
        ksize=3
    )

    sobel_x = np.abs(sobel_x)

    profile = np.median(sobel_x, axis=0).astype(np.float32)

    k = ensure_odd(SOBEL_X_SMOOTH_KERNEL)

    profile = cv2.GaussianBlur(
        profile.reshape(1, -1),
        (k, 1),
        0
    ).flatten()

    return profile


def refine_outer_edge(profile: np.ndarray, rough_x: int, side: str) -> int:
    """
    Refines rough edge using Sobel-X profile.

    left:
        search around rough boundary and pick first valid strong edge.

    right:
        search around rough boundary and pick last valid strong edge.
    """
    w = len(profile)
    win = int(REFINE_WINDOW_PX)

    x1 = max(0, int(rough_x) - win)
    x2 = min(w - 1, int(rough_x) + win)

    if x2 <= x1:
        return int(rough_x)

    window = profile[x1:x2 + 1]

    max_val = float(np.max(window))
    med_val = float(np.median(window))

    if max_val <= 0:
        return int(rough_x)

    threshold = med_val + EDGE_THRESHOLD_RATIO * (max_val - med_val)
    valid = np.where(window >= threshold)[0]

    if len(valid) == 0:
        return int(rough_x)

    if side == "left":
        refined = x1 + int(valid[0])
    elif side == "right":
        refined = x1 + int(valid[-1])
    else:
        raise ValueError("side must be 'left' or 'right'")

    return int(refined)


# ============================================================
# METHOD 1: FOREGROUND-BASED EDGE DETECTION
# ============================================================

def estimate_background_threshold(roi: np.ndarray) -> float:
    """
    Estimate dark background level from left/right border strips.
    """
    h, w = roi.shape[:2]

    strip_w = max(3, int(w * BACKGROUND_STRIP_RATIO))
    strip_w = min(strip_w, max(3, w // 4))

    bg_pixels = np.concatenate([
        roi[:, :strip_w].reshape(-1),
        roi[:, w - strip_w:].reshape(-1)
    ])

    bg_med = float(np.median(bg_pixels))
    mad = float(np.median(np.abs(bg_pixels - bg_med)))
    robust_sigma = 1.4826 * mad

    if robust_sigma < 1.0:
        robust_sigma = float(np.std(bg_pixels))

    if robust_sigma < 1.0:
        robust_sigma = 1.0

    threshold = bg_med + BACKGROUND_SIGMA_K * robust_sigma
    return threshold


def detect_edges_by_foreground(img: np.ndarray, side_name: str):
    """
    Detects edges using foreground/background separation.

    Best for bead/innerwall if the tire is clearly separated from background.
    """
    gray = to_gray_float32(img)
    h, w = gray.shape[:2]

    roi, roi_y1, roi_y2 = get_roi(gray)

    bg_threshold = estimate_background_threshold(roi)

    foreground = roi > bg_threshold

    occupancy = np.mean(foreground, axis=0).astype(np.float32)

    k = ensure_odd(OCCUPANCY_SMOOTH_KERNEL)

    occupancy_smooth = cv2.GaussianBlur(
        occupancy.reshape(1, -1),
        (k, 1),
        0
    ).flatten()

    valid_cols = occupancy_smooth >= MIN_COLUMN_OCCUPANCY

    runs = find_runs(valid_cols)

    runs = [
        (s, e)
        for s, e in runs
        if (e - s + 1) >= MIN_COMPONENT_WIDTH_PX
    ]

    if not runs:
        raise RuntimeError(
            f"[{side_name}] No foreground runs found. "
            f"Check contrast / MIN_COLUMN_OCCUPANCY / MIN_COMPONENT_WIDTH_PX."
        )

    rough_left, rough_right, selected_run = select_component_from_runs(
        runs=runs,
        image_width=w,
        mode=COMPONENT_SELECT_MODE
    )

    sobel_profile = get_sobel_x_profile(roi)

    left_edge = refine_outer_edge(
        profile=sobel_profile,
        rough_x=rough_left,
        side="left"
    )

    right_edge = refine_outer_edge(
        profile=sobel_profile,
        rough_x=rough_right,
        side="right"
    )

    if right_edge <= left_edge:
        raise RuntimeError(
            f"[{side_name}] Invalid foreground edges: "
            f"left={left_edge}, right={right_edge}"
        )

    return {
        "edge_method": "foreground",
        "left_edge_x": int(left_edge),
        "right_edge_x": int(right_edge),
        "edge_width": int(right_edge - left_edge),

        "rough_left_x": int(rough_left),
        "rough_right_x": int(rough_right),
        "selected_run": [int(selected_run[0]), int(selected_run[1])],

        "roi_y1": int(roi_y1),
        "roi_y2": int(roi_y2),

        "bg_threshold": float(bg_threshold),

        "image_height": int(h),
        "image_width": int(w),

        "runs": [[int(s), int(e)] for s, e in runs],
    }


# ============================================================
# METHOD 2: TREAD TEXTURE-BASED EDGE DETECTION
# ============================================================

def detect_edges_by_tread_texture(img: np.ndarray, side_name: str):
    """
    Tread-specific edge detection.

    Uses Sobel-Y texture energy, but searches only inside a tread search zone.
    This avoids selecting the left side strip / non-tread band as the left edge.
    """
    gray = to_gray_float32(img)
    h, w = gray.shape[:2]

    roi, roi_y1, roi_y2 = get_roi(gray)

    # ------------------------------------------------------------
    # Limit X search zone for tread
    # ------------------------------------------------------------
    search_x1 = int(w * TREAD_SEARCH_X_RATIO[0])
    search_x2 = int(w * TREAD_SEARCH_X_RATIO[1])

    search_x1 = max(0, min(search_x1, w - 2))
    search_x2 = max(search_x1 + 1, min(search_x2, w - 1))

    roi_search = roi[:, search_x1:search_x2]

    # ------------------------------------------------------------
    # Sobel-Y detects horizontal tread grooves
    # ------------------------------------------------------------
    sobel_y = cv2.Sobel(
        roi_search,
        cv2.CV_32F,
        dx=0,
        dy=1,
        ksize=3
    )

    sobel_y = np.abs(sobel_y)

    texture_profile = np.median(sobel_y, axis=0).astype(np.float32)

    k = ensure_odd(TEXTURE_SMOOTH_KERNEL)

    texture_profile = cv2.GaussianBlur(
        texture_profile.reshape(1, -1),
        (k, 1),
        0
    ).flatten()

    base_val = float(np.percentile(texture_profile, 20))
    high_val = float(np.percentile(texture_profile, 95))

    if high_val <= base_val:
        high_val = float(np.max(texture_profile))

    if high_val <= base_val:
        raise RuntimeError(
            f"[{side_name}] Texture profile has no useful variation."
        )

    threshold = base_val + TEXTURE_THRESHOLD_RATIO * (high_val - base_val)

    valid_cols = texture_profile >= threshold

    min_run_width = max(5, int(w * TEXTURE_MIN_RUN_WIDTH_RATIO))

    runs_local = find_runs(valid_cols)

    runs_local = [
        (s, e)
        for s, e in runs_local
        if (e - s + 1) >= min_run_width
    ]

    if not runs_local:
        raise RuntimeError(
            f"[{side_name}] No texture runs found. "
            f"Try lowering TEXTURE_THRESHOLD_RATIO or reducing TREAD_SEARCH_X_RATIO left value."
        )

    # ------------------------------------------------------------
    # Convert local run coordinates back to full image X coordinates
    # ------------------------------------------------------------
    rough_left = search_x1 + runs_local[0][0]
    rough_right = search_x1 + runs_local[-1][1]

    # ------------------------------------------------------------
    # Optional refinement
    # For your current image, keep TEXTURE_REFINE_WITH_SOBEL_X = False first.
    # ------------------------------------------------------------
    if TEXTURE_REFINE_WITH_SOBEL_X:
        sobel_x_profile = get_sobel_x_profile(roi)

        left_edge = refine_outer_edge(
            profile=sobel_x_profile,
            rough_x=rough_left,
            side="left"
        )

        right_edge = refine_outer_edge(
            profile=sobel_x_profile,
            rough_x=rough_right,
            side="right"
        )

        # Clamp refined result so it does not jump outside tread search zone.
        left_edge = max(search_x1, min(left_edge, search_x2))
        right_edge = max(search_x1, min(right_edge, search_x2))
    else:
        left_edge = rough_left
        right_edge = rough_right

    if right_edge <= left_edge:
        raise RuntimeError(
            f"[{side_name}] Invalid texture edges: "
            f"left={left_edge}, right={right_edge}"
        )

    return {
        "edge_method": "tread_texture_sobel_y_search_zone",
        "left_edge_x": int(left_edge),
        "right_edge_x": int(right_edge),
        "edge_width": int(right_edge - left_edge),

        "rough_left_x": int(rough_left),
        "rough_right_x": int(rough_right),

        "search_x1": int(search_x1),
        "search_x2": int(search_x2),

        "roi_y1": int(roi_y1),
        "roi_y2": int(roi_y2),

        "texture_threshold": float(threshold),
        "texture_base": float(base_val),
        "texture_high": float(high_val),

        "image_height": int(h),
        "image_width": int(w),

        "runs": [
            [int(search_x1 + s), int(search_x1 + e)]
            for s, e in runs_local
        ],
    }

# ============================================================
# MAIN EDGE DETECTION ROUTER
# ============================================================

def detect_tire_edges(img: np.ndarray, side_name: str = "camera"):
    """
    Detects left/right edges based on selected side method.
    """
    side_key = SIDE_NAME.lower().strip()
    method = DETECTION_METHOD_BY_SIDE.get(side_key, "foreground").lower().strip()

    if method == "texture":
        return detect_edges_by_tread_texture(img, side_name=side_name)

    if method == "foreground":
        return detect_edges_by_foreground(img, side_name=side_name)

    raise ValueError(
        f"Invalid detection method '{method}' for SIDE_NAME='{SIDE_NAME}'. "
        f"Use 'texture' or 'foreground'."
    )


# ============================================================
# X SHIFT ALIGNMENT
# ============================================================

def shift_image_x_same_size(img: np.ndarray, shift_x: int, border_value: int = 0):
    """
    Moves image left/right only.

    No resize.
    No scaling.
    No crop.
    Output size remains same.
    """
    h, w = img.shape[:2]

    M = np.float32([
        [1, 0, int(shift_x)],
        [0, 1, 0]
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
        borderValue=border
    )

    return aligned


def align_to_reference_edges(img: np.ndarray, ref_edges: dict, side_name: str = "camera"):
    """
    Detect current edges and align to reference edges.
    """
    cur_edges = detect_tire_edges(img, side_name=side_name)

    ref_left = int(ref_edges["left_edge_x"])
    ref_right = int(ref_edges["right_edge_x"])
    ref_width = int(ref_edges["edge_width"])

    cur_left = int(cur_edges["left_edge_x"])
    cur_right = int(cur_edges["right_edge_x"])
    cur_width = int(cur_edges["edge_width"])

    left_shift = ref_left - cur_left
    right_shift = ref_right - cur_right

    width_error = abs(cur_width - ref_width)
    shift_disagreement = abs(left_shift - right_shift)

    warnings = []

    if width_error > MAX_WIDTH_ERROR_PX:
        warnings.append(
            f"width_error={width_error} > {MAX_WIDTH_ERROR_PX}"
        )

    if shift_disagreement > MAX_SHIFT_DISAGREEMENT_PX:
        warnings.append(
            f"shift_disagreement={shift_disagreement} > {MAX_SHIFT_DISAGREEMENT_PX}"
        )

    if warnings and STRICT_VALIDATION:
        raise RuntimeError(
            f"[{side_name}] Validation failed: " + " | ".join(warnings)
        )

    shift_x = int(round((left_shift + right_shift) / 2.0))

    aligned = shift_image_x_same_size(
        img=img,
        shift_x=shift_x,
        border_value=BORDER_VALUE
    )

    after_left_expected = cur_left + shift_x
    after_right_expected = cur_right + shift_x

    try:
        after_edges = detect_tire_edges(
            aligned,
            side_name=side_name + "_after"
        )
        after_left_detected = int(after_edges["left_edge_x"])
        after_right_detected = int(after_edges["right_edge_x"])
    except Exception:
        after_edges = None
        after_left_detected = after_left_expected
        after_right_detected = after_right_expected

    info = {
        "ref_left_x": ref_left,
        "ref_right_x": ref_right,
        "ref_width": ref_width,

        "before_left_x": cur_left,
        "before_right_x": cur_right,
        "before_width": cur_width,

        "left_shift": int(left_shift),
        "right_shift": int(right_shift),
        "final_shift_x": int(shift_x),

        "after_left_expected": int(after_left_expected),
        "after_right_expected": int(after_right_expected),

        "after_left_detected": int(after_left_detected),
        "after_right_detected": int(after_right_detected),
        "after_width_detected": int(after_right_detected - after_left_detected),

        "after_left_error": int(abs(after_left_detected - ref_left)),
        "after_right_error": int(abs(after_right_detected - ref_right)),

        "width_error": int(width_error),
        "shift_disagreement": int(shift_disagreement),

        "input_shape": list(img.shape),
        "output_shape": list(aligned.shape),
        "same_height": bool(img.shape[0] == aligned.shape[0]),
        "same_width": bool(img.shape[1] == aligned.shape[1]),

        "warnings": warnings,
        "current_detector_info": cur_edges,
        "after_detector_info": after_edges
    }

    return aligned, info


# ============================================================
# DEBUG BOUNDING BOX
# ============================================================

def draw_bbox_overlay(
    img: np.ndarray,
    edges: dict,
    ref_edges: dict = None,
    shift_x: int = 0,
    title: str = "",
    draw_full_height: bool = True,
    make_color_debug: bool = True
):
    """
    Draws bounding boxes on original image.

    No normalization is applied.
    Grayscale is converted to BGR only so colored boxes can be drawn.
    """
    if make_color_debug:
        if img.ndim == 2:
            overlay = cv2.cvtColor(img.copy(), cv2.COLOR_GRAY2BGR)
        else:
            overlay = img.copy()
    else:
        overlay = img.copy()

    h, w = overlay.shape[:2]

    if draw_full_height:
        roi_y1 = 0
        roi_y2 = h - 1
    else:
        roi_y1 = int(edges.get("roi_y1", 0))
        roi_y2 = int(edges.get("roi_y2", h - 1))

    if overlay.dtype == np.uint16:
        max_val = 65535
    else:
        max_val = 255

    def get_color(color_name):
        if make_color_debug:
            if color_name == "blue":
                return (max_val, 0, 0)
            if color_name == "green":
                return (0, max_val, 0)
            if color_name == "yellow":
                return (0, max_val, max_val)
            if color_name == "red":
                return (0, 0, max_val)
            return (max_val, max_val, max_val)
        return max_val

    def draw_rect(left, right, color, label):
        left = int(left)
        right = int(right)

        left = max(0, min(left, w - 1))
        right = max(0, min(right, w - 1))

        cv2.rectangle(
            overlay,
            (left, roi_y1),
            (right, roi_y2),
            color,
            2
        )

        try:
            cv2.putText(
                overlay,
                label,
                (max(5, min(w - 250, left + 5)), max(30, roi_y1 + 30)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2,
                cv2.LINE_AA
            )
        except Exception:
            pass

    if ref_edges is not None:
        draw_rect(
            ref_edges["left_edge_x"],
            ref_edges["right_edge_x"],
            get_color("blue"),
            "REF"
        )

    draw_rect(
        edges["left_edge_x"],
        edges["right_edge_x"],
        get_color("green"),
        "CUR"
    )

    if shift_x != 0:
        draw_rect(
            edges["left_edge_x"] + shift_x,
            edges["right_edge_x"] + shift_x,
            get_color("yellow"),
            "AFTER_EXPECTED"
        )

    if title:
        try:
            cv2.putText(
                overlay,
                title,
                (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                get_color("red"),
                2,
                cv2.LINE_AA
            )
        except Exception:
            pass

    return overlay


# ============================================================
# MAIN
# ============================================================

def run_alignment():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    aligned_dir = os.path.join(OUTPUT_DIR, "aligned")
    debug_before_dir = os.path.join(OUTPUT_DIR, "debug_before")
    debug_after_dir = os.path.join(OUTPUT_DIR, "debug_after")

    os.makedirs(aligned_dir, exist_ok=True)

    if SAVE_DEBUG_BBOX:
        os.makedirs(debug_before_dir, exist_ok=True)
        os.makedirs(debug_after_dir, exist_ok=True)

    files = list_images(INPUT_DIR)

    if not files:
        raise RuntimeError(f"No images found in INPUT_DIR: {INPUT_DIR}")

    if REFERENCE_IMAGE_PATH is None:
        ref_path = os.path.join(INPUT_DIR, files[0])
        ref_file_name = files[0]
    else:
        ref_path = REFERENCE_IMAGE_PATH
        ref_file_name = os.path.basename(REFERENCE_IMAGE_PATH)

    ref_img = read_image(ref_path)

    ref_edges = detect_tire_edges(
        ref_img,
        side_name=SIDE_NAME + "_reference"
    )

    print("\n================ REFERENCE IMAGE ================")
    print("Reference file :", ref_file_name)
    print("Reference path :", ref_path)
    print("Side           :", SIDE_NAME)
    print("Method         :", DETECTION_METHOD_BY_SIDE.get(SIDE_NAME.lower(), "foreground"))
    print("Ref Left X     :", ref_edges["left_edge_x"])
    print("Ref Right X    :", ref_edges["right_edge_x"])
    print("Ref Width      :", ref_edges["edge_width"])
    print("Image shape    :", ref_img.shape)

    if SAVE_DEBUG_BBOX:
        ref_debug = draw_bbox_overlay(
            ref_img,
            edges=ref_edges,
            ref_edges=None,
            shift_x=0,
            title="REFERENCE"
        )

        ref_debug_path = os.path.join(
            OUTPUT_DIR,
            f"reference_bbox_{Path(ref_file_name).stem}.png"
        )
        save_image(ref_debug_path, ref_debug)

    csv_path = os.path.join(OUTPUT_DIR, f"{SIDE_NAME}_x_alignment_log.csv")

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        writer.writerow([
            "file",
            "status",
            "method",
            "ref_left_x",
            "ref_right_x",
            "ref_width",
            "before_left_x",
            "before_right_x",
            "before_width",
            "left_shift",
            "right_shift",
            "final_shift_x",
            "after_left_expected",
            "after_right_expected",
            "after_left_detected",
            "after_right_detected",
            "after_left_error",
            "after_right_error",
            "width_error",
            "shift_disagreement",
            "same_height",
            "same_width",
            "warnings",
            "message"
        ])

        for file in files:
            img_path = os.path.join(INPUT_DIR, file)

            try:
                img = read_image(img_path)

                aligned, info = align_to_reference_edges(
                    img=img,
                    ref_edges=ref_edges,
                    side_name=SIDE_NAME
                )

                save_path = os.path.join(aligned_dir, file)
                save_image(save_path, aligned)

                if SAVE_DEBUG_BBOX:
                    before_overlay = draw_bbox_overlay(
                        img,
                        edges=info["current_detector_info"],
                        ref_edges=ref_edges,
                        shift_x=info["final_shift_x"],
                        title="BEFORE"
                    )

                    before_debug_path = os.path.join(
                        debug_before_dir,
                        Path(file).stem + "_before_bbox.png"
                    )
                    save_image(before_debug_path, before_overlay)

                    after_edges = info["after_detector_info"]

                    if after_edges is not None:
                        after_overlay = draw_bbox_overlay(
                            aligned,
                            edges=after_edges,
                            ref_edges=ref_edges,
                            shift_x=0,
                            title="AFTER"
                        )
                    else:
                        after_edges_fallback = {
                            "left_edge_x": info["after_left_expected"],
                            "right_edge_x": info["after_right_expected"],
                            "roi_y1": info["current_detector_info"]["roi_y1"],
                            "roi_y2": info["current_detector_info"]["roi_y2"]
                        }

                        after_overlay = draw_bbox_overlay(
                            aligned,
                            edges=after_edges_fallback,
                            ref_edges=ref_edges,
                            shift_x=0,
                            title="AFTER_EXPECTED_ONLY"
                        )

                    after_debug_path = os.path.join(
                        debug_after_dir,
                        Path(file).stem + "_after_bbox.png"
                    )
                    save_image(after_debug_path, after_overlay)

                print("\n----------------------------")
                print("Image                 :", file)
                print("Before L/R/W          :",
                      info["before_left_x"],
                      info["before_right_x"],
                      info["before_width"])
                print("Reference L/R/W       :",
                      info["ref_left_x"],
                      info["ref_right_x"],
                      info["ref_width"])
                print("Left shift / Right    :",
                      info["left_shift"],
                      info["right_shift"])
                print("Final shift X         :", info["final_shift_x"])
                print("After expected L/R    :",
                      info["after_left_expected"],
                      info["after_right_expected"])
                print("After detected L/R    :",
                      info["after_left_detected"],
                      info["after_right_detected"])
                print("After error L/R       :",
                      info["after_left_error"],
                      info["after_right_error"])
                print("Width error           :", info["width_error"])
                print("Shift disagreement    :", info["shift_disagreement"])
                print("Same H/W              :",
                      info["same_height"],
                      info["same_width"])

                if info["warnings"]:
                    print("⚠ Warnings            :", " | ".join(info["warnings"]))

                print("Saved aligned         :", save_path)

                writer.writerow([
                    file,
                    "OK_WITH_WARNING" if info["warnings"] else "OK",
                    info["current_detector_info"].get("edge_method", ""),
                    info["ref_left_x"],
                    info["ref_right_x"],
                    info["ref_width"],
                    info["before_left_x"],
                    info["before_right_x"],
                    info["before_width"],
                    info["left_shift"],
                    info["right_shift"],
                    info["final_shift_x"],
                    info["after_left_expected"],
                    info["after_right_expected"],
                    info["after_left_detected"],
                    info["after_right_detected"],
                    info["after_left_error"],
                    info["after_right_error"],
                    info["width_error"],
                    info["shift_disagreement"],
                    info["same_height"],
                    info["same_width"],
                    " | ".join(info["warnings"]),
                    ""
                ])

            except Exception as e:
                print("\n[FAILED]", file, str(e))

                writer.writerow([
                    file,
                    "FAILED",
                    "",
                    ref_edges["left_edge_x"],
                    ref_edges["right_edge_x"],
                    ref_edges["edge_width"],
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    str(e)
                ])

    print("\n================ DONE ================")
    print("Output folder :", OUTPUT_DIR)
    print("Aligned images:", aligned_dir)
    print("CSV log       :", csv_path)

    if SAVE_DEBUG_BBOX:
        print("Debug before  :", debug_before_dir)
        print("Debug after   :", debug_after_dir)


if __name__ == "__main__":
    run_alignment()