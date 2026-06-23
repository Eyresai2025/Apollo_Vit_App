import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import json
import time
import threading
import shutil
from datetime import datetime
from typing import Any, Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import numpy as np
import pandas as pd
import torch

from src.models.Pipeline import inference_pipeline_bead_mahal_pca as bead
from src.models.Pipeline import inference_pipeline_innerwall_mahal_pca as innerwall
from src.models.Pipeline import inference_pipeline_sidewall1_mahal_pca as sidewall1
from src.models.Pipeline import inference_pipeline_sidewall2_mahal_pca as sidewall2
from src.models.Pipeline import inference_pipeline_tread_mahal_pca as tread
from src.models.Pipeline.yolo_patch_classifier import load_yolo_seg
from src.COMMON.exceptions import ModelLoadError
from src.COMMON.structured_logging import get_logger

logger = get_logger(__name__, component="AI_PIPELINE")

# Attempt to load optional modules with proper error handling
build_r_detector = None
try:
    from src.models.Pipeline.R_Detection_align_crop import build_r_detector
except ImportError as e:
    logger.warning(f"R-Detection module not available (ImportError): {e}")
except Exception as e:
    logger.error(f"Unexpected error loading R-Detection module: {e}")

TRTViTFeatureExtractor = None
try:
    from src.models.Pipeline.vit_trt_inference import TRTViTFeatureExtractor
except ImportError as e:
    logger.warning(f"TensorRT ViT inference not available (ImportError): {e}")
except Exception as e:
    logger.error(f"Unexpected error loading TensorRT ViT: {e}")

from src.camera.HARDWARE_TRIGGER import (
    get_camera_to_side_map,
    get_side_to_camera_map,
)

# Optional live inspection state
def set_live_progress(*args, **kwargs):
    """Default no-op for live progress when module unavailable."""
    pass

try:
    from src.COMMON.live_inspection_state import set_live_progress
except ImportError as e:
    logger.debug(f"Live inspection state module not available: {e}")
except Exception as e:
    logger.warning(f"Failed to import live_inspection_state: {e}")

# Optional R-inner mapping alignment
extract_sidewall_r_anchor_from_meta = None
try:
    from src.models.Pipeline.R_inner_mapping_alignment import (
        extract_sidewall_r_anchor_from_meta,
    )
except ImportError as e:
    logger.debug(f"R-inner mapping alignment not available: {e}")
except Exception as e:
    logger.warning(f"Failed to import R-inner mapping alignment: {e}")

# =========================================================
# THREAD OPTIMIZATION
# =========================================================
try:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    logger.debug("PyTorch thread optimization applied")
except Exception as e:
    logger.warning(f"Failed to optimize PyTorch threads: {e}")

try:
    cv2.setNumThreads(0)
    logger.debug("OpenCV thread optimization applied")
except Exception as e:
    logger.warning(f"Failed to optimize OpenCV threads: {e}")


# =========================================================
# CONFIGURATION & GLOBALS
# =========================================================
from src.COMMON.config import get_config

_config = get_config()

# Load from centralized configuration
DEVICE = _config.inference.device.value
PARALLEL_INFER = True
PARALLEL_CALIB = False
ENABLE_WARMUP = _config.inference.enable_warmup

INFER_SIDE_WORKERS = 5
CALIB_SIDE_WORKERS = 1

R_ALIGN_GPU_CONCURRENCY = _config.inference.r_align_gpu_concurrency
VIT_GPU_CONCURRENCY = _config.inference.vit_gpu_concurrency
YOLO_GPU_CONCURRENCY = _config.inference.yolo_gpu_concurrency

USE_SHARED_R_DETECTOR = _config.inference.use_shared_r_detector
SAVE_CYCLE_SUMMARY = _config.inference.save_cycle_summary
DEFAULT_TYRE_NAME = _config.inference.default_tyre_name
DEFAULT_USE_YOLO_SEG = _config.inference.use_yolo_seg
SEG_IMGSZ = _config.inference.seg_imgsz

ENABLE_STAGE_PIPELINE = True
PIPELINE_FALLBACK_TO_INFER_SINGLE = False
ENABLE_TRT_VIT = True
CLEAN_YOLO_CACHE = True

CAMERA_CAPTURE_ENABLED = True
CAPTURE_IMAGE_FORMAT = ".png"
CAPTURE_JPEG_QUALITY = 95

# =========================================================
# SKU / PER-SIDE CALIBRATION ARTIFACT HELPERS
# Final structure:
#   media/AI_Calibration_Files/<SKU>/calibration_<side>/artifacts
# =========================================================

CALIBRATION_ROOT_DIR_NAME = "AI_Calibration_Files"

SIDE_CALIBRATION_DIRS = {
    "sidewall1": "calibration_sidewall1",
    "sidewall2": "calibration_sidewall2",
    "innerwall": "calibration_innerwall",
    "tread": "calibration_tread",
    "bead": "calibration_bead",
} 


# =========================================================
# SIDE MODULES / ORDER / CAMERA MAP
# =========================================================
SIDE_MODULES = {
    "innerwall": innerwall,
    "sidewall1": sidewall1,
    "sidewall2": sidewall2,
    "tread": tread,
    "bead": bead,
}

# DEFAULT_SIDE_ORDER = ["innerwall", "sidewall1", "sidewall2", "tread", "bead"]
# Current active inspection sides.
# Innerwall and bead are not enabled yet.
DEFAULT_SIDE_ORDER = ["sidewall1", "sidewall2", "tread"]

def _build_camera_serial_map_from_env() -> Dict[str, str]:
    """
    Returns:
        {
            "sidewall1": "serial_244802149",
            "sidewall2": "serial_244802163",
            "innerwall": "serial_251102086",
            "tread": "serial_251401655",
            "bead": "serial_251300826",
        }

    Source:
        .env → HARDWARE_TRIGGER.py → get_side_to_camera_map()
    """
    side_to_camera = get_side_to_camera_map()

    return {
        side_name: f"serial_{serial}"
        for side_name, serial in side_to_camera.items()
    }


CAMERA_SERIAL_MAP = _build_camera_serial_map_from_env()


# =========================================================
# CACHE
# =========================================================
_RUNTIME_CACHE: Dict[str, Dict[str, Any]] = {}
_WARMED_RUNTIME_KEYS: set = set()

def clear_runtime_cache():
    """
    Call this after calibration is completed or when SKU/artifacts are changed.
    """
    _RUNTIME_CACHE.clear()
    _WARMED_RUNTIME_KEYS.clear()
    print("[MAIN] runtime cache cleared")

# =========================================================
# FOLDER STRUCTURE HELPERS
# =========================================================
def _get_today_capture_root(media_root: str, sku_name: str = "UNKNOWN_SKU") -> str:
    """
    Final capture input structure:
        media/Capture_Input/<SKU>/<date>/Cycle_N
    """
    date_str = datetime.now().strftime("%d-%m-%Y")

    today_dir = os.path.join(
        media_root,
        "Capture_Input",
        sku_name,
        date_str,
    )

    os.makedirs(today_dir, exist_ok=True)
    return today_dir


def _next_cycle_number(today_capture_root: str) -> int:
    existing = [
        d for d in os.listdir(today_capture_root)
        if os.path.isdir(os.path.join(today_capture_root, d))
        and d.startswith("Cycle_")
    ]

    nums = []
    for name in existing:
        try:
            nums.append(int(name.split("_", 1)[1]))
        except ValueError:
            pass

    return max(nums) + 1 if nums else 1


def build_cycle_capture_dir(
    media_root: str,
    sku_name: str = "UNKNOWN_SKU",
) -> tuple[str, str]:
    """
    Creates:
        media/Capture_Input/<SKU>/<date>/Cycle_N
    """
    today_root = _get_today_capture_root(
        media_root=media_root,
        sku_name=sku_name,
    )

    n = _next_cycle_number(today_root)

    cycle_id = f"Cycle_{n}"
    cycle_dir = os.path.join(today_root, cycle_id)

    os.makedirs(cycle_dir, exist_ok=True)

    print(f"[CAPTURE] New cycle folder: {cycle_dir}")

    return cycle_dir, cycle_id


def _camera_serial_folder(cycle_capture_dir: str, serial: str) -> str:
    folder = os.path.join(cycle_capture_dir, serial)
    os.makedirs(folder, exist_ok=True)
    return folder


# =========================================================
# CAMERA CAPTURE HELPERS
# =========================================================
def _save_image(img_np: np.ndarray, out_path: str) -> None:
    ext = os.path.splitext(out_path)[1].lower()

    if img_np.dtype == np.uint16:
        if ext not in (".png", ".tiff", ".tif"):
            img_np = (img_np / 256).astype(np.uint8)

    if ext in (".jpg", ".jpeg"):
        cv2.imwrite(out_path, img_np, [cv2.IMWRITE_JPEG_QUALITY, CAPTURE_JPEG_QUALITY])
    else:
        cv2.imwrite(out_path, img_np)


def capture_and_save_images(
    multi_camera_manager,
    cycle_capture_dir: str,
    sides_to_run: List[str],
) -> Dict[str, str]:
    """
    Capture and save images from the live camera manager.

    Supports both:
      old manager output: {serial_number: image}
      new shared-camera manager output: {side_name: image}
    """
    print("[CAPTURE] Starting camera capture for all sides ...")

    raw_images: Dict[str, np.ndarray] = multi_camera_manager.capture_all()

    if hasattr(multi_camera_manager, "camera_to_side"):
        serial_to_side = {
            str(serial): side
            for serial, side in multi_camera_manager.camera_to_side.items()
        }
    else:
        serial_to_side = get_camera_to_side_map()

    if hasattr(multi_camera_manager, "side_to_camera"):
        side_to_camera = {
            str(side): str(serial)
            for side, serial in multi_camera_manager.side_to_camera.items()
        }
    else:
        side_to_camera = get_side_to_camera_map()

    known_sides = set(side_to_camera.keys()) | set(sides_to_run)
    image_map: Dict[str, str] = {}

    for image_key, img in raw_images.items():
        image_key = str(image_key)

        if img is None:
            side = image_key if image_key in known_sides else serial_to_side.get(image_key, image_key)
            print(f"[CAPTURE][WARN] No image for key={image_key} side={side}")
            continue

        # New manager returns side names directly.
        if image_key in known_sides:
            side = image_key
            serial_for_folder = side_to_camera.get(side, image_key)
        else:
            # Backward compatibility: old manager returned serial numbers.
            side = serial_to_side.get(image_key)
            serial_for_folder = image_key

        if not side:
            print(f"[CAPTURE][WARN] Camera key {image_key} not mapped in .env.")
            side = f"camera_{image_key}"

        # Keep existing application folder style: serial_<serial>.
        # For shared camera, innerwall and bead may both have serial_250500042;
        # use side-specific filename so one does not overwrite the other.
        folder_name = f"serial_{serial_for_folder}"
        cam_folder = _camera_serial_folder(cycle_capture_dir, folder_name)

        file_name = f"{side}{CAPTURE_IMAGE_FORMAT}"
        out_path = os.path.join(cam_folder, file_name)

        _save_image(img, out_path)

        print(f"[CAPTURE] Saved key={image_key} side={side} -> {out_path}")

        if side in sides_to_run:
            image_map[side] = out_path
        else:
            print(f"[CAPTURE][WARN] Side {side} saved but not selected in sides_to_run.")

    return image_map


# =========================================================
# COMMON HELPERS
# =========================================================
def _json_safe(obj: Any) -> Any:
    try:
        import numpy as np

        if isinstance(obj, dict):
            return {str(k): _json_safe(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_json_safe(x) for x in obj]
        if isinstance(obj, tuple):
            return [_json_safe(x) for x in obj]
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
    except Exception:
        pass

    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(x) for x in obj]
    if isinstance(obj, tuple):
        return [_json_safe(x) for x in obj]

    try:
        json.dumps(obj)
        return obj
    except Exception:
        return str(obj)


def _required_file(path: Optional[str], label: str) -> str:
    if not path:
        raise ValueError(f"{label} is required")

    if not os.path.isfile(path):
        raise FileNotFoundError(f"{label} not found: {path}")

    return path


def _normalize_device(device: str) -> str:
    if device == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA not available, using CPU")
        return "cpu"

    return device


def _resolve_sides(sides_to_run: Optional[List[str]]) -> List[str]:
    if not sides_to_run:
        return DEFAULT_SIDE_ORDER.copy()

    if sides_to_run == ["all"]:
        return DEFAULT_SIDE_ORDER.copy()

    return sides_to_run





def _get_sku_calibration_dir(media_root: str, sku_name: str) -> str:
    """
    Returns:
        media/AI_Calibration_Files/<SKU>
    """
    sku_dir = os.path.join(
        media_root,
        CALIBRATION_ROOT_DIR_NAME,
        sku_name,
    )

    if not os.path.isdir(sku_dir):
        raise FileNotFoundError(
            f"SKU calibration folder not found: {sku_dir}"
        )

    return sku_dir


def _get_side_calibration_dir(
    media_root: str,
    sku_name: str,
    side_name: str,
) -> str:
    """
    Returns:
        media/AI_Calibration_Files/<SKU>/calibration_<side>
    """
    if side_name not in SIDE_CALIBRATION_DIRS:
        raise ValueError(f"Unknown side for calibration folder: {side_name}")

    side_dir = os.path.join(
        _get_sku_calibration_dir(media_root, sku_name),
        SIDE_CALIBRATION_DIRS[side_name],
    )

    if not os.path.isdir(side_dir):
        raise FileNotFoundError(
            f"Side calibration folder not found for {side_name}: {side_dir}"
        )

    return side_dir


def _get_side_artifacts_dir(
    media_root: str,
    sku_name: str,
    side_name: str,
) -> str:
    """
    Returns:
        media/AI_Calibration_Files/<SKU>/calibration_<side>/artifacts
    """
    artifacts_dir = os.path.join(
        _get_side_calibration_dir(media_root, sku_name, side_name),
        "artifacts",
    )

    if not os.path.isdir(artifacts_dir):
        raise FileNotFoundError(
            f"Artifacts folder not found for {side_name}: {artifacts_dir}"
        )

    return artifacts_dir

def _read_side_offset_ratio(side_artifacts_dir: str, side_name: str) -> float:
    """
    Reads non-R side offset ratio from artifacts folder.

    Expected file:
        media/AI_Calibration_Files/<SKU>/calibration_<side>/artifacts/offset_ratio.json

    Example:
        {
            "offset_ratio": 0.35
        }
    """

    if side_name in ["sidewall1", "sidewall2"]:
        return 0.0

    offset_json = os.path.join(side_artifacts_dir, "offset_ratio.json")

    if not os.path.isfile(offset_json):
        print(
            f"[MAIN][WARN] offset_ratio.json missing for {side_name}. "
            "Using 0.0. Crop may be wrong."
        )
        return 0.0

    with open(offset_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    return float(data.get("offset_ratio", 0.0))
def _get_sku_artifacts_dir(
    media_root: str,
    sku_name: str,
    side_name: Optional[str] = None,
):
    """
    Backward-compatible helper.

    If side_name is passed:
        returns that side's artifacts folder.

    If side_name is not passed:
        returns the SKU calibration root. This avoids breaking older print/debug code.
    """
    if side_name is not None:
        return _get_side_artifacts_dir(media_root, sku_name, side_name)

    return _get_sku_calibration_dir(media_root, sku_name)


def _side_artifacts_ref_image(
    media_root: str,
    sku_name: str,
    side_name: str,
) -> str:
    ref_img = os.path.join(
        _get_side_artifacts_dir(media_root, sku_name, side_name),
        "alignment_reference_polarized.png",
    )

    if not os.path.isfile(ref_img):
        raise FileNotFoundError(
            f"Reference image not found for {side_name}: {ref_img}"
        )

    return ref_img


def _shared_artifacts_ref_image(media_root: str, sku_name: str) -> str:
    """
    Backward-compatible helper.
    Uses sidewall1 reference by default if older code calls this.
    """
    return _side_artifacts_ref_image(
        media_root=media_root,
        sku_name=sku_name,
        side_name="sidewall1",
    )

# =========================================================
# SKU-SPECIFIC MODEL HELPERS
# Structure:
#   media/AI_Calibration_Files/<SKU>/models/r_detector.pt
#   media/AI_Calibration_Files/<SKU>/models/sidewall1_checkpoint.pth
#   media/AI_Calibration_Files/<SKU>/models/sidewall2_checkpoint.pth
#   media/AI_Calibration_Files/<SKU>/models/tread_checkpoint.pth
# =========================================================

SKU_MODEL_DIR_NAME = "models"

SIDE_CHECKPOINT_FILE_CANDIDATES = {
    "sidewall1": [
        "sidewall1_checkpoint.pth",
        "checkpoint_sidewall1.pth",
        "sw1_checkpoint.pth",
        "checkpoint_sw1.pth",
    ],
    "sidewall2": [
        "sidewall2_checkpoint.pth",
        "checkpoint_sidewall2.pth",
        "sw2_checkpoint.pth",
        "checkpoint_sw2.pth",
    ],
    "tread": [
        "tread_checkpoint.pth",
        "checkpoint_tread.pth",
    ],
    "innerwall": [
        "innerwall_checkpoint.pth",
        "checkpoint_innerwall.pth",
    ],
    "bead": [
        "bead_checkpoint.pth",
        "checkpoint_bead.pth",
    ],
}

R_DETECTOR_FILE_CANDIDATES = [
    "r_detector.pt",
    "R_detector.pt",
    "best_R.pt",
    "best_R_CEAT_DEMO.pt",
    "R_Detection.pt",
]


def _get_sku_models_dir(media_root: str, sku_name: str) -> str:
    return os.path.join(
        _get_sku_calibration_dir(media_root, sku_name),
        SKU_MODEL_DIR_NAME,
    )


def _find_first_existing(paths: List[str]) -> Optional[str]:
    for path in paths:
        if path and os.path.isfile(path):
            return path
    return None


def _resolve_sku_r_detector_path(
    media_root: str,
    sku_name: str,
    fallback_r_detector_path: Optional[str],
) -> str:
    sku_models_dir = _get_sku_models_dir(media_root, sku_name)

    candidates = [
        os.path.join(sku_models_dir, name)
        for name in R_DETECTOR_FILE_CANDIDATES
    ]

    found = _find_first_existing(candidates)

    if found:
        print(f"[MAIN][MODEL] Using SKU R-detector: {found}")
        return found

    if fallback_r_detector_path and os.path.isfile(fallback_r_detector_path):
        print(f"[MAIN][MODEL] Using fallback/global R-detector: {fallback_r_detector_path}")
        return fallback_r_detector_path

    raise FileNotFoundError(
        "R detector model not found.\n"
        f"Checked SKU model folder: {sku_models_dir}\n"
        f"Fallback path: {fallback_r_detector_path}"
    )


def _resolve_side_checkpoint_path(
    media_root: str,
    sku_name: str,
    side_name: str,
    fallback_checkpoint_path: Optional[str],
) -> str:
    sku_models_dir = _get_sku_models_dir(media_root, sku_name)

    candidates = [
        os.path.join(sku_models_dir, file_name)
        for file_name in SIDE_CHECKPOINT_FILE_CANDIDATES.get(side_name, [])
    ]

    found = _find_first_existing(candidates)

    if found:
        print(f"[MAIN][MODEL] Using SKU checkpoint for {side_name}: {found}")
        return found

    if fallback_checkpoint_path and os.path.isfile(fallback_checkpoint_path):
        print(f"[MAIN][MODEL] Using fallback/global checkpoint for {side_name}: {fallback_checkpoint_path}")
        return fallback_checkpoint_path

    raise FileNotFoundError(
        f"VIT checkpoint not found for side: {side_name}\n"
        f"Checked SKU model folder: {sku_models_dir}\n"
        f"Fallback path: {fallback_checkpoint_path}"
    )

# =========================================================
# IMAGE MAP HELPERS
# =========================================================
def build_image_map_from_capture_dir(
    cycle_capture_dir: str,
    sides_to_run: List[str],
) -> Dict[str, str]:
    image_map: Dict[str, str] = {}
    valid_exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

    camera_serial_map = _build_camera_serial_map_from_env()

    for side_name in sides_to_run:
        serial_folder_name = camera_serial_map.get(side_name)

        if not serial_folder_name:
            raise ValueError(f"No camera serial mapping for side: {side_name}")

        folder_path = os.path.join(cycle_capture_dir, serial_folder_name)

        if not os.path.isdir(folder_path):
            raise FileNotFoundError(
                f"Camera folder not found for {side_name}: {folder_path}"
            )

        files = [
            os.path.join(folder_path, f)
            for f in os.listdir(folder_path)
            if f.lower().endswith(valid_exts)
        ]

        if not files:
            raise FileNotFoundError(f"No image found for {side_name} in {folder_path}")

        files.sort(key=os.path.getmtime, reverse=True)
        image_map[side_name] = files[0]

    return image_map


def get_latest_image_from_folder(folder_path: str) -> Optional[str]:
    if not os.path.isdir(folder_path):
        return None

    valid_exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

    files = [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.lower().endswith(valid_exts)
    ]

    if not files:
        return None

    files.sort(key=os.path.getmtime, reverse=True)
    return files[0]


def build_image_map_from_capture_root(
    capture_root: str,
    sides_to_run: List[str],
) -> Dict[str, str]:
    image_map: Dict[str, str] = {}

    camera_serial_map = _build_camera_serial_map_from_env()

    for side_name in sides_to_run:
        serial_folder = camera_serial_map.get(side_name)

        if not serial_folder:
            raise ValueError(f"No camera serial mapping for side: {side_name}")

        folder_path = os.path.join(capture_root, serial_folder)

        latest_image = get_latest_image_from_folder(folder_path)

        if not latest_image:
            raise FileNotFoundError(f"No image found for {side_name} in {folder_path}")

        image_map[side_name] = latest_image

    return image_map


# =========================================================
# RUNTIME MANAGEMENT
# =========================================================
def combine_tire_decision(side_results: Dict[str, Dict[str, Any]]) -> str:
    labels = [x.get("final_label", "") for x in side_results.values()]

    if any(x == "DEFECT" for x in labels):
        return "DEFECT"

    if any(x == "SUSPECT" for x in labels):
        return "SUSPECT"

    if any(x in ["INVALID", "FAILED"] for x in labels):
        return "INVALID"

    return "OK"


def build_seg_models(
    device: str,
    seg_model_a_path: str,
    seg_model_b_path: str,
) -> Dict[str, Any]:
    seg_model_a_path = _required_file(seg_model_a_path, "seg_model_a_path")
    seg_model_b_path = _required_file(seg_model_b_path, "seg_model_b_path")

    same_model = os.path.abspath(seg_model_a_path) == os.path.abspath(seg_model_b_path)

    try:
        seg_a = load_yolo_seg(seg_model_a_path, device=device, imgsz=SEG_IMGSZ)
    except TypeError:
        seg_a = load_yolo_seg(seg_model_a_path, device=device)

    if same_model:
        seg_b = seg_a
        print("[MAIN] one shared classification model loaded for seg_a and seg_b")
    else:
        try:
            seg_b = load_yolo_seg(seg_model_b_path, device=device, imgsz=SEG_IMGSZ)
        except TypeError:
            seg_b = load_yolo_seg(seg_model_b_path, device=device)

        print("[MAIN] segmentation models loaded separately")

    return {
        "seg_a": seg_a,
        "seg_b": seg_b,
    }

def _get_runtime_cache_key(
    sku_name,
    device,
    seg_model_a_path,
    seg_model_b_path,
    vit_checkpoint_path,
    r_detector_path,
    media_root,
    sides_to_run,
) -> str:
    return "||".join(
        [
            sku_name,
            device,
            seg_model_a_path,
            seg_model_b_path,
            vit_checkpoint_path,
            r_detector_path,
            media_root,
            ",".join(sides_to_run),
        ]
    )


def _apply_tyre_name_to_runtimes(runtimes: Dict[str, Any], tyre_name: str) -> None:
    for runtime in runtimes.values():
        if isinstance(runtime, dict):
            runtime["tyre_name"] = tyre_name


def warmup_all_runtimes(runtimes: Dict[str, Any], sides_to_run: List[str]) -> None:
    for side_name in sides_to_run:
        runtime = runtimes.get(side_name)

        if runtime is None:
            continue

        module = SIDE_MODULES[side_name]

        if hasattr(module, "warmup_runtime"):
            print(f"[MAIN] warming up {side_name}")
            module.warmup_runtime(runtime)


def _build_same_model_side_configs(
    media_root,
    sku_name,
    vit_checkpoint_path,
    r_detector_path,
    tyre_name=DEFAULT_TYRE_NAME,
    use_yolo_seg=DEFAULT_USE_YOLO_SEG,
    sides_to_run=None,
) -> Dict[str, Dict[str, Any]]:
    selected_sides = _resolve_sides(sides_to_run)

    resolved_r_detector_path = _resolve_sku_r_detector_path(
        media_root=media_root,
        sku_name=sku_name,
        fallback_r_detector_path=r_detector_path,
    )

    side_configs: Dict[str, Dict[str, Any]] = {}

    for side_name in selected_sides:
        side_artifacts_dir = _get_side_artifacts_dir(
            media_root=media_root,
            sku_name=sku_name,
            side_name=side_name,
        )

        offset_ratio = _read_side_offset_ratio(
            side_artifacts_dir=side_artifacts_dir,
            side_name=side_name,
        )

        side_checkpoint_path = _resolve_side_checkpoint_path(
            media_root=media_root,
            sku_name=sku_name,
            side_name=side_name,
            fallback_checkpoint_path=vit_checkpoint_path,
        )

        side_ref_image = os.path.join(
            side_artifacts_dir,
            "alignment_reference_polarized.png",
        )

        # Sidewall needs this file.
        # New AI-team tread flow does NOT need this file.
        if side_name in ["sidewall1", "sidewall2"]:
            if not os.path.isfile(side_ref_image):
                raise FileNotFoundError(
                    f"Missing alignment reference for {side_name}: {side_ref_image}"
                )
        else:
            if not os.path.isfile(side_ref_image):
                side_ref_image = None

        side_configs[side_name] = dict(
            checkpoint_path=side_checkpoint_path,
            output_dir=media_root,
            yolo_r_path=resolved_r_detector_path,
            use_yolo_seg=use_yolo_seg,
            tyre_name=tyre_name,

            calibration_artifact_dir=side_artifacts_dir,
            x_align_artifacts_dir=side_artifacts_dir,
            offset_ratio=offset_ratio,

            ref_image_path=side_ref_image,
        )

        print(
            f"[MAIN][CONFIG] {side_name} | "
            f"artifacts={side_artifacts_dir} | "
            f"checkpoint={side_checkpoint_path} | "
            f"offset_ratio={offset_ratio}"
        )

    return side_configs


def _build_optional_trt_vit(checkpoint_path: str, device: str, side_name: str):
    trt_vit = None
    use_trt_vit = False

    if not ENABLE_TRT_VIT:
        return trt_vit, use_trt_vit

    if not checkpoint_path:
        return trt_vit, use_trt_vit

    if not str(checkpoint_path).lower().endswith(".engine"):
        print(f"[MAIN] PyTorch ViT checkpoint for {side_name}: {checkpoint_path}")
        return trt_vit, use_trt_vit

    if TRTViTFeatureExtractor is None:
        print(
            f"[MAIN][WARN] TRT ViT engine found for {side_name}, "
            "but TRTViTFeatureExtractor import failed. Falling back."
        )
        return trt_vit, use_trt_vit

    trt_vit = TRTViTFeatureExtractor(checkpoint_path, device=device)
    use_trt_vit = True

    print(f"[MAIN] TRT ViT engine loaded for {side_name}: {checkpoint_path}")

    return trt_vit, use_trt_vit


def _load_runtime_with_optional_trt(
    module,
    side_name: str,
    side_cfg: Dict[str, Any],
    device: str,
    seg_models: Dict[str, Any],
    shared_r_detector,
):
    checkpoint_path = side_cfg["checkpoint_path"]
    trt_vit, use_trt_vit = _build_optional_trt_vit(checkpoint_path, device, side_name)

    kwargs = dict(
        device=device,
        seg_models=seg_models,
        r_detector_override=shared_r_detector,
        use_yolo_seg_override=side_cfg["use_yolo_seg"],
        checkpoint_path_override=checkpoint_path,
        output_dir_override=side_cfg["output_dir"],
        ref_image_path_override=side_cfg.get("ref_image_path"),
        yolo_r_path_override=side_cfg.get("yolo_r_path"),
        tyre_name_override=side_cfg.get("tyre_name"),
        load_artifacts=True,
        calibration_artifact_dir_override=side_cfg.get("calibration_artifact_dir"),
    )

    if use_trt_vit:
        kwargs["trt_vit"] = trt_vit
        kwargs["use_trt_vit"] = True

    try:
        return module.load_runtime(**kwargs)

    except TypeError as e:
        removed_any = False

        if "trt_vit" in kwargs or "use_trt_vit" in kwargs:
            kwargs.pop("trt_vit", None)
            kwargs.pop("use_trt_vit", None)
            removed_any = True

        if "calibration_artifact_dir_override" in kwargs:
            kwargs.pop("calibration_artifact_dir_override", None)
            removed_any = True

        if removed_any:
            print(
                f"[MAIN][WARN] {side_name} load_runtime rejected one optional kwarg. "
                f"Retrying without optional kwargs. error={e}"
            )
            return module.load_runtime(**kwargs)

        raise


def build_all_runtimes(
    sku_name,
    media_root,
    seg_model_a_path,
    seg_model_b_path,
    vit_checkpoint_path,
    r_detector_path,
    device="cuda",
    capture_root="",
    tyre_name=DEFAULT_TYRE_NAME,
    side_configs=None,
    sides_to_run=None,
) -> Dict[str, Any]:
    device = _normalize_device(device)
    sides_to_run = _resolve_sides(sides_to_run)

    resolved_r_detector_path = _resolve_sku_r_detector_path(
        media_root=media_root,
        sku_name=sku_name,
        fallback_r_detector_path=r_detector_path,
    )

    _required_file(resolved_r_detector_path, "r_detector_path")

    if side_configs is None:
        side_configs = _build_same_model_side_configs(
            media_root=media_root,
            sku_name=sku_name,
            vit_checkpoint_path=vit_checkpoint_path,
            r_detector_path=resolved_r_detector_path,
            tyre_name=tyre_name,
            sides_to_run=sides_to_run,
        )

    r_detector_path = resolved_r_detector_path

    model_signature = "||".join(
        [
            side_configs[s]["checkpoint_path"]
            for s in sides_to_run
            if s in side_configs
        ]
    )

    cache_key = _get_runtime_cache_key(
        sku_name,
        device,
        seg_model_a_path,
        seg_model_b_path,
        model_signature,
        r_detector_path,
        media_root,
        sides_to_run,
    )

    if cache_key in _RUNTIME_CACHE:
        print(f"[MAIN] using cached runtimes for {cache_key}")
        return _RUNTIME_CACHE[cache_key]

    seg_models = build_seg_models(device, seg_model_a_path, seg_model_b_path)

    shared_r_detector = None

    if USE_SHARED_R_DETECTOR:
        if build_r_detector is None:
            raise RuntimeError("build_r_detector import failed")

        shared_r_detector = build_r_detector(r_detector_path, conf=0.3, device=device)
        print("[MAIN] shared R-detector loaded once")

    runtimes: Dict[str, Any] = {}

    for side_name in sides_to_run:
        module = SIDE_MODULES[side_name]
        side_cfg = side_configs[side_name]

        print(f"[MAIN] loading runtime for {side_name}")

        runtimes[side_name] = _load_runtime_with_optional_trt(
            module=module,
            side_name=side_name,
            side_cfg=side_cfg,
            device=device,
            seg_models=seg_models,
            shared_r_detector=shared_r_detector,
        )
        runtimes[side_name]["side_config"] = side_cfg
        runtimes[side_name]["offset_ratio"] = float(side_cfg.get("offset_ratio", 0.0))

    _RUNTIME_CACHE[cache_key] = runtimes

    return runtimes


def _maybe_warmup_runtimes(
    runtimes,
    sku_name,
    device,
    capture_root,
    seg_model_a_path,
    seg_model_b_path,
    vit_checkpoint_path,
    r_detector_path,
    tyre_name,
    media_root,
    sides_to_run,
) -> None:
    cache_key = _get_runtime_cache_key(
        sku_name,
        device,
        seg_model_a_path,
        seg_model_b_path,
        vit_checkpoint_path,
        r_detector_path,
        media_root,
        sides_to_run,
    )

    if not ENABLE_WARMUP:
        return

    if cache_key in _WARMED_RUNTIME_KEYS:
        print(f"[MAIN] runtimes already warmed for {cache_key}")
        return

    warmup_all_runtimes(runtimes, sides_to_run)

    _WARMED_RUNTIME_KEYS.add(cache_key)


def preload_live_runtimes(
    capture_root,
    media_root,
    sku_name="SKU_001",
    device=DEVICE,
    seg_model_a_path=None,
    seg_model_b_path=None,
    vit_checkpoint_path=None,
    r_detector_path=None,
    tyre_name=DEFAULT_TYRE_NAME,
    side_configs=None,
    sides_to_run=None,
) -> bool:
    sides_to_run = _resolve_sides(sides_to_run)

    capture_root = os.path.abspath(capture_root)
    media_root = os.path.abspath(media_root)
    device = _normalize_device(device)

    seg_model_a_path = _required_file(seg_model_a_path, "seg_model_a_path")
    seg_model_b_path = _required_file(seg_model_b_path, "seg_model_b_path")
    vit_checkpoint_path = _required_file(vit_checkpoint_path, "vit_checkpoint_path")
    r_detector_path = _required_file(r_detector_path, "r_detector_path")

    runtimes = build_all_runtimes(
        sku_name=sku_name,
        media_root=media_root,
        seg_model_a_path=seg_model_a_path,
        seg_model_b_path=seg_model_b_path,
        vit_checkpoint_path=vit_checkpoint_path,
        r_detector_path=r_detector_path,
        device=device,
        capture_root=capture_root,
        tyre_name=tyre_name,
        side_configs=side_configs,
        sides_to_run=sides_to_run,
    )

    _maybe_warmup_runtimes(
        runtimes=runtimes,
        sku_name=sku_name,
        device=device,
        capture_root=capture_root,
        seg_model_a_path=seg_model_a_path,
        seg_model_b_path=seg_model_b_path,
        vit_checkpoint_path=vit_checkpoint_path,
        r_detector_path=r_detector_path,
        tyre_name=tyre_name,
        media_root=media_root,
        sides_to_run=sides_to_run,
    )

    return True


# =========================================================
# PER-SIDE INFERENCE
# =========================================================
def _sem_context(sem):
    return sem if sem is not None else nullcontext()


def _module_supports_stage_pipeline(module) -> bool:
    required = [
        "read_and_polarize",
        "align_crop_from_preprocessed",
        "to_gray",
        "patchify_array_indexed",
        "get_patch_embeddings_from_arrays",
        "process_precomputed_embeddings",
        "run_yolo_on_vit_defect_patches",
    ]

    return all(hasattr(module, name) for name in required)


def _run_one_side_infer_legacy(
    side_name,
    image_path,
    runtime,
    cycle_dir,
    r_gpu_sem,
    vit_gpu_sem,
    yolo_gpu_sem,
):
    side_dir = os.path.join(cycle_dir, side_name)
    os.makedirs(side_dir, exist_ok=True)

    module = SIDE_MODULES[side_name]

    t0 = time.perf_counter()

    result = module.infer_single_image(
        raw_path=image_path,
        runtime=runtime,
        output_root=side_dir,
        r_gpu_sem=r_gpu_sem,
        vit_gpu_sem=vit_gpu_sem,
        yolo_gpu_sem=yolo_gpu_sem,
    )

    result["side_latency_sec"] = round(time.perf_counter() - t0, 3)
    result["pipeline_mode"] = "legacy_infer_single_image"

    return side_name, result


def run_side_pipeline(
    side_name,
    image_path,
    runtime,
    cycle_dir,
    r_gpu_sem,
    vit_gpu_sem,
    yolo_gpu_sem,
    sidewall_r_anchor=None,
):
    module = SIDE_MODULES[side_name]

    side_t0 = time.perf_counter()
    name = os.path.splitext(os.path.basename(image_path))[0]

    result: Dict[str, Any] = {
        "side_name": side_name,
        "image": name,
        "final_label": "FAILED",
        "pipeline_mode": "stage_pipeline",
        "vit_valid_patches": 0,
        "vit_defect_patches": 0,
        "yolo_detections": 0,
        "align_time": 0.0,
        "vit_time": 0.0,
        "yolo_time": 0.0,
    }

    side_root_dir = os.path.join(cycle_dir, side_name)
    side_crop_dir = os.path.join(side_root_dir, "crop")
    side_final_dir = os.path.join(side_root_dir, "final")

    os.makedirs(side_crop_dir, exist_ok=True)
    os.makedirs(side_final_dir, exist_ok=True)

    crop_path = os.path.join(side_crop_dir, "crop.png")
    vit_df = pd.DataFrame()

    # =========================================================
    # STAGE 1: READ / POLARIZE / ALIGN / CROP
    # =========================================================
    t_align = time.perf_counter()

    try:
        _, pre_bgr = module.read_and_polarize(image_path)

        with _sem_context(r_gpu_sem):
            # -------------------------------------------------
            # Non-R sides: innerwall / tread / bead
            # -------------------------------------------------
            if side_name in ["innerwall", "tread", "bead"]:
                if sidewall_r_anchor is None:
                    raise RuntimeError(
                        f"{side_name} requires sidewall_r_anchor from current sidewall1/sidewall2."
                    )

                x_align_debug_path = os.path.join(
                    side_final_dir,
                    f"{name}_{side_name}_xalign_debug.png",
                )

                x_align_artifacts_dir = runtime.get("x_align_artifacts_dir") or runtime.get("calibration_artifact_dir")

                if not x_align_artifacts_dir:
                    raise RuntimeError(
                        f"{side_name} missing runtime['calibration_artifact_dir'] / x_align_artifacts_dir"
                    )

                # New AI-team tread uses clean x-align signature.
                if side_name == "tread":
                    crop_bgr, non_r_meta = module.align_crop_from_preprocessed(
                        pre_bgr=pre_bgr,
                        sidewall_r_anchor=sidewall_r_anchor,
                        offset_ratio=float(runtime.get("offset_ratio", 0.0)),
                        x_align_artifacts_dir=x_align_artifacts_dir,
                        create_x_reference_if_missing=False,
                        x_align_debug_path=x_align_debug_path,
                        return_meta=True,
                    )

                # Keep old-compatible signature for innerwall/bead when you enable them later.
                else:
                    crop_bgr, non_r_meta = module.align_crop_from_preprocessed(
                        pre_bgr=pre_bgr,
                        ref_pre_bgr=None,
                        r_detector=None,
                        save_template_path=None,
                        ref_info=None,
                        use_incoming_r_detection=False,
                        sidewall_r_anchor=sidewall_r_anchor,
                        offset_ratio=float(runtime.get("offset_ratio", 0.0)),
                        x_align_artifacts_dir=x_align_artifacts_dir,
                        create_x_reference_if_missing=False,
                        x_align_debug_path=x_align_debug_path,
                        return_meta=True,
                    )

                result["alignment_meta"] = non_r_meta

            # -------------------------------------------------
            # R sides: sidewall1 / sidewall2
            # -------------------------------------------------
            else:
                crop_anchor_ref_path = runtime.get("crop_anchor_ref_path")

                crop_anchor_debug_path = os.path.join(
                    side_crop_dir,
                    f"{side_name}_crop_anchor_debug.png",
                )

                aligned_template_path = os.path.join(
                    side_crop_dir,
                    "aligned_template.png",
                )

                crop_bgr, sidewall_meta = module.align_crop_from_preprocessed(
                    pre_bgr=pre_bgr,
                    ref_pre_bgr=runtime["ref_pre_bgr"],
                    r_detector=runtime.get("r_detector"),
                    save_template_path=aligned_template_path,
                    reference_r=runtime.get("reference_r"),
                    crop_anchor_ref_path=crop_anchor_ref_path,
                    crop_anchor_debug_path=crop_anchor_debug_path,
                    debug_name=f"INFER_{side_name}_{name}",
                    return_meta=True,
                )

                result["alignment_meta"] = sidewall_meta

                if side_name in ["sidewall1", "sidewall2"]:
                    if extract_sidewall_r_anchor_from_meta is None:
                        raise RuntimeError(
                            "extract_sidewall_r_anchor_from_meta import failed"
                        )

                    result["sidewall_r_anchor"] = extract_sidewall_r_anchor_from_meta(
                        sidewall_meta
                    )

        crop_gray = module.to_gray(crop_bgr)
        cv2.imwrite(crop_path, crop_gray)

        result["crop_path"] = crop_path
        result["align_time"] = round(time.perf_counter() - t_align, 3)

        print(f"[PIPELINE] {side_name} alignment done | {result['align_time']:.3f}s")

    except Exception as e:
        result["align_time"] = round(time.perf_counter() - t_align, 3)
        result["error"] = f"alignment failed: {e}"
        result["side_latency_sec"] = round(time.perf_counter() - side_t0, 3)

        print(f"[PIPELINE][ERROR] {side_name} alignment failed | error={e}")
        return side_name, result

    # =========================================================
    # STAGE 2: VIT / TEMPLATE MATCHING
    # =========================================================
    t_vit = time.perf_counter()

    try:
        with _sem_context(vit_gpu_sem):
            patch_records = module.patchify_array_indexed(
                crop_gray,
                patch_h=module.BIG_PATCH_H,
                patch_w=module.BIG_PATCH_W,
                step_h=module.BIG_STEP_H,
                step_w=module.BIG_STEP_W,
                cover_edges=module.COVER_EDGES,
            )

            embeddings, valid_records = module.get_patch_embeddings_from_arrays(
                model=runtime["model"],
                patch_records=patch_records,
                device=runtime.get("device", DEVICE),
                tfm=runtime.get("patch_transform", module._build_transform()),
            )

            if len(valid_records) > 0:
                defect_cache_dir = os.path.join(side_crop_dir, "__yolo_cache")

                vit_df, stitched_path = module.process_precomputed_embeddings(
                    embeddings=embeddings,
                    valid_records=valid_records,
                    runtime=runtime,
                    save_dir=side_final_dir,
                    defect_cache_dir=defect_cache_dir,
                )

                result["template_stitched_path"] = stitched_path
            else:
                vit_df = pd.DataFrame()
                result["template_stitched_path"] = None

        if vit_df is not None and not vit_df.empty:
            valid_df = vit_df[
                vit_df["classification"].isin(["GOOD", "DEFECT"])
            ].copy()

            result["vit_valid_patches"] = int(len(valid_df))
            result["vit_defect_patches"] = (
                int((valid_df["classification"] == "DEFECT").sum())
                if len(valid_df)
                else 0
            )
        else:
            result["vit_valid_patches"] = 0
            result["vit_defect_patches"] = 0

        result["vit_time"] = round(time.perf_counter() - t_vit, 3)

        print(
            f"[PIPELINE] {side_name} ViT/template done | "
            f"{result['vit_time']:.3f}s | "
            f"vit_defects={result['vit_defect_patches']}"
        )

    except Exception as e:
        result["vit_time"] = round(time.perf_counter() - t_vit, 3)
        result["error"] = f"ViT/template failed: {e}"
        result["side_latency_sec"] = round(time.perf_counter() - side_t0, 3)

        print(f"[PIPELINE][ERROR] {side_name} ViT/template failed | error={e}")
        return side_name, result

    # =========================================================
    # STAGE 3: YOLO ONLY ON VIT DEFECT PATCHES
    # =========================================================
    t_yolo = time.perf_counter()

    try:
        if result["vit_defect_patches"] > 0 and runtime.get(
            "use_yolo_seg",
            DEFAULT_USE_YOLO_SEG,
        ):
            with _sem_context(yolo_gpu_sem):
                yolo_df, final_stitched_path, dim_summary = (
                    module.run_yolo_on_vit_defect_patches(
                        vit_df=vit_df,
                        save_dir=side_final_dir,
                        seg_models=runtime["seg_models"],
                        conf_threshold=module.SEG_CONF_THRESHOLD,
                        crop_path=crop_path,
                        tyre_name=runtime.get("tyre_name"),
                    )
                )

            result["yolo_detections"] = int(len(yolo_df)) if yolo_df is not None else 0
            result["final_stitched_path"] = final_stitched_path
            result["dim_summary"] = dim_summary

            if (
                result["yolo_detections"] > 0
                and final_stitched_path
                and os.path.isfile(final_stitched_path)
            ):
                result["final_label"] = "DEFECT"
            else:
                result["final_label"] = "SUSPECT"

        else:
            result["yolo_detections"] = 0

            if result["vit_defect_patches"] == 0:
                result["final_label"] = "OK"
            else:
                result["final_label"] = "SUSPECT"

        result["yolo_time"] = round(time.perf_counter() - t_yolo, 3)

        print(
            f"[PIPELINE] {side_name} YOLO done | "
            f"{result['yolo_time']:.3f}s | "
            f"detections={result['yolo_detections']} | "
            f"label={result['final_label']}"
        )

    except Exception as e:
        result["yolo_time"] = round(time.perf_counter() - t_yolo, 3)
        result["error"] = f"YOLO failed: {e}"

        if result["vit_defect_patches"] > 0:
            result["final_label"] = "SUSPECT"
        else:
            result["final_label"] = "OK"

        print(f"[PIPELINE][ERROR] {side_name} YOLO failed | error={e}")

    if CLEAN_YOLO_CACHE:
        try:
            defect_cache_dir = os.path.join(side_crop_dir, "__yolo_cache")

            if os.path.isdir(defect_cache_dir):
                shutil.rmtree(defect_cache_dir, ignore_errors=True)

        except Exception:
            pass

    result["side_latency_sec"] = round(time.perf_counter() - side_t0, 3)

    return side_name, result

def _run_one_side_infer(
    side_name,
    image_path,
    runtime,
    cycle_dir,
    r_gpu_sem,
    vit_gpu_sem,
    yolo_gpu_sem,
    sidewall_r_anchor=None,
):
    module = SIDE_MODULES[side_name]

    if ENABLE_STAGE_PIPELINE and _module_supports_stage_pipeline(module):
        try:
            side_name_out, result = run_side_pipeline(
                side_name,
                image_path,
                runtime,
                cycle_dir,
                r_gpu_sem,
                vit_gpu_sem,
                yolo_gpu_sem,
                sidewall_r_anchor=sidewall_r_anchor,
            )

            if result.get("final_label") != "FAILED":
                return side_name_out, result

            if PIPELINE_FALLBACK_TO_INFER_SINGLE:
                print(
                    f"[PIPELINE][WARN] {side_name} stage pipeline returned FAILED. "
                    "Falling back to infer_single_image."
                )

                return _run_one_side_infer_legacy(
                    side_name,
                    image_path,
                    runtime,
                    cycle_dir,
                    r_gpu_sem,
                    vit_gpu_sem,
                    yolo_gpu_sem,
                )

            return side_name_out, result

        except Exception as e:
            if not PIPELINE_FALLBACK_TO_INFER_SINGLE:
                raise

            print(
                f"[PIPELINE][WARN] {side_name} stage pipeline crashed. "
                f"Falling back. error={e}"
            )

            return _run_one_side_infer_legacy(
                side_name,
                image_path,
                runtime,
                cycle_dir,
                r_gpu_sem,
                vit_gpu_sem,
                yolo_gpu_sem,
            )

    print(f"[MAIN] {side_name} does not expose stage-pipeline helpers. Using infer_single_image.")

    return _run_one_side_infer_legacy(
        side_name,
        image_path,
        runtime,
        cycle_dir,
        r_gpu_sem,
        vit_gpu_sem,
        yolo_gpu_sem,
    )


# =========================================================
# RUN FULL INFERENCE CYCLE
# =========================================================
def run_cycle(
    image_map,
    runtimes,
    output_root,
    cycle_id,
    sides_to_run=None,
    r_gpu_sem=None,
    vit_gpu_sem=None,
    yolo_gpu_sem=None,
    sku_name=None,
    tyre_name=None,
):
    sides_to_run = _resolve_sides(sides_to_run)
    logger.info(
        "AI inspection cycle started",
        extra={
            "event_code": "AI_CYCLE_STARTED",
            "cycle_id": cycle_id,
            "tyre_id": tyre_name or "-",
            "sku_name": sku_name or "-",
            "status": "STARTED",
            "details": {"sides": list(sides_to_run)},
        },
    )

    cycle_dir = os.path.join(output_root, cycle_id)
    os.makedirs(cycle_dir, exist_ok=True)

    cycle_t0 = time.perf_counter()
    side_results: Dict[str, Dict[str, Any]] = {}

    for side_name in sides_to_run:
        if side_name not in image_map:
            raise ValueError(f"Missing image for side: {side_name}")

    non_r_sides = {"innerwall", "tread", "bead"}
    anchor_candidates = ["sidewall1", "sidewall2"]

    need_sidewall_anchor = any(s in non_r_sides for s in sides_to_run)

    anchor_side = None
    sidewall_r_anchor = None

    if need_sidewall_anchor:
        for candidate in anchor_candidates:
            if candidate in sides_to_run:
                anchor_side = candidate
                break

        if anchor_side is None:
            raise RuntimeError(
                "Live inference for innerwall/tread/bead requires sidewall1 or sidewall2 "
                "in sides_to_run, because current sidewall R anchor is required."
            )

        print(f"[PIPELINE] Running anchor side first: {anchor_side}")

        _, anchor_result = _run_one_side_infer(
            side_name=anchor_side,
            image_path=image_map[anchor_side],
            runtime=runtimes[anchor_side],
            cycle_dir=cycle_dir,
            r_gpu_sem=r_gpu_sem,
            vit_gpu_sem=vit_gpu_sem,
            yolo_gpu_sem=yolo_gpu_sem,
            sidewall_r_anchor=None,
        )

        side_results[anchor_side] = anchor_result

        if anchor_result.get("final_label") == "FAILED":
            raise RuntimeError(
                f"Anchor side failed: {anchor_side} | {anchor_result.get('error')}"
            )

        sidewall_r_anchor = anchor_result.get("sidewall_r_anchor")

        if sidewall_r_anchor is None:
            raise RuntimeError(
                f"Anchor side did not return sidewall_r_anchor: {anchor_side}"
            )

        print(f"[PIPELINE] Current sidewall R anchor: {sidewall_r_anchor}")

    remaining_sides = [s for s in sides_to_run if s != anchor_side]

    if PARALLEL_INFER and remaining_sides:
        workers = min(INFER_SIDE_WORKERS, len(remaining_sides))

        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = {}

            for side_name in remaining_sides:
                anchor_for_side = (
                    sidewall_r_anchor
                    if side_name in non_r_sides
                    else None
                )

                fut = ex.submit(
                    _run_one_side_infer,
                    side_name,
                    image_map[side_name],
                    runtimes[side_name],
                    cycle_dir,
                    r_gpu_sem,
                    vit_gpu_sem,
                    yolo_gpu_sem,
                    anchor_for_side,
                )

                futures[fut] = side_name

            for fut in as_completed(futures):
                side_name = futures[fut]

                try:
                    _, result = fut.result()
                    side_results[side_name] = result

                    set_live_progress(
                        phase="INFERENCE",
                        active_zone=side_name,
                        images_captured=len(side_results),
                        total_images=len(sides_to_run),
                        message=f"Inference completed for {side_name}",
                    )

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                except Exception as e:
                    side_results[side_name] = {
                        "image": os.path.basename(image_map[side_name]),
                        "final_label": "FAILED",
                        "error": str(e),
                    }

                    set_live_progress(
                        phase="INFERENCE",
                        active_zone=side_name,
                        images_captured=len(side_results),
                        total_images=len(sides_to_run),
                        message=f"Inference failed for {side_name}",
                    )

                    logger.exception(
                        "Zone inference failed",
                        extra={
                            "event_code": "AI_ZONE_FAILED",
                            "error_code": "AI-001",
                            "cycle_id": cycle_id,
                            "tyre_id": tyre_name or "-",
                            "sku_name": sku_name or "-",
                            "zone": side_name,
                            "status": "FAILED",
                        },
                    )

    else:
        for side_name in remaining_sides:
            anchor_for_side = (
                sidewall_r_anchor
                if side_name in non_r_sides
                else None
            )

            _, result = _run_one_side_infer(
                side_name,
                image_map[side_name],
                runtimes[side_name],
                cycle_dir,
                r_gpu_sem,
                vit_gpu_sem,
                yolo_gpu_sem,
                sidewall_r_anchor=anchor_for_side,
            )

            side_results[side_name] = result

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    cycle_latency_sec = round(time.perf_counter() - cycle_t0, 3)
    final_tire_label = combine_tire_decision(side_results)

    total_align = sum(float(r.get("align_time", 0) or 0) for r in side_results.values())
    total_vit = sum(float(r.get("vit_time", 0) or 0) for r in side_results.values())
    total_yolo = sum(float(r.get("yolo_time", 0) or 0) for r in side_results.values())

    seq_total = total_align + total_vit + total_yolo
    speedup = round(seq_total / cycle_latency_sec, 2) if cycle_latency_sec > 0 else 0

    rows = []

    for side_name in sides_to_run:
        row = {
            "cycle_id": cycle_id,
            "sku_name": sku_name,
            "tyre_name": tyre_name,
            "side": side_name,
            "input_image": image_map[side_name],
            "cycle_latency_sec": cycle_latency_sec,
        }

        row.update(_json_safe(side_results.get(side_name, {})))
        rows.append(row)

    if SAVE_CYCLE_SUMMARY:
        pd.DataFrame(rows).to_csv(
            os.path.join(cycle_dir, "side_results.csv"),
            index=False,
        )

        with open(os.path.join(cycle_dir, "tire_summary.json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "cycle_id": cycle_id,
                    "sku_name": sku_name,
                    "tyre_name": tyre_name,
                    "final_tire_label": final_tire_label,
                    "cycle_latency_sec": cycle_latency_sec,
                    "image_map": image_map,
                    "side_results": _json_safe(side_results),
                },
                f,
                indent=2,
                ensure_ascii=False,
            )

    logger.info(
        "AI inspection cycle completed",
        extra={
            "event_code": "AI_CYCLE_COMPLETED",
            "cycle_id": cycle_id,
            "tyre_id": tyre_name or "-",
            "sku_name": sku_name or "-",
            "status": final_tire_label,
            "duration_ms": round(cycle_latency_sec * 1000.0, 3),
            "details": {
                "stage_sum_sec": round(seq_total, 3),
                "estimated_speedup": speedup,
                "side_labels": {
                    side: result.get("final_label", "UNKNOWN")
                    for side, result in side_results.items()
                },
            },
        },
    )

    return {
        "cycle_id": cycle_id,
        "sku_name": sku_name,
        "tyre_name": tyre_name,
        "final_label": final_tire_label,
        "final_tire_label": final_tire_label,
        "cycle_latency_sec": cycle_latency_sec,
        "stage_sum_sec": round(seq_total, 3),
        "estimated_speedup": speedup,
        "side_results": side_results,
        "output_dir": cycle_dir,
        "image_map": image_map,
        "cycle_dir": cycle_dir,
    }