import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import json
import time
import threading
import shutil
from contextlib import nullcontext
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

try:
    from src.models.Pipeline.R_inner_mapping_alignment import build_r_detector
except Exception:
    try:
        from src.models.Pipeline.R_inner_mapping_alignment import build_r_detector
    except Exception:
        build_r_detector = None
from src.COMMON.db import save_cycle_metadata

try:
    from src.models.Pipeline.vit_trt_inference import TRTViTFeatureExtractor
except Exception:
    try:
        from vit_trt_inference import TRTViTFeatureExtractor
    except Exception:
        TRTViTFeatureExtractor = None

# Same optimization idea as AI-team file: keep Python/OpenCV/Torch CPU thread usage controlled.
try:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
except Exception:
    pass
try:
    cv2.setNumThreads(0)
except Exception:
    pass

# =========================================================
# GLOBALS
# =========================================================
DEVICE             = "cuda"
PARALLEL_INFER = True
PARALLEL_CALIB = False
ENABLE_WARMUP      = True
INFER_SIDE_WORKERS = 5
CALIB_SIDE_WORKERS = 1

R_ALIGN_GPU_CONCURRENCY = 5
VIT_GPU_CONCURRENCY = 5
YOLO_GPU_CONCURRENCY = 5

USE_SHARED_R_DETECTOR = True
SAVE_CYCLE_SUMMARY    = True
DEFAULT_TYRE_NAME     = "195_65_R15"
DEFAULT_USE_YOLO_SEG  = True
SEG_IMGSZ = 224

ENABLE_STAGE_PIPELINE = True
PIPELINE_FALLBACK_TO_INFER_SINGLE = True
ENABLE_TRT_VIT = True
CLEAN_YOLO_CACHE = True

# ── Camera capture globals ─────────────────────────────────────────────────────
# Set to True when running with real cameras (deployment mode).
# Set to False to skip capture and use images already on disk (demo / local mode).
CAMERA_CAPTURE_ENABLED = False

# Image format used when saving captured frames from cameras.
CAPTURE_IMAGE_FORMAT = ".png"  

# JPEG quality (0-100). Only used when CAPTURE_IMAGE_FORMAT = ".jpg"
CAPTURE_JPEG_QUALITY = 95

# =========================================================
# SIDE MODULES / ORDER / CAMERA MAP
# =========================================================
SIDE_MODULES = {
    "innerwall": innerwall,
    "sidewall1": sidewall1,
    "sidewall2": sidewall2,
    "tread":     tread,
    "bead":      bead,
}

DEFAULT_SIDE_ORDER = ["innerwall", "sidewall1", "sidewall2", "tread", "bead"]

# Maps each tyre side → camera serial number folder name.
# Must match the serial numbers used in MultiCameraManager / LineScanCamera.
CAMERA_SERIAL_MAP = {
    "bead":      "serial_254701283",
    "innerwall": "serial_254701292",
    "sidewall1": "serial_254901428",
    "tread":     "serial_254901430",
    "sidewall2": "serial_254901432",
}

# =========================================================
# CACHE
# =========================================================
_RUNTIME_CACHE:        Dict[str, Dict[str, Any]] = {}
_WARMED_RUNTIME_KEYS:  set = set()


# =========================================================
# ── FOLDER STRUCTURE HELPERS ──────────────────────────────
# =========================================================

def _get_today_capture_root(media_root: str) -> str:
    """
    Returns  <media_root>/capture/<DD-MM-YYYY>/
    Creates the directory if it does not exist.
    """
    date_str  = datetime.now().strftime("%d-%m-%Y")
    today_dir = os.path.join(media_root, "capture", date_str)
    os.makedirs(today_dir, exist_ok=True)
    return today_dir


def _next_cycle_number(today_capture_root: str) -> int:
    """
    Scans <today_capture_root> for existing Cycle_N folders and
    returns the next integer so cycles increment automatically:
        Cycle_1, Cycle_2, Cycle_3 …
    """
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
    return (max(nums) + 1) if nums else 1


def build_cycle_capture_dir(media_root: str) -> tuple[str, str]:
    """
    Creates and returns the capture directory for the current cycle:
        <media_root>/capture/<DD-MM-YYYY>/Cycle_N/

    Returns
    -------
    (cycle_capture_dir, cycle_id)
        cycle_capture_dir : absolute path to the new Cycle_N folder
        cycle_id          : string like "Cycle_3"  (used for output naming too)
    """
    today_root = _get_today_capture_root(media_root)
    n          = _next_cycle_number(today_root)
    cycle_id   = f"Cycle_{n}"
    cycle_dir  = os.path.join(today_root, cycle_id)
    os.makedirs(cycle_dir, exist_ok=True)
    print(f"[CAPTURE] New cycle folder: {cycle_dir}")
    return cycle_dir, cycle_id


def _camera_serial_folder(cycle_capture_dir: str, serial: str) -> str:
    """
    Creates and returns  <cycle_capture_dir>/<serial>/
    e.g.  …/Cycle_3/serial_254901428/
    """
    folder = os.path.join(cycle_capture_dir, serial)
    os.makedirs(folder, exist_ok=True)
    return folder


# =========================================================
# ── CAMERA CAPTURE & SAVE ─────────────────────────────────
# =========================================================

def _save_image(img_np: np.ndarray, out_path: str) -> None:
    """
    Saves a numpy image (uint8 or uint16) to *out_path*.
    Handles Mono16 → uint16 PNG and uint8 → JPEG / PNG.
    """
    ext = os.path.splitext(out_path)[1].lower()

    if img_np.dtype == np.uint16:
        # OpenCV imwrite supports uint16 only for PNG / TIFF
        if ext not in (".png", ".tiff", ".tif"):
            # Convert to 8-bit for JPEG etc.
            img_np = (img_np / 256).astype(np.uint8)

    if ext in (".jpg", ".jpeg"):
        cv2.imwrite(out_path, img_np, [cv2.IMWRITE_JPEG_QUALITY, CAPTURE_JPEG_QUALITY])
    else:
        cv2.imwrite(out_path, img_np)


def capture_and_save_images(
    multi_camera_manager,           # MultiCameraManager instance
    cycle_capture_dir: str,
    sides_to_run: List[str],
) -> Dict[str, str]:
    """
    Triggers parallel capture on all cameras, then saves each image into:
        <cycle_capture_dir>/<serial_XXXXXXXXX>/image<EXT>

    Returns
    -------
    image_map : { side_name → absolute image path }
    """
    print("[CAPTURE] Starting camera capture for all sides …")
    # returns { serial_str: np.ndarray }
    raw_images: Dict[str, np.ndarray] = multi_camera_manager.capture_all()

    # Build reverse map:  serial_str → side_name  (for labelling)
    serial_to_side = {v.replace("serial_", ""): k for k, v in CAMERA_SERIAL_MAP.items()}

    image_map: Dict[str, str] = {}

    for serial_str, img in raw_images.items():
        if img is None:
            side = serial_to_side.get(str(serial_str), str(serial_str))
            print(f"[CAPTURE][WARN] No image for serial {serial_str} (side={side}), skipping.")
            continue

        # Folder:  Cycle_N/serial_254901428/
        folder_name = f"serial_{serial_str}"
        cam_folder  = _camera_serial_folder(cycle_capture_dir, folder_name)

        # File name: image.png  (one image per cycle per camera)
        file_name = f"image{CAPTURE_IMAGE_FORMAT}"
        out_path  = os.path.join(cam_folder, file_name)

        _save_image(img, out_path)
        print(f"[CAPTURE] Saved {serial_str} → {out_path}")

        # Map side_name → saved path
        side = serial_to_side.get(str(serial_str))
        if side and side in sides_to_run:
            image_map[side] = out_path
        else:
            print(f"[CAPTURE][WARN] Serial {serial_str} not in CAMERA_SERIAL_MAP or not in sides_to_run.")

    return image_map


# =========================================================
# HELPERS (unchanged)
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
    sku_dir = os.path.join(media_root, "calibration", sku_name)
    if not os.path.isdir(sku_dir):
        raise FileNotFoundError(f"SKU calibration folder not found: {sku_dir}")
    return sku_dir


def _get_sku_artifacts_dir(media_root: str, sku_name: str) -> str:
    artifacts_dir = os.path.join(_get_sku_calibration_dir(media_root, sku_name), "artifacts")
    if not os.path.isdir(artifacts_dir):
        raise FileNotFoundError(f"SKU artifacts folder not found: {artifacts_dir}")
    return artifacts_dir


def _shared_artifacts_ref_image(media_root: str, sku_name: str) -> str:
    ref_img = os.path.join(
        _get_sku_artifacts_dir(media_root, sku_name),
        "alignment_reference_polarized.png"
    )
    if not os.path.isfile(ref_img):
        raise FileNotFoundError(f"Reference image not found: {ref_img}")
    return ref_img


# =========================================================
# ── IMAGE MAP BUILDERS ────────────────────────────────────
# =========================================================

def build_image_map_from_capture_dir(
    cycle_capture_dir: str,
    sides_to_run: List[str],
) -> Dict[str, str]:
    """
    Reads already-saved images from a cycle capture directory:
        <cycle_capture_dir>/serial_XXXXXXXXX/image<EXT>

    Use this when CAMERA_CAPTURE_ENABLED = False (demo / local mode)
    and the images were placed manually or by a previous capture run.
    """
    image_map: Dict[str, str] = {}
    valid_exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

    for side_name in sides_to_run:
        serial_folder_name = CAMERA_SERIAL_MAP.get(side_name)
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


def build_image_map_from_capture_root(capture_root: str, sides_to_run: List[str]) -> Dict[str, str]:
    """Legacy helper kept for backward compatibility (demo folder mode)."""
    image_map: Dict[str, str] = {}
    for side_name in sides_to_run:
        serial_folder = CAMERA_SERIAL_MAP[side_name]
        folder_path   = os.path.join(capture_root, serial_folder)
        latest_image  = get_latest_image_from_folder(folder_path)
        if not latest_image:
            raise FileNotFoundError(f"No image found for {side_name} in {folder_path}")
        image_map[side_name] = latest_image
    return image_map


# =========================================================
# RUNTIME MANAGEMENT (unchanged)
# =========================================================

def combine_tire_decision(side_results: Dict[str, Dict[str, Any]]) -> str:
    labels = [x.get("final_label", "") for x in side_results.values()]
    if any(x == "DEFECT"  for x in labels): return "DEFECT"
    if any(x == "SUSPECT" for x in labels): return "SUSPECT"
    if any(x in ["INVALID", "FAILED"] for x in labels): return "INVALID"
    return "OK"


def build_seg_models(device: str, seg_model_a_path: str, seg_model_b_path: str) -> Dict[str, Any]:
    _required_file(seg_model_a_path, "seg_model_a_path")
    _required_file(seg_model_b_path, "seg_model_b_path")

    try:
        seg_a = load_yolo_seg(seg_model_a_path, device=device, imgsz=SEG_IMGSZ)
    except TypeError:
        seg_a = load_yolo_seg(seg_model_a_path, device=device)

    try:
        seg_b = load_yolo_seg(seg_model_b_path, device=device, imgsz=SEG_IMGSZ)
    except TypeError:
        seg_b = load_yolo_seg(seg_model_b_path, device=device)

    print("[MAIN] segmentation models loaded once")
    return {"seg_a": seg_a, "seg_b": seg_b}


def _get_runtime_cache_key(
    sku_name, device, seg_model_a_path, seg_model_b_path,
    vit_checkpoint_path, r_detector_path, media_root, sides_to_run,
) -> str:
    return "||".join([
        sku_name, device, seg_model_a_path, seg_model_b_path,
        vit_checkpoint_path, r_detector_path, media_root,
        ",".join(sides_to_run),
    ])

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
) -> Dict[str, Dict[str, Any]]:
    shared_ref_image = _shared_artifacts_ref_image(media_root, sku_name)

    common = dict(
        checkpoint_path=vit_checkpoint_path,
        output_dir=media_root,   # IMPORTANT: keep media_root, not sku_calibration_dir
        yolo_r_path=r_detector_path,
        use_yolo_seg=use_yolo_seg,
        tyre_name=tyre_name,
    )

    return {
        "innerwall": {**common, "ref_image_path": shared_ref_image},
        "sidewall1": {**common, "ref_image_path": shared_ref_image},
        "sidewall2": {**common, "ref_image_path": shared_ref_image},
        "tread":     {**common, "ref_image_path": shared_ref_image},
        "bead":      {**common, "ref_image_path": shared_ref_image},
    }



def _build_optional_trt_vit(checkpoint_path: str, device: str, side_name: str):
    """
    If checkpoint_path is a TensorRT .engine file, load the TRT ViT extractor.
    If it is a .pth/.pt file or TRT class is unavailable, return PyTorch fallback.
    """
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
            "but TRTViTFeatureExtractor import failed. Falling back to pipeline load_runtime."
        )
        return trt_vit, use_trt_vit

    trt_vit = TRTViTFeatureExtractor(checkpoint_path, device=device)
    use_trt_vit = True
    print(f"[MAIN] TRT ViT engine loaded for {side_name}: {checkpoint_path}")
    return trt_vit, use_trt_vit


def _load_runtime_with_optional_trt(module, side_name: str, side_cfg: Dict[str, Any],
                                    device: str, seg_models: Dict[str, Any], shared_r_detector):
    """
    Calls each side module load_runtime.
    Supports new AI-team kwargs: trt_vit and use_trt_vit.
    Falls back cleanly if your current side modules do not yet accept those kwargs.
    """
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
    )

    if use_trt_vit:
        kwargs["trt_vit"] = trt_vit
        kwargs["use_trt_vit"] = True

    try:
        return module.load_runtime(**kwargs)
    except TypeError as e:
        # Old side modules may not accept trt_vit/use_trt_vit yet.
        if "trt_vit" in kwargs or "use_trt_vit" in kwargs:
            print(f"[MAIN][WARN] {side_name} load_runtime does not accept TRT kwargs yet. Falling back. error={e}")
            kwargs.pop("trt_vit", None)
            kwargs.pop("use_trt_vit", None)
            return module.load_runtime(**kwargs)
        raise

def build_all_runtimes(
    sku_name, media_root, seg_model_a_path, seg_model_b_path,
    vit_checkpoint_path, r_detector_path,
    device="cuda", capture_root="", tyre_name=DEFAULT_TYRE_NAME,
    side_configs=None, sides_to_run=None,
) -> Dict[str, Any]:
    device       = _normalize_device(device)
    sides_to_run = _resolve_sides(sides_to_run)

    _required_file(vit_checkpoint_path, "vit_checkpoint_path")
    _required_file(r_detector_path,     "r_detector_path")

    if side_configs is None:
        side_configs = _build_same_model_side_configs(
            media_root=media_root,
            sku_name=sku_name,
            vit_checkpoint_path=vit_checkpoint_path,
            r_detector_path=r_detector_path,
            tyre_name=tyre_name,
        )

    cache_key = _get_runtime_cache_key(
        sku_name, device, seg_model_a_path, seg_model_b_path,
        vit_checkpoint_path, r_detector_path, media_root, sides_to_run,
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
        module   = SIDE_MODULES[side_name]
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

    _RUNTIME_CACHE[cache_key] = runtimes
    return runtimes


def _maybe_warmup_runtimes(
    runtimes, sku_name, device, capture_root,
    seg_model_a_path, seg_model_b_path, vit_checkpoint_path,
    r_detector_path, tyre_name, media_root, sides_to_run,
) -> None:
    cache_key = _get_runtime_cache_key(
        sku_name, device, seg_model_a_path, seg_model_b_path,
        vit_checkpoint_path, r_detector_path, media_root, sides_to_run,
    )
    if not ENABLE_WARMUP:
        return
    if cache_key in _WARMED_RUNTIME_KEYS:
        print(f"[MAIN] runtimes already warmed for {cache_key}")
        return
    warmup_all_runtimes(runtimes, sides_to_run)
    _WARMED_RUNTIME_KEYS.add(cache_key)


def preload_live_runtimes(
    capture_root, media_root, sku_name="SKU_001", device=DEVICE,
    seg_model_a_path=None, seg_model_b_path=None,
    vit_checkpoint_path=None, r_detector_path=None,
    tyre_name=DEFAULT_TYRE_NAME, side_configs=None, sides_to_run=None,
) -> bool:
    sides_to_run = _resolve_sides(sides_to_run)
    capture_root = os.path.abspath(capture_root)
    media_root   = os.path.abspath(media_root)
    device       = _normalize_device(device)

    seg_model_a_path    = _required_file(seg_model_a_path,    "seg_model_a_path")
    seg_model_b_path    = _required_file(seg_model_b_path,    "seg_model_b_path")
    vit_checkpoint_path = _required_file(vit_checkpoint_path, "vit_checkpoint_path")
    r_detector_path     = _required_file(r_detector_path,     "r_detector_path")

    runtimes = build_all_runtimes(
        sku_name=sku_name, media_root=media_root,
        seg_model_a_path=seg_model_a_path, seg_model_b_path=seg_model_b_path,
        vit_checkpoint_path=vit_checkpoint_path, r_detector_path=r_detector_path,
        device=device, capture_root=capture_root, tyre_name=tyre_name,
        side_configs=side_configs, sides_to_run=sides_to_run,
    )
    _maybe_warmup_runtimes(
        runtimes=runtimes, sku_name=sku_name, device=device,
        capture_root=capture_root, seg_model_a_path=seg_model_a_path,
        seg_model_b_path=seg_model_b_path, vit_checkpoint_path=vit_checkpoint_path,
        r_detector_path=r_detector_path, tyre_name=tyre_name,
        media_root=media_root, sides_to_run=sides_to_run,
    )
    return True


# =========================================================
# PER-SIDE INFER / CYCLE (unchanged)
# =========================================================

def _sem_context(sem):
    return sem if sem is not None else nullcontext()


def _module_supports_stage_pipeline(module) -> bool:
    """
    The AI-team file manually splits each side into:
    read/polarize -> align/crop -> patch embeddings -> ViT classification -> YOLO.
    Only use it if the side module exposes all required helpers.
    """
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


def _run_one_side_infer_legacy(side_name, image_path, runtime, cycle_dir,
                               r_gpu_sem, vit_gpu_sem, yolo_gpu_sem):
    """Your original behavior: delegate the full side processing to infer_single_image."""
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


def run_side_pipeline(side_name, image_path, runtime, cycle_dir,
                      r_gpu_sem, vit_gpu_sem, yolo_gpu_sem):
    """
    AI-team functionality merged into your GUI main cycle.

    For each side, it runs stages separately:
      1) read + polarize + R/reference alignment
      2) save crop image under <cycle>/<side>/crop/crop.png
      3) patchify + ViT/template matching, saving template output under final
      4) run YOLO only on ViT defect patches
      5) clean temporary YOLO cache

    Output is still inside your existing media/output/Cycle_N/<side>/ structure.
    """
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

    # ============================================================
    # STAGE 1: Read / polarize / align / crop
    # ============================================================
    t_align = time.perf_counter()
    try:
        _, pre_bgr = module.read_and_polarize(image_path)

        with _sem_context(r_gpu_sem):
            if side_name in ["innerwall", "tread", "bead"]:
                crop_bgr = module.align_crop_from_preprocessed(
                    pre_bgr=pre_bgr,
                    ref_pre_bgr=runtime["ref_pre_bgr"],
                    r_detector=runtime.get("r_detector"),
                    save_template_path=None,
                    ref_info=runtime.get("reference_band_info"),
                    use_incoming_r_detection=False,
                )
            else:
                crop_bgr = module.align_crop_from_preprocessed(
                    pre_bgr=pre_bgr,
                    ref_pre_bgr=runtime["ref_pre_bgr"],
                    r_detector=runtime.get("r_detector"),
                    save_template_path=None,
                    reference_r=runtime.get("reference_r"),
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

    # ============================================================
    # STAGE 2: ViT / template matching
    # ============================================================
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
            valid_df = vit_df[vit_df["classification"].isin(["GOOD", "DEFECT"])].copy()
            result["vit_valid_patches"] = int(len(valid_df))
            result["vit_defect_patches"] = int((valid_df["classification"] == "DEFECT").sum()) if len(valid_df) else 0
        else:
            result["vit_valid_patches"] = 0
            result["vit_defect_patches"] = 0

        result["vit_time"] = round(time.perf_counter() - t_vit, 3)
        print(
            f"[PIPELINE] {side_name} ViT/template done | "
            f"{result['vit_time']:.3f}s | vit_defects={result['vit_defect_patches']}"
        )

    except Exception as e:
        result["vit_time"] = round(time.perf_counter() - t_vit, 3)
        result["error"] = f"ViT/template failed: {e}"
        result["side_latency_sec"] = round(time.perf_counter() - side_t0, 3)
        print(f"[PIPELINE][ERROR] {side_name} ViT/template failed | error={e}")
        return side_name, result

    # ============================================================
    # STAGE 3: YOLO only on ViT defect patches
    # ============================================================
    t_yolo = time.perf_counter()
    try:
        if result["vit_defect_patches"] > 0 and runtime.get("use_yolo_seg", DEFAULT_USE_YOLO_SEG):
            with _sem_context(yolo_gpu_sem):
                yolo_df, final_stitched_path, dim_summary = module.run_yolo_on_vit_defect_patches(
                    vit_df=vit_df,
                    save_dir=side_final_dir,
                    seg_models=runtime["seg_models"],
                    conf_threshold=module.SEG_CONF_THRESHOLD,
                    crop_path=crop_path,
                    tyre_name=runtime.get("tyre_name"),
                )

            result["yolo_detections"] = int(len(yolo_df)) if yolo_df is not None else 0
            result["final_stitched_path"] = final_stitched_path
            result["dim_summary"] = dim_summary

            if result["yolo_detections"] > 0 and final_stitched_path and os.path.isfile(final_stitched_path):
                result["final_label"] = "DEFECT"
            else:
                result["final_label"] = "SUSPECT"
        else:
            result["yolo_detections"] = 0
            result["final_label"] = "OK" if result["vit_defect_patches"] == 0 else "SUSPECT"

        result["yolo_time"] = round(time.perf_counter() - t_yolo, 3)
        print(
            f"[PIPELINE] {side_name} YOLO done | "
            f"{result['yolo_time']:.3f}s | detections={result['yolo_detections']} | label={result['final_label']}"
        )

    except Exception as e:
        result["yolo_time"] = round(time.perf_counter() - t_yolo, 3)
        result["error"] = f"YOLO failed: {e}"
        result["final_label"] = "SUSPECT" if result["vit_defect_patches"] > 0 else "OK"
        print(f"[PIPELINE][ERROR] {side_name} YOLO failed | error={e}")

    # Remove temporary patch cache from crop folder.
    if CLEAN_YOLO_CACHE:
        try:
            defect_cache_dir = os.path.join(side_crop_dir, "__yolo_cache")
            if os.path.isdir(defect_cache_dir):
                shutil.rmtree(defect_cache_dir, ignore_errors=True)
        except Exception:
            pass

    result["side_latency_sec"] = round(time.perf_counter() - side_t0, 3)
    return side_name, result


def _run_one_side_infer(side_name, image_path, runtime, cycle_dir,
                        r_gpu_sem, vit_gpu_sem, yolo_gpu_sem):
    """
    Uses AI-team stage pipeline when supported.
    Falls back to your original infer_single_image flow when unsupported.
    """
    module = SIDE_MODULES[side_name]

    if ENABLE_STAGE_PIPELINE and _module_supports_stage_pipeline(module):
        try:
            side_name_out, result = run_side_pipeline(
                side_name, image_path, runtime, cycle_dir,
                r_gpu_sem, vit_gpu_sem, yolo_gpu_sem,
            )
            if result.get("final_label") != "FAILED":
                return side_name_out, result

            if PIPELINE_FALLBACK_TO_INFER_SINGLE:
                print(f"[PIPELINE][WARN] {side_name} stage pipeline returned FAILED. Falling back to infer_single_image.")
                return _run_one_side_infer_legacy(
                    side_name, image_path, runtime, cycle_dir,
                    r_gpu_sem, vit_gpu_sem, yolo_gpu_sem,
                )
            return side_name_out, result

        except Exception as e:
            if not PIPELINE_FALLBACK_TO_INFER_SINGLE:
                raise
            print(f"[PIPELINE][WARN] {side_name} stage pipeline crashed. Falling back. error={e}")
            return _run_one_side_infer_legacy(
                side_name, image_path, runtime, cycle_dir,
                r_gpu_sem, vit_gpu_sem, yolo_gpu_sem,
            )

    print(f"[MAIN] {side_name} does not expose stage-pipeline helpers. Using infer_single_image.")
    return _run_one_side_infer_legacy(
        side_name, image_path, runtime, cycle_dir,
        r_gpu_sem, vit_gpu_sem, yolo_gpu_sem,
    )

def run_cycle(
    image_map, runtimes, output_root, cycle_id,
    sides_to_run=None, r_gpu_sem=None, vit_gpu_sem=None, yolo_gpu_sem=None,
    sku_name=None, tyre_name=None,
):
    sides_to_run = _resolve_sides(sides_to_run)
    cycle_dir    = os.path.join(output_root, cycle_id)
    os.makedirs(cycle_dir, exist_ok=True)

    cycle_t0     = time.perf_counter()
    side_results: Dict[str, Dict[str, Any]] = {}

    for side_name in sides_to_run:
        if side_name not in image_map:
            raise ValueError(f"Missing image for side: {side_name}")

    if PARALLEL_INFER:
        workers = min(INFER_SIDE_WORKERS, len(sides_to_run))
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = {
                ex.submit(
                    _run_one_side_infer, side_name, image_map[side_name],
                    runtimes[side_name], cycle_dir,
                    r_gpu_sem, vit_gpu_sem, yolo_gpu_sem,
                ): side_name
                for side_name in sides_to_run
            }
            for fut in as_completed(futures):
                side_name = futures[fut]
                try:
                    _, result = fut.result()
                    side_results[side_name] = result
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception as e:
                    side_results[side_name] = {
                        "image": os.path.basename(image_map[side_name]),
                        "final_label": "FAILED",
                        "error": str(e),
                    }
                    print(f"[MAIN][ERROR] inference failed | side={side_name} | error={e}")
    else:
        for side_name in sides_to_run:
            _, result = _run_one_side_infer(
                side_name, image_map[side_name], runtimes[side_name], cycle_dir,
                r_gpu_sem, vit_gpu_sem, yolo_gpu_sem,
            )
            side_results[side_name] = result
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    cycle_latency_sec  = round(time.perf_counter() - cycle_t0, 3)
    final_tire_label   = combine_tire_decision(side_results)

    total_align = sum(float(r.get("align_time", 0) or 0) for r in side_results.values())
    total_vit   = sum(float(r.get("vit_time", 0) or 0) for r in side_results.values())
    total_yolo  = sum(float(r.get("yolo_time", 0) or 0) for r in side_results.values())
    if any("align_time" in r or "vit_time" in r or "yolo_time" in r for r in side_results.values()):
        seq_total = total_align + total_vit + total_yolo
        speedup = round(seq_total / cycle_latency_sec, 2) if cycle_latency_sec > 0 else 0
        # print(
        #     f"[PIPELINE SUMMARY] align_sum={total_align:.3f}s | vit_sum={total_vit:.3f}s | "
        #     f"yolo_sum={total_yolo:.3f}s | sequential_est={seq_total:.3f}s | "
        #     f"actual_cycle={cycle_latency_sec:.3f}s | speedup_est={speedup}x"
        # )

    rows = []
    for side_name in sides_to_run:
        row = {
            "cycle_id":          cycle_id,
            "sku_name":          sku_name,
            "tyre_name":         tyre_name,
            "side":              side_name,
            "input_image":       image_map[side_name],
            "cycle_latency_sec": cycle_latency_sec,
        }
        row.update(_json_safe(side_results.get(side_name, {})))
        rows.append(row)

    if SAVE_CYCLE_SUMMARY:
        pd.DataFrame(rows).to_csv(
            os.path.join(cycle_dir, "side_results.csv"), index=False
        )
        with open(os.path.join(cycle_dir, "tire_summary.json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "cycle_id":           cycle_id,
                    "sku_name":           sku_name,
                    "tyre_name":          tyre_name,
                    "final_tire_label":   final_tire_label,
                    "cycle_latency_sec":  cycle_latency_sec,
                    "image_map":          image_map,
                    "side_results":       _json_safe(side_results),
                },
                f, indent=2, ensure_ascii=False,
            )

    print(f"[MAIN FINAL] {cycle_id} -> {final_tire_label} | cycle_time={cycle_latency_sec:.3f}s")
    return {
        "cycle_id":           cycle_id,
        "sku_name":           sku_name,
        "tyre_name":          tyre_name,
        "final_label":        final_tire_label,
        "cycle_latency_sec":  cycle_latency_sec,
        "side_results":       side_results,
        "image_map":          image_map,
        "cycle_dir":          cycle_dir,
    }


# =========================================================
# ── MAIN ENTRY POINT ──────────────────────────────────────
# =========================================================

def run_capture_folder_cycle(
    media_root: str,
    sku_name:   str  = "SKU_001",
    cycle_id:   Optional[str] = None,
    device:     str  = DEVICE,
    seg_model_a_path:    Optional[str] = None,
    seg_model_b_path:    Optional[str] = None,
    vit_checkpoint_path: Optional[str] = None,
    r_detector_path:     Optional[str] = None,
    tyre_name:   str  = DEFAULT_TYRE_NAME,
    side_configs: Optional[Dict[str, Dict[str, Any]]] = None,
    sides_to_run: Optional[List[str]] = None,
    # ── Camera capture args (only used when CAMERA_CAPTURE_ENABLED=True) ──
    multi_camera_manager=None,
    # ── Demo / local mode: point directly at an existing capture root ──
    demo_capture_root: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Full inspection cycle:

    CAMERA MODE  (CAMERA_CAPTURE_ENABLED = True)
    ─────────────────────────────────────────────
    1. Creates  media/capture/<DD-MM-YYYY>/Cycle_N/
    2. Captures images from all cameras via *multi_camera_manager*
    3. Saves each image to  Cycle_N/serial_XXXXXXXXX/image.png
    4. Runs AI inference on the saved images
    5. Writes outputs to  media/output/Cycle_N/

    DEMO / LOCAL MODE  (CAMERA_CAPTURE_ENABLED = False)
    ────────────────────────────────────────────────────
    Uses *demo_capture_root* (or the latest Cycle_N folder under
    media/capture/<today>/) to find existing images, then runs inference.
    """
    sides_to_run = _resolve_sides(sides_to_run)
    media_root   = os.path.abspath(media_root)
    device       = _normalize_device(device)

    os.makedirs(media_root, exist_ok=True)

    seg_model_a_path    = _required_file(seg_model_a_path,    "seg_model_a_path")
    seg_model_b_path    = _required_file(seg_model_b_path,    "seg_model_b_path")
    vit_checkpoint_path = _required_file(vit_checkpoint_path, "vit_checkpoint_path")
    r_detector_path     = _required_file(r_detector_path,     "r_detector_path")

    sku_calibration_dir = _get_sku_calibration_dir(media_root, sku_name)
    shared_artifacts_dir = _get_sku_artifacts_dir(media_root, sku_name)

    print(f"[MAIN] selected sku_name     : {sku_name}")
    print(f"[MAIN] selected tyre_name    : {tyre_name}")
    print(f"[MAIN] sku_calibration_dir   : {sku_calibration_dir}")
    print(f"[MAIN] sku_artifacts_dir     : {shared_artifacts_dir}")

    # ── Step 1: Determine cycle capture directory & cycle_id ──────────────
    if CAMERA_CAPTURE_ENABLED:
        # Real camera mode: create a fresh dated Cycle_N folder
        cycle_capture_dir, cycle_id = build_cycle_capture_dir(media_root)
    else:
        # Demo / local mode
        if demo_capture_root:
            # Caller supplied an explicit folder (e.g. the old LOCAL_MULTI_SIDE_TEST_FOLDER)
            cycle_capture_dir = os.path.abspath(demo_capture_root)
            if cycle_id is None:
                cycle_id = f"Cycle_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        else:
            # Auto-detect the latest Cycle_N under today's dated folder
            today_root = _get_today_capture_root(media_root)
            existing   = [
                d for d in os.listdir(today_root)
                if os.path.isdir(os.path.join(today_root, d)) and d.startswith("Cycle_")
            ]
            if not existing:
                raise FileNotFoundError(
                    f"No Cycle_N folders found under {today_root}. "
                    "Place images there or enable CAMERA_CAPTURE_ENABLED."
                )
            existing.sort(key=lambda d: int(d.split("_", 1)[1]))
            latest_cycle      = existing[-1]
            cycle_capture_dir = os.path.join(today_root, latest_cycle)
            cycle_id          = cycle_id or latest_cycle

    print(f"[MAIN] cycle_capture_dir : {cycle_capture_dir}")
    print(f"[MAIN] cycle_id          : {cycle_id}")

    # ── Step 2: Build image map ───────────────────────────────────────────
    if CAMERA_CAPTURE_ENABLED:
        if multi_camera_manager is None:
            raise ValueError(
                "CAMERA_CAPTURE_ENABLED=True but multi_camera_manager was not passed."
            )
        image_map = capture_and_save_images(
            multi_camera_manager, cycle_capture_dir, sides_to_run
        )
    else:
        image_map = build_image_map_from_capture_dir(cycle_capture_dir, sides_to_run)

    print("[MAIN] image_map:")
    for side_name in sides_to_run:
        print(f"    {side_name}: {image_map.get(side_name, 'MISSING')}")

    # ── Step 3: Build / reuse runtimes ───────────────────────────────────
    runtimes = build_all_runtimes(
        sku_name=sku_name, media_root=media_root,
        seg_model_a_path=seg_model_a_path, seg_model_b_path=seg_model_b_path,
        vit_checkpoint_path=vit_checkpoint_path, r_detector_path=r_detector_path,
        device=device, capture_root=cycle_capture_dir,
        tyre_name=tyre_name, side_configs=side_configs, sides_to_run=sides_to_run,
    )
    _apply_tyre_name_to_runtimes(runtimes, tyre_name)

    _maybe_warmup_runtimes(
        runtimes=runtimes, sku_name=sku_name, device=device,
        capture_root=cycle_capture_dir,
        seg_model_a_path=seg_model_a_path, seg_model_b_path=seg_model_b_path,
        vit_checkpoint_path=vit_checkpoint_path, r_detector_path=r_detector_path,
        tyre_name=tyre_name, media_root=media_root, sides_to_run=sides_to_run,
    )

    # ── Step 4: GPU semaphores ────────────────────────────────────────────
    r_gpu_sem    = threading.Semaphore(R_ALIGN_GPU_CONCURRENCY)
    vit_gpu_sem  = threading.Semaphore(VIT_GPU_CONCURRENCY)
    yolo_gpu_sem = threading.Semaphore(YOLO_GPU_CONCURRENCY)

    # ── Step 5: Run inference cycle ───────────────────────────────────────
    output_root = os.path.join(media_root, "output")
    os.makedirs(output_root, exist_ok=True)

    result = run_cycle(
    image_map=image_map,
    runtimes=runtimes,
    output_root=output_root,
    cycle_id=cycle_id,
    sides_to_run=sides_to_run,
    r_gpu_sem=r_gpu_sem,
    vit_gpu_sem=vit_gpu_sem,
    yolo_gpu_sem=yolo_gpu_sem,
    sku_name=sku_name,
    tyre_name=tyre_name,
    )

    try:
        save_cycle_metadata(result)
        print(f"[DB] saved cycle metadata | cycle_id={result.get('cycle_id')}")
    except Exception as e:
        print(f"[DB][ERROR] save failed | error={e}")

    return result