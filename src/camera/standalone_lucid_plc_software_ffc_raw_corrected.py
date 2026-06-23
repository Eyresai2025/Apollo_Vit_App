# standalone_lucid_plc_software_ffc_raw_corrected.py
# ============================================================
# Lucid line-scan multi-camera capture with PLC SOFTWARE trigger
# + Software Flat Field Correction (FFC)
#
# What this file does:
#   - Main group trigger : PLC DB74.DBX0.3
#   - Bead trigger       : PLC DB74.DBX86.0
#   - Camera trigger     : TriggerSoftware after PLC rising edge
#   - Captures 42000 height image using 3 chunks of 14000
#   - Saves BOTH:
#       1) raw 16-bit PNG image
#       2) software FFC corrected 16-bit PNG image
#
# Important:
#   - No matplotlib
#   - No image display
#   - No histogram
#   - No gain plot
#   - Same physical camera serial 250500042 is opened only once
#     and used for both inner and bead roles.
#
# FFC method used here:
#   - Software column-gain correction.
#   - For every captured full image, column mean is calculated.
#   - Target column level is selected using PERCENTILE_95.
#   - Per-column gain = target / column_mean.
#   - Gain is clipped between GAIN_RANGE_MIN and GAIN_RANGE_MAX.
#   - Raw and corrected images are saved.
#
# For true production FFC, normally compute the gain table from a flat/white
# calibration image and reuse the same table. For your current testing, this
# file computes correction per captured image and saves raw + corrected output.
# ============================================================

import os
import time
import ctypes
import queue
import threading
import signal
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import cv2
import numpy as np

from arena_api.system import system
from arena_api.buffer import BufferFactory

try:
    import snap7 # type: ignore
    from snap7.util import get_bool # type: ignore
except Exception:
    snap7 = None
    get_bool = None


# ============================================================
# MODE
# ============================================================

CAPTURE_MODE = "PLC_SOFTWARE"
# FREE         -> no trigger, continuous camera stream
# SOFTWARE     -> Python directly executes TriggerSoftware
# PLC_SOFTWARE -> Siemens PLC tag HIGH, then Python executes TriggerSoftware


# ============================================================
# PLC SETTINGS
# ============================================================

PLC_IP = "192.168.10.1"
PLC_RACK = 0
PLC_SLOT = 1
PLC_DB = 74

MAIN_PLC_BYTE = 0
MAIN_PLC_BIT = 3          # DB74.DBX0.3

BEAD_PLC_BYTE = 86
BEAD_PLC_BIT = 0          # DB74.DBX86.0

# Important for bead because trigger comes after short delay.
PLC_POLL_DELAY_SEC = 0.005


# ============================================================
# GLOBAL CAPTURE SETTINGS
# ============================================================

# Default output folder: saves inside the same folder where this script is kept.
# You can also override it from CMD before running:
#   set APOLLO_FFC_SAVE_DIR=C:\Temp\Trail2_FFC_Test
SCRIPT_DIR = Path(__file__).resolve().parent
SAVE_DIR = os.environ.get(
    "APOLLO_FFC_SAVE_DIR",
    str(SCRIPT_DIR / "Trail_FFC_DEFECT_Amazer4G"),
)

NUM_FULL_IMAGES = 1
NUM_BEAD_IMAGES = 1

# 42000 final image = 3 buffers/chunks of 14000 height.
CAMERA_HEIGHT = 14000
FINAL_HEIGHT = 42000

# HEIGHT_BASED = present logic: capture until FINAL_HEIGHT rows
# TIME_BASED   = capture continuously for TIME_CAPTURE_SEC seconds
CAPTURE_BUILD_MODE = "HEIGHT_BASED"
TIME_CAPTURE_SEC = 2.0

PIXEL_FORMAT = "Mono16"

NUM_STREAM_BUFFERS = 16
BUFFER_TIMEOUT_MS = 30000

PNG_COMPRESSION = 0

# True  = save output PNG as 8-bit single-channel
# False = save output PNG as 16-bit single-channel
SAVE_AS_8BIT = True

# png or bmp
SAVE_IMAGE_FORMAT = "png"
# Keep this small because each 4K x 42000 Mono16 image is large.
SAVE_QUEUE_SIZE = 4

PACKET_SIZE = 9000
PACKET_DELAY = 1000

TRIGGER_ACTIVATION = "RisingEdge"
AFTER_TRIGGER_DELAY_SEC = 0.02

# Normal reset timing for 4K cameras.
AFTER_ACQ_STOP_DELAY_SEC = 0.15
AFTER_STOP_STREAM_DELAY_SEC = 0.20
AFTER_START_STREAM_DELAY_SEC = 0.25

FLUSH_COUNT = 16

# Shared inner/bead camera timing fix.
SHARED_INNER_BEAD_SERIAL = "250500042"

FAST_RESET_SHARED_CAMERA = False
FAST_RESET_SLEEP_SEC = 0.02
FAST_FLUSH_COUNT = 4
FAST_FLUSH_TIMEOUT_MS = 5

# If bead TriggerSoftware happens later than this after PLC edge,
# log will show LATE_TRIGGER.
MAX_ALLOWED_BEAD_TRIGGER_DELAY_MS = 75.0

# Keep this False for clean log.
VERBOSE_CONFIG_LOGS = False


# ============================================================
# SOFTWARE FFC SETTINGS
# ============================================================

ENABLE_SOFTWARE_FFC = True
SAVE_RAW_IMAGES = True
SAVE_CORRECTED_IMAGES = True

# Do not enable camera-side FFC here because we need true raw image also.
# This file captures raw from camera and applies software correction.
# If this is True and the camera node supports it, raw image will already
# be camera-corrected, so keep False for raw + corrected testing.
ENABLE_CAMERA_SIDE_FFC_DURING_CAPTURE = False

# Keep this False for clean production/test logs.
# Some Lucid models report FlatFieldCorrection* nodes as NOT_IMPLEMENTED from Arena Python.
# That is OK because this script uses software FFC after capture.
TRY_CAMERA_SIDE_FFC_NODES = False
FFC_SELECTOR = "FlatFieldCorrection1"

GAIN_RANGE_MIN = 1.0
GAIN_RANGE_MAX = 15.99
GAIN_TARGET_MODE = "PERCENTILE_95"       # options: MAX / MEAN / PERCENTILE_95

# Apply correction in row blocks to avoid high RAM usage.
FFC_ROW_BLOCK = 512

# Optional: save per-image gain table as .npy for debugging/testing.
SAVE_GAIN_NPY = False


# ============================================================
# CAMERA SERIAL CONFIG
#
# IMPORTANT:
# Do NOT repeat serial 250500042 twice.
# Same camera has two roles: inner + bead.
#
# 4K cameras: width 4096
# 2K camera : width 2048
# ============================================================

CAMERA_CONFIGS: Dict[str, Dict[str, Any]] = {
    "254901428": {
        "enabled": True,
        "camera_name": "sidewall2",
        "width": 4096,
        "line_rate": 8169.0,
        "exposure_us": 122.0,
        "gain": 12.0,
        "roles": [
            {"name": "sidewall2", "group": "main", "enabled": True},
        ],
    },
    "254901432": {
        "enabled": True,
        "camera_name": "sidewall1",
        "width": 4096,
        "line_rate": 8169.0,
        "exposure_us": 122.0,
        "gain": 12.0,
        "roles": [
            {"name": "sidewall1", "group": "main", "enabled": True},
        ],
    },
    "254901430": {
        "enabled": True,
        "camera_name": "tread",
        "width": 4096,
        "line_rate": 8169.0,
        "exposure_us": 122.0,
        "gain": 12.0,
        "roles": [
            {"name": "tread", "group": "main", "enabled": True},
        ],
    },
    "250500042": {
        "enabled": True,
        "camera_name": "inner_camera_used_for_inner_and_bead",
        "width": 2048,
        "line_rate": None,       # 2K camera: line-rate node skipped
        "exposure_us": 120.0,
        "gain": 12.0,
        "roles": [
            {"name": "inner", "group": "main", "enabled": True},
            {"name": "bead", "group": "bead", "enabled": True},
        ],
    },
}


# ============================================================
# GLOBALS
# ============================================================

save_queue: "queue.Queue[Optional[Tuple[str, str, int, np.ndarray, Dict[str, Any]]]]" = queue.Queue(
    maxsize=SAVE_QUEUE_SIZE
)
running = True

shutdown_event = threading.Event()


def request_shutdown(reason: str = "user requested stop") -> None:
    global running

    if not shutdown_event.is_set():
        log(f"[STOP] shutdown requested: {reason}")

    shutdown_event.set()
    running = False

    try:
        save_queue.put_nowait(None)
    except Exception:
        pass


def _handle_sigint(signum, frame) -> None:
    request_shutdown("Ctrl+C / SIGINT")


try:
    signal.signal(signal.SIGINT, _handle_sigint)
except Exception:
    pass
# ============================================================
# LOGGING
# ============================================================

def log(msg: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    print(f"{ts} | {msg}", flush=True)


# ============================================================
# NODE HELPERS
# ============================================================

def set_node(nodemap, name: str, value: Any, verbose: Optional[bool] = None) -> bool:
    if verbose is None:
        verbose = VERBOSE_CONFIG_LOGS

    try:
        node = nodemap.get_node(name)
        if node and node.is_writable:
            node.value = value
            if verbose:
                log(f"[SET_OK] {name}={node.value}")
            return True

        if verbose:
            log(f"[SET_SKIP] {name} not writable/not found")
        return False

    except Exception as e:
        log(f"[SET_FAIL] {name} -> {value}: {e}")
        return False


def read_node(nodemap, name: str, default: Any = "-") -> Any:
    try:
        node = nodemap.get_node(name)
        if node and node.is_readable:
            return node.value
    except Exception:
        pass
    return default


def execute_node(nodemap, name: str) -> bool:
    try:
        node = nodemap.get_node(name)
        if node:
            node.execute()
            return True
    except Exception as e:
        log(f"[EXEC_FAIL] {name}: {e}")
    return False


# ============================================================
# CAMERA-SIDE FFC NODE HELPERS
# ============================================================

def try_select_camera_ffc_slot(nodemap) -> bool:
    for value in (FFC_SELECTOR, "Flat Field Correction 1"):
        try:
            node = nodemap.get_node("FlatFieldCorrectionSelector")
            if node and node.is_writable:
                node.value = value
                log(f"[FFC_CAMERA] FlatFieldCorrectionSelector={value}")
                return True
        except Exception as e:
            log(f"[FFC_CAMERA] selector value failed {value}: {e}")
    return False


def try_enable_camera_side_ffc(nodemap, enable: bool) -> bool:
    try_select_camera_ffc_slot(nodemap)
    try:
        node = nodemap.get_node("FlatFieldCorrectionEnable")
        if node and node.is_writable:
            node.value = bool(enable)
            log(f"[FFC_CAMERA] FlatFieldCorrectionEnable={node.value}")
            return True
        log("[FFC_CAMERA] FlatFieldCorrectionEnable not writable/not found")
    except Exception as e:
        log(f"[FFC_CAMERA] Could not set FlatFieldCorrectionEnable={enable}: {e}")
    return False


# ============================================================
# PLC HELPERS
# ============================================================

def create_plc_client():
    if snap7 is None:
        raise RuntimeError("python-snap7 not installed. Install with: pip install python-snap7")

    plc = snap7.client.Client()
    plc.connect(PLC_IP, PLC_RACK, PLC_SLOT)

    if not plc.get_connected():
        raise RuntimeError(f"PLC connection failed: {PLC_IP}")

    return plc


def read_plc_bool(plc, db: int, byte: int, bit: int) -> bool:
    data = plc.db_read(db, byte, 1)
    return get_bool(data, 0, bit)


def wait_plc_fresh_rising_edge(plc, byte: int, bit: int, label: str) -> Optional[float]:
    """
    Safe PLC edge wait:
    1. If bit is already HIGH, wait until LOW.
    2. Then wait for fresh LOW -> HIGH.
    3. Ctrl+C safe.
    """
    tag = f"DB{PLC_DB}.DBX{byte}.{bit}"

    log(f"[{label}] PLC {tag} WAIT_LOW")

    while not shutdown_event.is_set():
        state = read_plc_bool(plc, PLC_DB, byte, bit)

        if not state:
            log(f"[{label}] PLC {tag} LOW_READY")
            break

        time.sleep(PLC_POLL_DELAY_SEC)

    if shutdown_event.is_set():
        log(f"[{label}] PLC {tag} STOPPED_WHILE_WAIT_LOW")
        return None

    log(f"[{label}] PLC {tag} WAIT_HIGH")

    while not shutdown_event.is_set():
        state = read_plc_bool(plc, PLC_DB, byte, bit)

        if state:
            edge_ts = time.perf_counter()
            log(f"[{label}] PLC {tag} HIGH_EDGE")
            return edge_ts

        time.sleep(PLC_POLL_DELAY_SEC)

    log(f"[{label}] PLC {tag} STOPPED_WHILE_WAIT_HIGH")
    return None


# ============================================================
# IMAGE / BUFFER HELPERS
# ============================================================

def convert_buffer(buffer) -> np.ndarray:
    copied = BufferFactory.copy(buffer)

    try:
        width = int(copied.width)
        height = int(copied.height)
        total_bytes = len(copied.data)

        c_arr = (ctypes.c_ubyte * total_bytes).from_address(
            ctypes.addressof(copied.pbytes)
        )
        np_arr = np.ctypeslib.as_array(c_arr)

        bpp = total_bytes // (width * height)

        if bpp == 2:
            img = np_arr.view(np.uint16).reshape(height, width)
        else:
            img = np_arr.reshape(height, width)

        return img.copy()

    finally:
        BufferFactory.destroy(copied)


def flush_buffers(camera, cam_name: str, max_count: int = FLUSH_COUNT, timeout_ms: int = 100, log_it: bool = True) -> int:
    flushed = 0

    for _ in range(max_count):
        try:
            buf = camera.get_buffer(timeout=timeout_ms)
            camera.requeue_buffer(buf)
            flushed += 1
        except Exception:
            break

    if log_it:
        log(f"[{cam_name}] FLUSH buffers={flushed}")

    return flushed

def get_buffer_interruptible(camera, role_tag: str, total_timeout_ms: int = BUFFER_TIMEOUT_MS):
    """
    Arena get_buffer with one huge timeout can make Ctrl+C stuck.
    This waits in 500 ms steps so Ctrl+C can stop quickly.
    """
    start = time.perf_counter()
    last_error = None

    while not shutdown_event.is_set():
        try:
            return camera.get_buffer(timeout=500)
        except Exception as e:
            last_error = e

            elapsed_ms = (time.perf_counter() - start) * 1000.0

            if elapsed_ms >= total_timeout_ms:
                raise RuntimeError(
                    f"[{role_tag}] get_buffer timeout/error after "
                    f"{elapsed_ms:.0f} ms: {last_error}"
                )

    raise RuntimeError(f"[{role_tag}] stop requested while waiting for camera buffer")

def save_uint16_png(path: Path, image: np.ndarray) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)

    if image.ndim != 2:
        raise RuntimeError(
            f"Expected single-channel Mono image, got shape={image.shape}, dtype={image.dtype}"
        )

    if SAVE_AS_8BIT:
        # Save as 8-bit single-channel PNG
        if image.dtype != np.uint16:
            image = image.astype(np.uint16)

        save_img = (image / 256).clip(0, 255).astype(np.uint8)

    else:
        # Save as 16-bit single-channel PNG
        if image.dtype != np.uint16:
            save_img = image.astype(np.uint16)
        else:
            save_img = image

    log(
        f"[SAVE_DEBUG] path={path} "
        f"shape={save_img.shape} dtype={save_img.dtype} ndim={save_img.ndim}"
    )

    ok = cv2.imwrite(
        str(path),
        save_img,
        [cv2.IMWRITE_PNG_COMPRESSION, PNG_COMPRESSION],
    )

    return bool(ok)


# ============================================================
# SOFTWARE FFC HELPERS - NO PLOTS / NO HISTOGRAM / NO DISPLAY
# ============================================================

def get_target_pixel(column_profile: np.ndarray) -> float:
    if GAIN_TARGET_MODE == "MAX":
        target = float(np.max(column_profile))
    elif GAIN_TARGET_MODE == "MEAN":
        target = float(np.mean(column_profile))
    elif GAIN_TARGET_MODE == "PERCENTILE_95":
        target = float(np.percentile(column_profile, 95))
    else:
        raise RuntimeError(f"Unknown GAIN_TARGET_MODE: {GAIN_TARGET_MODE}")

    return target


def compute_ffc_gain_from_image(image: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Compute one gain value per column.
    This follows your FFC code logic, but removes plots and histogram.
    """
    if image.ndim != 2:
        raise RuntimeError(f"FFC expects 2D Mono image. Got shape={image.shape}")

    column_profile = np.mean(image, axis=0, dtype=np.float64)
    target = get_target_pixel(column_profile)

    epsilon = 1e-6
    gain_values = np.where(
        column_profile > epsilon,
        target / column_profile,
        1.0,
    )

    gain_values = np.clip(gain_values, GAIN_RANGE_MIN, GAIN_RANGE_MAX).astype(np.float32)

    stats = {
        "target_mode": GAIN_TARGET_MODE,
        "target": target,
        "profile_min": float(np.min(column_profile)),
        "profile_max": float(np.max(column_profile)),
        "profile_mean": float(np.mean(column_profile)),
        "gain_min": float(np.min(gain_values)),
        "gain_max": float(np.max(gain_values)),
        "gain_mean": float(np.mean(gain_values)),
        "gain_count_at_max": int(np.sum(gain_values >= GAIN_RANGE_MAX)),
    }

    return gain_values, stats


def apply_software_ffc_chunked(image: np.ndarray, gain_values: np.ndarray) -> Tuple[np.ndarray, int]:
    """
    Apply per-column gain in row blocks to avoid huge temporary arrays.
    """
    if image.ndim != 2:
        raise RuntimeError(f"FFC expects 2D image. Got shape={image.shape}")

    height, width = image.shape
    if len(gain_values) != width:
        raise RuntimeError(f"Gain width mismatch: gains={len(gain_values)}, image_width={width}")

    corrected = np.empty_like(image, dtype=np.uint16)
    gain_2d = gain_values.reshape(1, -1).astype(np.float32)
    saturated_count = 0

    for row0 in range(0, height, FFC_ROW_BLOCK):
        row1 = min(row0 + FFC_ROW_BLOCK, height)

        block = image[row0:row1, :].astype(np.float32, copy=False)
        block = block * gain_2d
        np.clip(block, 0, 65535, out=block)

        saturated_count += int(np.sum(block >= 65535))
        corrected[row0:row1, :] = block.astype(np.uint16)

    return corrected, saturated_count


def save_worker() -> None:
    global running

    while running or not save_queue.empty():
        try:
            item = save_queue.get(timeout=1)
        except queue.Empty:
            continue

        if item is None:
            save_queue.task_done()
            continue

        role_name, serial, image_index, image, info = item
        role_tag = role_name.upper()

        try:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            cycle_dir = Path(SAVE_DIR) / f"Cycle_{int(image_index)}" / role_name

            raw_path = cycle_dir / "raw" / f"{role_name}_{serial}_Cycle_{int(image_index)}_{ts}_raw.png"
            corrected_path = cycle_dir / "corrected" / f"{role_name}_{serial}_Cycle_{int(image_index)}_{ts}_ffc_corrected.png"
            gain_path = cycle_dir / "gain" / f"{role_name}_{serial}_Cycle_{int(image_index)}_{ts}_ffc_gain.npy"

            if SAVE_RAW_IMAGES:
                ok_raw = save_uint16_png(raw_path, image)
                if ok_raw:
                    log(f"[{role_tag}] SAVE_RAW_OK {raw_path}")
                else:
                    log(f"[{role_tag}] SAVE_RAW_ERROR {raw_path}")

            if ENABLE_SOFTWARE_FFC and SAVE_CORRECTED_IMAGES:
                gain_values, stats = compute_ffc_gain_from_image(image)
                corrected, saturated = apply_software_ffc_chunked(image, gain_values)

                ok_corr = save_uint16_png(corrected_path, corrected)
                if ok_corr:
                    log(f"[{role_tag}] SAVE_FFC_OK {corrected_path}")
                else:
                    log(f"[{role_tag}] SAVE_FFC_ERROR {corrected_path}")

                log(
                    f"[{role_tag}] FFC_STATS serial={serial} img={image_index} "
                    f"target_mode={stats['target_mode']} target={stats['target']:.2f} "
                    f"profile_min={stats['profile_min']:.2f} profile_max={stats['profile_max']:.2f} "
                    f"gain_min={stats['gain_min']:.4f} gain_max={stats['gain_max']:.4f} "
                    f"gain_at_max={stats['gain_count_at_max']} saturated_pixels={saturated}"
                )

                if SAVE_GAIN_NPY:
                    gain_path.parent.mkdir(parents=True, exist_ok=True)
                    np.save(str(gain_path), gain_values)
                    log(f"[{role_tag}] SAVE_GAIN_OK {gain_path}")

                del corrected

        except Exception as e:
            log(f"[{role_tag}] SAVE_WORKER_ERROR serial={serial} img={image_index}: {e}")

        finally:
            del image
            save_queue.task_done()


# ============================================================
# CAMERA CONFIGURATION
# ============================================================

def configure_stream(camera, cam_name: str) -> None:
    tl = camera.tl_stream_nodemap

    set_node(tl, "StreamAutoNegotiatePacketSize", True)
    set_node(tl, "StreamPacketResendEnable", True)
    set_node(tl, "StreamBufferHandlingMode", "OldestFirst")

    if VERBOSE_CONFIG_LOGS:
        log(f"[{cam_name}] stream configured")


def configure_camera(camera) -> Optional[Dict[str, Any]]:
    nodemap = camera.nodemap

    serial = str(read_node(nodemap, "DeviceSerialNumber", "UNKNOWN"))

    if serial not in CAMERA_CONFIGS:
        log(f"[SKIP] Unknown serial: {serial}")
        return None

    cfg = CAMERA_CONFIGS[serial]

    if not cfg.get("enabled", True):
        log(f"[SKIP] Camera disabled in config: {serial}")
        return None

    camera_name = cfg["camera_name"]
    width = int(cfg["width"])
    line_rate = cfg.get("line_rate")
    exposure_us = float(cfg.get("exposure_us", 120.0))
    gain = float(cfg.get("gain", 12.0))

    cam_name = f"{camera_name}/{serial}"

    log(f"[CONFIG] START serial={serial} name={camera_name}")

    configure_stream(camera, cam_name)

    # Always turn trigger off while configuring.
    set_node(nodemap, "TriggerMode", "Off")

    set_node(nodemap, "Width", width)
    set_node(nodemap, "Height", CAMERA_HEIGHT)
    set_node(nodemap, "PixelFormat", PIXEL_FORMAT)
    set_node(nodemap, "AcquisitionMode", "Continuous")

    if line_rate is not None:
        set_node(nodemap, "AcquisitionLineRateEnable", True)
        set_node(nodemap, "AcquisitionLineRate", float(line_rate))

        time.sleep(0.05)

        safe_exposure = min(
            exposure_us,
            0.90 * (1_000_000.0 / float(line_rate)),
        )
    else:
        safe_exposure = exposure_us
        log(f"[CONFIG] serial={serial} 2K camera line-rate skipped")

    set_node(nodemap, "ExposureAutoLimitAuto", "Off")
    time.sleep(0.02)
    set_node(nodemap, "ExposureTime", float(safe_exposure))
    set_node(nodemap, "Gain", float(gain))

    set_node(nodemap, "GevSCPSPacketSize", PACKET_SIZE)
    set_node(nodemap, "GevSCPD", PACKET_DELAY)

    # Keep camera-side FFC disabled for true raw + software-corrected testing.
    # We do not touch FlatFieldCorrection* camera nodes by default because these
    # nodes return NOT_IMPLEMENTED on your cameras through Arena Python.
    # Software FFC below still works and saves raw + corrected images.
    if TRY_CAMERA_SIDE_FFC_NODES:
        try_enable_camera_side_ffc(
            nodemap,
            bool(ENABLE_CAMERA_SIDE_FFC_DURING_CAPTURE),
        )
    else:
        log(f"[FFC_CAMERA] camera-side FFC node setup skipped; using software FFC only")

    if CAPTURE_MODE == "FREE":
        set_node(nodemap, "TriggerMode", "Off")

    elif CAPTURE_MODE in ["SOFTWARE", "PLC_SOFTWARE"]:
        set_node(nodemap, "TriggerSelector", "AcquisitionStart")
        set_node(nodemap, "TriggerSource", "Software")
        set_node(nodemap, "TriggerActivation", TRIGGER_ACTIVATION)
        set_node(nodemap, "TriggerMode", "On")

    else:
        raise RuntimeError("This file supports PLC_SOFTWARE / SOFTWARE / FREE only.")

    actual_width = read_node(nodemap, "Width")
    actual_height = read_node(nodemap, "Height")
    actual_exp = read_node(nodemap, "ExposureTime")
    actual_gain = read_node(nodemap, "Gain")
    actual_line_rate = read_node(nodemap, "AcquisitionLineRate")
    trigger_selector = read_node(nodemap, "TriggerSelector")
    trigger_source = read_node(nodemap, "TriggerSource")
    trigger_mode = read_node(nodemap, "TriggerMode")

    log(
        f"[CONFIG] OK serial={serial} width={actual_width} height={actual_height} "
        f"exp={actual_exp} gain={actual_gain} line_rate={actual_line_rate} "
        f"trigger={trigger_selector}/{trigger_source}/{trigger_mode}"
    )

    return {
        "serial": serial,
        "camera_name": camera_name,
        "width": int(width),
        "camera_height": CAMERA_HEIGHT,
        "final_height": FINAL_HEIGHT,
        "cam_name": cam_name,
    }


def get_stream_buffer_count(info: Dict[str, Any]) -> int:
    return NUM_STREAM_BUFFERS


# ============================================================
# CAPTURE
# ============================================================

def reset_acquisition_for_next_cycle(camera, info: Dict[str, Any], role_name: Optional[str] = None) -> None:
    """
    For TriggerSelector=AcquisitionStart:
    After capture, stop acquisition so next TriggerSoftware can start fresh.

    For shared inner/bead serial 250500042:
        AcquisitionStop -> small sleep -> small flush

    Do not stop_stream/start_stream for this shared camera,
    because bead trigger comes quickly.
    """
    nodemap = camera.nodemap
    cam_name = info["cam_name"]
    serial = info["serial"]
    role_tag = (role_name or "camera").upper()

    if CAPTURE_MODE not in ["SOFTWARE", "PLC_SOFTWARE"]:
        return

    reset_start = time.perf_counter()

    execute_node(nodemap, "AcquisitionStop")

    if FAST_RESET_SHARED_CAMERA and serial == SHARED_INNER_BEAD_SERIAL:
        time.sleep(FAST_RESET_SLEEP_SEC)

        flushed = flush_buffers(
            camera,
            cam_name,
            max_count=FAST_FLUSH_COUNT,
            timeout_ms=FAST_FLUSH_TIMEOUT_MS,
            log_it=False,
        )

        reset_ms = (time.perf_counter() - reset_start) * 1000.0

        log(
            f"[{role_tag}] CAMERA_READY serial={serial} "
            f"fast_reset_ms={reset_ms:.1f} flushed={flushed}"
        )
        return

    time.sleep(AFTER_ACQ_STOP_DELAY_SEC)

    try:
        camera.stop_stream()
    except Exception as e:
        log(f"[{role_tag}] stop_stream warning: {e}")

    time.sleep(AFTER_STOP_STREAM_DELAY_SEC)

    try:
        camera.start_stream(get_stream_buffer_count(info))
    except Exception as e:
        log(f"[{role_tag}] start_stream warning: {e}")

    time.sleep(AFTER_START_STREAM_DELAY_SEC)

    flushed = flush_buffers(camera, cam_name, log_it=False)

    reset_ms = (time.perf_counter() - reset_start) * 1000.0

    log(
        f"[{role_tag}] CAMERA_READY serial={serial} "
        f"normal_reset_ms={reset_ms:.1f} flushed={flushed}"
    )


def get_capture_dtype():
    return np.uint8 if str(PIXEL_FORMAT).strip().lower() == "mono8" else np.uint16


def capture_time_based_image(camera, info: Dict[str, Any], task: "CaptureTask") -> np.ndarray:
    """
    Time-based capture mode.

    Instead of stopping after FINAL_HEIGHT rows, collect every camera buffer
    for TIME_CAPTURE_SEC seconds and vertically stack the frames into one
    final single-channel image. Raw/FFC/save logic remains unchanged.
    """
    width = int(info["width"])
    serial = info["serial"]

    role_name = task.role_name
    role_tag = role_name.upper()
    image_index = task.image_index

    capture_sec = max(0.1, float(TIME_CAPTURE_SEC))
    expected_dtype = get_capture_dtype()

    frames = []
    total_rows = 0
    chunk_id = 0

    start_time = time.perf_counter()
    end_time = start_time + capture_sec

    log(
        f"[{role_tag}] TIME_CAPTURE_START serial={serial} "
        f"img={image_index} duration_sec={capture_sec:.2f}"
    )

    while time.perf_counter() < end_time:
        if shutdown_event.is_set():
            raise RuntimeError(f"[{role_tag}] stop requested during time capture")

        remaining_ms = int(max(1, (end_time - time.perf_counter()) * 1000.0))
        timeout_ms = min(500, remaining_ms)

        try:
            buffer = camera.get_buffer(timeout=timeout_ms)
        except Exception:
            # No frame available during this small slice. Continue until time ends.
            continue

        try:
            frame = convert_buffer(buffer)
        finally:
            camera.requeue_buffer(buffer)

        if frame.ndim != 2:
            raise RuntimeError(f"[{role_tag}] expected 2D frame, got shape={frame.shape}")

        h, w = frame.shape

        if w != width:
            log(f"[{role_tag}] WIDTH_WARNING got={w} expected={width}")

        copy_w = min(w, width)
        frame = frame[:, :copy_w]

        if frame.dtype != expected_dtype:
            frame = frame.astype(expected_dtype, copy=False)

        if copy_w < width:
            padded = np.zeros((h, width), dtype=expected_dtype)
            padded[:, :copy_w] = frame
            frame = padded

        frames.append(frame.copy())
        total_rows += int(frame.shape[0])
        chunk_id += 1

        log(
            f"[{role_tag}] TIME_CHUNK {chunk_id} "
            f"rows_added={frame.shape[0]} total_rows={total_rows}"
        )

    if not frames:
        raise RuntimeError(f"[{role_tag}] no frames captured in {capture_sec:.2f} sec")

    full_img = np.vstack(frames).astype(expected_dtype, copy=False)
    elapsed = time.perf_counter() - start_time

    log(
        f"[{role_tag}] TIME_STITCH_DONE serial={serial} "
        f"img={image_index} chunks={chunk_id} rows={full_img.shape[0]} "
        f"width={full_img.shape[1]} time={elapsed:.2f}s"
    )

    return full_img

def capture_one_full_image(camera, info: Dict[str, Any], task: "CaptureTask") -> None:
    nodemap = camera.nodemap
    width = int(info["width"])
    final_height = int(info.get("final_height", FINAL_HEIGHT))
    camera_height = int(info.get("camera_height", CAMERA_HEIGHT))
    serial = info["serial"]

    role_name = task.role_name
    role_tag = role_name.upper()
    image_index = task.image_index

    if CAPTURE_MODE in ["SOFTWARE", "PLC_SOFTWARE"]:
        trigger_before = time.perf_counter()
        delay_from_plc_ms = (trigger_before - task.plc_edge_ts) * 1000.0

        late_msg = ""
        if role_name == "bead" and delay_from_plc_ms > MAX_ALLOWED_BEAD_TRIGGER_DELAY_MS:
            late_msg = " LATE_TRIGGER"

        log(
            f"[{role_tag}] TRIGGER_SOFTWARE serial={serial} "
            f"img={image_index} plc_to_trigger_ms={delay_from_plc_ms:.1f}{late_msg}"
        )

        execute_node(nodemap, "TriggerSoftware")
        time.sleep(AFTER_TRIGGER_DELAY_SEC)

    elif CAPTURE_MODE == "FREE":
        log(f"[{role_tag}] FREE_CAPTURE serial={serial} img={image_index}")

    else:
        raise RuntimeError(f"Unsupported CAPTURE_MODE: {CAPTURE_MODE}")

    capture_build_mode = str(CAPTURE_BUILD_MODE).strip().upper()

    if capture_build_mode == "TIME_BASED":
        full_img = capture_time_based_image(camera, info, task)

        # Save raw + FFC corrected image in background.
        # Same save worker handles raw/FFC/PNG/BMP/8-bit/16-bit.
        save_queue.put((role_name, serial, image_index, full_img, dict(info)))

        log(
            f"[{role_tag}] SAVE_QUEUED time_based_raw_and_ffc "
            f"serial={serial} img={image_index}"
        )
        return

    if capture_build_mode != "HEIGHT_BASED":
        log(
            f"[{role_tag}] CAPTURE_BUILD_MODE_WARNING unknown={CAPTURE_BUILD_MODE}; "
            f"using HEIGHT_BASED"
        )

    full_dtype = get_capture_dtype()
    full_img = np.zeros((final_height, width), dtype=full_dtype)

    current_row = 0
    chunk_id = 0
    expected_chunks = int(np.ceil(final_height / camera_height))
    start_time = time.perf_counter()

    while current_row < final_height:
        if shutdown_event.is_set():
            raise RuntimeError(f"[{role_tag}] stop requested before buffer capture")

        buffer = get_buffer_interruptible(camera, role_tag, BUFFER_TIMEOUT_MS)

        try:
            frame = convert_buffer(buffer)
        finally:
            camera.requeue_buffer(buffer)

        h, w = frame.shape

        if w != width:
            log(f"[{role_tag}] WIDTH_WARNING got={w} expected={width}")

        copy_h = min(h, final_height - current_row)
        copy_w = min(w, width)

        # Supports both Mono16 and Mono8 and protects against width mismatch.
        full_img[
            current_row:current_row + copy_h,
            0:copy_w
        ] = frame[:copy_h, :copy_w].astype(full_dtype, copy=False)

        current_row += copy_h
        chunk_id += 1

        log(
            f"[{role_tag}] CHUNK {chunk_id}/{expected_chunks} "
            f"rows={current_row}/{final_height}"
        )

    elapsed = time.perf_counter() - start_time

    log(
        f"[{role_tag}] STITCH_DONE serial={serial} "
        f"img={image_index} rows={current_row}/{final_height} time={elapsed:.2f}s"
    )

    # Save raw + FFC corrected image in background.
    # Queue may block if disk saving is slow, which is safer for RAM.
    save_queue.put((role_name, serial, image_index, full_img, dict(info)))

    log(f"[{role_tag}] SAVE_QUEUED raw_and_ffc serial={serial} img={image_index}")


# ============================================================
# CAMERA ACTOR
# One actor = one physical camera.
# ============================================================

@dataclass
class CaptureTask:
    role_name: str
    group: str
    image_index: int
    plc_edge_ts: float
    submit_ts: float
    done_event: threading.Event
    error: list


class CameraActor:
    def __init__(self, camera):
        self.camera = camera
        self.serial = str(read_node(camera.nodemap, "DeviceSerialNumber", "UNKNOWN"))
        self.cfg = CAMERA_CONFIGS[self.serial]

        self.q: "queue.Queue[Optional[CaptureTask]]" = queue.Queue()
        self.thread: Optional[threading.Thread] = None

        self.ready_event = threading.Event()
        self.error: Optional[Exception] = None
        self.info: Optional[Dict[str, Any]] = None

        self.state_lock = threading.Lock()
        self.state = "STARTING"

    def set_state(self, state: str) -> None:
        with self.state_lock:
            self.state = state

    def is_ready(self) -> bool:
        with self.state_lock:
            return self.state == "READY" and self.q.empty()

    def start(self) -> None:
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

        self.ready_event.wait()

        if self.error is not None:
            raise RuntimeError(f"[{self.serial}] camera actor failed: {self.error}")

    def submit(self, role_name: str, group: str, image_index: int, plc_edge_ts: Optional[float] = None) -> CaptureTask:
        task = CaptureTask(
            role_name=role_name,
            group=group,
            image_index=image_index,
            plc_edge_ts=plc_edge_ts or time.perf_counter(),
            submit_ts=time.perf_counter(),
            done_event=threading.Event(),
            error=[],
        )

        ready = self.is_ready()

        log(
            f"[{group.upper()}] QUEUE role={role_name} "
            f"img={image_index} serial={self.serial} camera_ready={ready}"
        )

        self.q.put(task)

        return task

    def stop(self) -> None:
        try:
            self.q.put_nowait(None)
        except Exception:
            pass

        try:
            self.camera.stop_stream()
        except Exception:
            pass

        if self.thread is not None:
            self.thread.join(timeout=3.0)

            if self.thread.is_alive():
                log(f"[STOP] camera actor still alive serial={self.serial}")

    def _run(self) -> None:
        try:
            self.info = configure_camera(self.camera)

            if self.info is None:
                raise RuntimeError("configure_camera returned None")

            self.camera.start_stream(get_stream_buffer_count(self.info))

            flush_buffers(self.camera, self.info["cam_name"], log_it=False)

            self.set_state("READY")
            log(f"[READY] serial={self.serial} camera_ready=True")

            self.ready_event.set()

            while not shutdown_event.is_set():
                try:
                    task = self.q.get(timeout=0.1)
                except queue.Empty:
                    continue

                if task is None:
                    self.q.task_done()
                    break

                try:
                    self.set_state("BUSY")

                    queue_wait_ms = (time.perf_counter() - task.submit_ts) * 1000.0
                    edge_wait_ms = (time.perf_counter() - task.plc_edge_ts) * 1000.0

                    log(
                        f"[{task.role_name.upper()}] CAMERA_START serial={self.serial} "
                        f"img={task.image_index} queue_wait_ms={queue_wait_ms:.1f} "
                        f"edge_wait_ms={edge_wait_ms:.1f}"
                    )

                    capture_one_full_image(
                        self.camera,
                        self.info,
                        task,
                    )

                    reset_acquisition_for_next_cycle(
                        self.camera,
                        self.info,
                        task.role_name,
                    )

                    self.set_state("READY")

                    log(
                        f"[{task.role_name.upper()}] DONE serial={self.serial} "
                        f"img={task.image_index}"
                    )

                except Exception as e:
                    self.set_state("ERROR")
                    task.error.append(str(e))

                    log(
                        f"[{task.role_name.upper()}] ERROR serial={self.serial} "
                        f"img={task.image_index}: {e}"
                    )

                finally:
                    task.done_event.set()
                    self.q.task_done()

        except Exception as e:
            self.error = e
            self.ready_event.set()

        finally:
            try:
                self.camera.stop_stream()
            except Exception:
                pass


# ============================================================
# ROLE TARGETS
# ============================================================

def build_role_targets(actors, group_name: str):
    targets = []

    for actor in actors:
        roles = actor.cfg.get("roles", [])

        for role in roles:
            if not role.get("enabled", True):
                continue

            if role.get("group") == group_name:
                targets.append((actor, role["name"]))

    return targets


def wait_all_tasks(tasks, label: str, cycle: int) -> None:
    for task in tasks:
        while not task.done_event.is_set():
            if shutdown_event.is_set():
                break
            task.done_event.wait(timeout=0.1)

    errors = []

    for task in tasks:
        if task.error:
            errors.append(
                f"role={task.role_name}, image={task.image_index}, error={task.error[0]}"
            )

    if shutdown_event.is_set():
        log(f"[{label}] CYCLE_DONE cycle={cycle} status=STOPPED")
        return

    if errors:
        log(f"[{label}] CYCLE_DONE cycle={cycle} status=ERROR")
        for e in errors:
            log(f"[{label}] {e}")
    else:
        log(f"[{label}] CYCLE_DONE cycle={cycle} status=OK")


# ============================================================
# PLC CONTROLLERS
# ============================================================

def main_plc_controller(main_targets) -> None:
    if not main_targets:
        log("[MAIN] no main camera roles enabled")
        return

    plc = create_plc_client()
    log(f"[MAIN] PLC_CONNECTED ip={PLC_IP}")

    try:
        for cycle in range(1, NUM_FULL_IMAGES + 1):
            log(f"[MAIN] WAIT_TRIGGER cycle={cycle}/{NUM_FULL_IMAGES}")

            edge_ts = wait_plc_fresh_rising_edge(
                plc,
                MAIN_PLC_BYTE,
                MAIN_PLC_BIT,
                "MAIN",
            )
            if edge_ts is None or shutdown_event.is_set():
                break
            log(f"[MAIN] RELEASE cycle={cycle}")

            tasks = []

            for actor, role_name in main_targets:
                task = actor.submit(
                    role_name,
                    "main",
                    cycle,
                    plc_edge_ts=edge_ts,
                )
                tasks.append(task)

            wait_all_tasks(tasks, "MAIN", cycle)

    finally:
        try:
            plc.disconnect()
        except Exception:
            pass

        log("[MAIN] PLC_DISCONNECTED")


def bead_plc_controller(bead_targets) -> None:
    if not bead_targets:
        log("[BEAD] no bead camera roles enabled")
        return

    plc = create_plc_client()
    log(f"[BEAD] PLC_CONNECTED ip={PLC_IP}")

    try:
        for cycle in range(1, NUM_BEAD_IMAGES + 1):
            log(f"[BEAD] WAIT_TRIGGER cycle={cycle}/{NUM_BEAD_IMAGES}")

            edge_ts = wait_plc_fresh_rising_edge(
                plc,
                BEAD_PLC_BYTE,
                BEAD_PLC_BIT,
                "BEAD",
            )

            if edge_ts is None or shutdown_event.is_set():
                break

            log(f"[BEAD] RELEASE cycle={cycle}")

            tasks = []

            for actor, role_name in bead_targets:
                ready = actor.is_ready()

                log(
                    f"[BEAD] EDGE_CHECK cycle={cycle} serial={actor.serial} "
                    f"camera_ready_at_edge={ready} queue_size={actor.q.qsize()}"
                )

                task = actor.submit(
                    role_name,
                    "bead",
                    cycle,
                    plc_edge_ts=edge_ts,
                )
                tasks.append(task)

            wait_all_tasks(tasks, "BEAD", cycle)

    finally:
        try:
            plc.disconnect()
        except Exception:
            pass

        log("[BEAD] PLC_DISCONNECTED")


# ============================================================
# SOFTWARE / FREE MODE FALLBACK
# ============================================================

def software_capture_controller(all_targets) -> None:
    if not all_targets:
        log("[SOFTWARE] no roles enabled")
        return

    total_cycles = max(NUM_FULL_IMAGES, NUM_BEAD_IMAGES)

    for cycle in range(1, total_cycles + 1):
        log(f"[SOFTWARE] START cycle={cycle}/{total_cycles}")

        tasks = []
        fake_edge_ts = time.perf_counter()

        for actor, role_name, group_name in all_targets:
            if group_name == "main" and cycle > NUM_FULL_IMAGES:
                continue

            if group_name == "bead" and cycle > NUM_BEAD_IMAGES:
                continue

            task = actor.submit(
                role_name,
                group_name,
                cycle,
                plc_edge_ts=fake_edge_ts,
            )
            tasks.append(task)

        wait_all_tasks(tasks, "SOFTWARE", cycle)


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    global running

    os.makedirs(SAVE_DIR, exist_ok=True)

    saver = threading.Thread(target=save_worker, daemon=True)
    saver.start()

    actors = []

    try:
        device_infos = system.device_infos

        enabled_serials = {
            serial
            for serial, cfg in CAMERA_CONFIGS.items()
            if cfg.get("enabled", True)
        }

        selected_device_infos = []

        for info in device_infos:
            serial = str(info["serial"])

            if serial in enabled_serials:
                selected_device_infos.append(info)
                log(f"[CAMERA_FOUND] serial={serial}")
            else:
                log(f"[CAMERA_SKIP] serial={serial}")

        if not selected_device_infos:
            raise RuntimeError("No enabled configured cameras found")

        devices = system.create_device(selected_device_infos)

        if not devices:
            raise RuntimeError("No Lucid cameras opened")

        log("=" * 80)
        log(f"[START] detected_cameras={len(devices)} mode={CAPTURE_MODE}")
        log(f"[START] main_trigger=DB{PLC_DB}.DBX{MAIN_PLC_BYTE}.{MAIN_PLC_BIT}")
        log(f"[START] bead_trigger=DB{PLC_DB}.DBX{BEAD_PLC_BYTE}.{BEAD_PLC_BIT}")
        log(f"[START] save_dir={SAVE_DIR}")
        log(f"[START] final_height={FINAL_HEIGHT} camera_height={CAMERA_HEIGHT}")
        log(
            f"[START] software_ffc={ENABLE_SOFTWARE_FFC} "
            f"target={GAIN_TARGET_MODE} gain_clip={GAIN_RANGE_MIN}-{GAIN_RANGE_MAX}"
        )
        log(
            f"[START] pixel_format={PIXEL_FORMAT} save_as_8bit={SAVE_AS_8BIT} "
            f"save_format={SAVE_IMAGE_FORMAT}"
        )
        log(
            f"[START] capture_build_mode={CAPTURE_BUILD_MODE} "
            f"time_capture_sec={TIME_CAPTURE_SEC}"
        )
        log("=" * 80)

        for cam in devices:
            serial = str(read_node(cam.nodemap, "DeviceSerialNumber", "UNKNOWN"))

            if serial not in CAMERA_CONFIGS:
                log(f"[CAMERA_SKIP] serial={serial} not configured")
                continue

            cfg = CAMERA_CONFIGS[serial]

            if not cfg.get("enabled", True):
                log(f"[CAMERA_SKIP] serial={serial} disabled")
                continue

            actor = CameraActor(cam)
            actor.start()
            actors.append(actor)

        if not actors:
            raise RuntimeError("No enabled configured cameras started")

        main_targets = build_role_targets(actors, "main")
        bead_targets = build_role_targets(actors, "bead")

        log("=" * 80)
        log("[ROLE_SUMMARY]")

        for actor, role_name in main_targets:
            log(f"[ROLE] group=MAIN serial={actor.serial} role={role_name}")

        for actor, role_name in bead_targets:
            log(f"[ROLE] group=BEAD serial={actor.serial} role={role_name}")

        log("=" * 80)

        if CAPTURE_MODE == "PLC_SOFTWARE":
            plc_threads = []

            if main_targets:
                t_main = threading.Thread(
                    target=main_plc_controller,
                    args=(main_targets,),
                    daemon=True,
                )
                t_main.start()
                plc_threads.append(t_main)

            if bead_targets:
                t_bead = threading.Thread(
                    target=bead_plc_controller,
                    args=(bead_targets,),
                    daemon=True,
                )
                t_bead.start()
                plc_threads.append(t_bead)

            try:
                for t in plc_threads:
                    while t.is_alive() and not shutdown_event.is_set():
                        t.join(timeout=0.2)
            except KeyboardInterrupt:
                request_shutdown("Ctrl+C while waiting for PLC threads")

        elif CAPTURE_MODE in ["SOFTWARE", "FREE"]:
            all_targets = []

            for actor in actors:
                for role in actor.cfg.get("roles", []):
                    if not role.get("enabled", True):
                        continue

                    all_targets.append((actor, role["name"], role["group"]))

            software_capture_controller(all_targets)

        else:
            raise RuntimeError(f"Unsupported CAPTURE_MODE: {CAPTURE_MODE}")

        log("[SAVE] waiting for raw + FFC corrected images to finish writing...")
        save_queue.join()
        log("[SAVE] all queued images saved")

    finally:
        for actor in actors:
            try:
                actor.stop()
            except Exception:
                pass

        running = False

        try:
            save_queue.put_nowait(None)
        except Exception:
            pass

        try:
            system.destroy_device()
        except Exception:
            pass

        log("[DONE] ALL_CAMERA_CAPTURE_COMPLETED")


if __name__ == "__main__":
    main()
