# lucid_plc_ffc_env_runner.py
# ============================================================
# UI runner for standalone_lucid_plc_software_ffc_raw_corrected.py
#
# Keep this file in:
#   src/camera/lucid_plc_ffc_env_runner.py
#
# It reads settings from environment variables supplied by the PyQt Auto tab,
# patches the standalone module globals, resets runtime queues/events, then
# calls main().
# ============================================================

import json
import os
import queue
import sys
from pathlib import Path

def env_str(name: str, default):
    value = os.environ.get(name)
    if value is None or str(value).strip() == "":
        return default
    return str(value).strip()


def env_int(name: str, default: int) -> int:
    try:
        return int(float(env_str(name, default)))
    except Exception:
        return int(default)


def env_float(name: str, default: float) -> float:
    try:
        return float(env_str(name, default))
    except Exception:
        return float(default)


def env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return bool(default)
    return str(value).strip().lower() in ("1", "true", "yes", "y", "on")


def main() -> None:
    here = Path(__file__).resolve().parent
    if str(here) not in sys.path:
        sys.path.insert(0, str(here))

    try:
        import standalone_lucid_plc_software_ffc_raw_corrected as cap
    except Exception as e:
        raise RuntimeError(
            "Could not import standalone_lucid_plc_software_ffc_raw_corrected.py. "
            "Keep this runner in the same folder as the standalone script. "
            f"Import error: {e}"
        )

    # --------------------------------------------------------
    # Capture mode / PLC
    # --------------------------------------------------------
    cap.CAPTURE_MODE = env_str("APOLLO_CAPTURE_MODE", getattr(cap, "CAPTURE_MODE", "PLC_SOFTWARE"))

    cap.PLC_IP = env_str("APOLLO_PLC_IP", getattr(cap, "PLC_IP", "192.168.10.1"))
    cap.PLC_RACK = env_int("APOLLO_PLC_RACK", getattr(cap, "PLC_RACK", 0))
    cap.PLC_SLOT = env_int("APOLLO_PLC_SLOT", getattr(cap, "PLC_SLOT", 1))
    cap.PLC_DB = env_int("APOLLO_PLC_DB", getattr(cap, "PLC_DB", 74))
    cap.MAIN_PLC_BYTE = env_int("APOLLO_MAIN_PLC_BYTE", getattr(cap, "MAIN_PLC_BYTE", 0))
    cap.MAIN_PLC_BIT = env_int("APOLLO_MAIN_PLC_BIT", getattr(cap, "MAIN_PLC_BIT", 3))
    cap.BEAD_PLC_BYTE = env_int("APOLLO_BEAD_PLC_BYTE", getattr(cap, "BEAD_PLC_BYTE", 86))
    cap.BEAD_PLC_BIT = env_int("APOLLO_BEAD_PLC_BIT", getattr(cap, "BEAD_PLC_BIT", 0))
    cap.PLC_POLL_DELAY_SEC = env_float("APOLLO_PLC_POLL_DELAY_SEC", getattr(cap, "PLC_POLL_DELAY_SEC", 0.005))

    # --------------------------------------------------------
    # Capture settings
    # --------------------------------------------------------
    cap.SAVE_DIR = env_str("APOLLO_FFC_SAVE_DIR", getattr(cap, "SAVE_DIR", str(here / "Auto_FFC_Capture")))
    cap.NUM_FULL_IMAGES = env_int("APOLLO_NUM_FULL_IMAGES", getattr(cap, "NUM_FULL_IMAGES", 1))
    cap.NUM_BEAD_IMAGES = env_int("APOLLO_NUM_BEAD_IMAGES", getattr(cap, "NUM_BEAD_IMAGES", 1))
    cap.CAMERA_HEIGHT = env_int("APOLLO_CAMERA_HEIGHT", getattr(cap, "CAMERA_HEIGHT", 14000))
    cap.FINAL_HEIGHT = env_int("APOLLO_FINAL_HEIGHT", getattr(cap, "FINAL_HEIGHT", 42000))
    cap.CAPTURE_BUILD_MODE = env_str(
        "APOLLO_CAPTURE_BUILD_MODE",
        getattr(cap, "CAPTURE_BUILD_MODE", "HEIGHT_BASED"),
    )

    cap.TIME_CAPTURE_SEC = env_float(
        "APOLLO_TIME_CAPTURE_SEC",
        getattr(cap, "TIME_CAPTURE_SEC", 5.0),
    )
    cap.PIXEL_FORMAT = env_str("APOLLO_PIXEL_FORMAT", getattr(cap, "PIXEL_FORMAT", "Mono16"))
    cap.NUM_STREAM_BUFFERS = env_int("APOLLO_NUM_STREAM_BUFFERS", getattr(cap, "NUM_STREAM_BUFFERS", 16))
    cap.BUFFER_TIMEOUT_MS = env_int("APOLLO_BUFFER_TIMEOUT_MS", getattr(cap, "BUFFER_TIMEOUT_MS", 30000))
    cap.PNG_COMPRESSION = env_int("APOLLO_PNG_COMPRESSION", getattr(cap, "PNG_COMPRESSION", 0))
    cap.SAVE_AS_8BIT = env_bool(
        "APOLLO_SAVE_AS_8BIT",
        getattr(cap, "SAVE_AS_8BIT", True),
    )
    cap.SAVE_IMAGE_FORMAT = env_str(
        "APOLLO_SAVE_IMAGE_FORMAT",
        getattr(cap, "SAVE_IMAGE_FORMAT", "png"),
    ).lower()
    cap.PACKET_SIZE = env_int("APOLLO_PACKET_SIZE", getattr(cap, "PACKET_SIZE", 9000))
    cap.PACKET_DELAY = env_int("APOLLO_PACKET_DELAY", getattr(cap, "PACKET_DELAY", 1000))

    # --------------------------------------------------------
    # FFC settings
    # --------------------------------------------------------
    cap.ENABLE_SOFTWARE_FFC = env_bool("APOLLO_ENABLE_SOFTWARE_FFC", getattr(cap, "ENABLE_SOFTWARE_FFC", True))
    cap.SAVE_RAW_IMAGES = env_bool("APOLLO_SAVE_RAW_IMAGES", getattr(cap, "SAVE_RAW_IMAGES", True))
    cap.SAVE_CORRECTED_IMAGES = env_bool("APOLLO_SAVE_CORRECTED_IMAGES", getattr(cap, "SAVE_CORRECTED_IMAGES", True))
    cap.SAVE_GAIN_NPY = env_bool("APOLLO_SAVE_GAIN_NPY", getattr(cap, "SAVE_GAIN_NPY", False))
    cap.GAIN_TARGET_MODE = env_str("APOLLO_GAIN_TARGET_MODE", getattr(cap, "GAIN_TARGET_MODE", "PERCENTILE_95"))
    cap.GAIN_RANGE_MIN = env_float("APOLLO_GAIN_RANGE_MIN", getattr(cap, "GAIN_RANGE_MIN", 1.0))
    cap.GAIN_RANGE_MAX = env_float("APOLLO_GAIN_RANGE_MAX", getattr(cap, "GAIN_RANGE_MAX", 15.99))
    cap.FFC_ROW_BLOCK = env_int("APOLLO_FFC_ROW_BLOCK", getattr(cap, "FFC_ROW_BLOCK", 512))

    # --------------------------------------------------------
    # Camera configs from UI table
    # --------------------------------------------------------
    camera_json = os.environ.get("APOLLO_CAMERA_CONFIGS_JSON", "").strip()
    if camera_json:
        try:
            configs = json.loads(camera_json)
            if isinstance(configs, dict) and configs:
                cap.CAMERA_CONFIGS = configs
        except Exception as e:
            raise RuntimeError(f"Invalid APOLLO_CAMERA_CONFIGS_JSON: {e}")

    # The standalone file creates save_queue at import time. Recreate it after
    # applying SAVE_QUEUE_SIZE/settings to avoid stale queue sizes on each start.
    cap.SAVE_QUEUE_SIZE = env_int("APOLLO_SAVE_QUEUE_SIZE", getattr(cap, "SAVE_QUEUE_SIZE", 4))
    cap.save_queue = queue.Queue(maxsize=cap.SAVE_QUEUE_SIZE)
    cap.running = True

    try:
        cap.shutdown_event.clear()
    except Exception:
        pass

    print("=" * 80, flush=True)
    print("[UI_RUNNER] Settings loaded from PyQt Auto tab", flush=True)
    print(f"[UI_RUNNER] CAPTURE_MODE={cap.CAPTURE_MODE}", flush=True)
    print(f"[UI_RUNNER] SAVE_DIR={cap.SAVE_DIR}", flush=True)
    print(f"[UI_RUNNER] NUM_FULL_IMAGES={cap.NUM_FULL_IMAGES}", flush=True)
    print(f"[UI_RUNNER] NUM_BEAD_IMAGES={cap.NUM_BEAD_IMAGES}", flush=True)
    print(f"[UI_RUNNER] FINAL_HEIGHT={cap.FINAL_HEIGHT} CAMERA_HEIGHT={cap.CAMERA_HEIGHT}", flush=True)
    print(f"[UI_RUNNER] PIXEL_FORMAT={cap.PIXEL_FORMAT}", flush=True)
    print(f"[UI_RUNNER] FFC={cap.ENABLE_SOFTWARE_FFC} RAW={cap.SAVE_RAW_IMAGES} CORRECTED={cap.SAVE_CORRECTED_IMAGES}", flush=True)
    print(
        f"[UI_RUNNER] SAVE_AS_8BIT={cap.SAVE_AS_8BIT} "
        f"SAVE_IMAGE_FORMAT={cap.SAVE_IMAGE_FORMAT}",
        flush=True,
    )
    print(
        f"[UI_RUNNER] CAPTURE_BUILD_MODE={cap.CAPTURE_BUILD_MODE} "
        f"TIME_CAPTURE_SEC={cap.TIME_CAPTURE_SEC}",
        flush=True,
    )
    print(f"[UI_RUNNER] CAMERA_CONFIGS={list(cap.CAMERA_CONFIGS.keys())}", flush=True)
    print("=" * 80, flush=True)

    cap.main()


if __name__ == "__main__":
    main()
