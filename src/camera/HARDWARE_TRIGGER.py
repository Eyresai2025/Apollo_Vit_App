# src/camera/HARDWARE_TRIGGER.py
# =========================================================
# Apollo Live Camera Manager - PLC SOFTWARE TRIGGER VERSION
#
# Final standalone logic merged into application style:
#   - Main group waits PLC DB74.DBX0.3
#   - Bead group waits PLC DB74.DBX86.0
#   - Camera trigger is Software trigger:
#       TriggerSelector = AcquisitionStart
#       TriggerSource   = Software
#       TriggerMode     = On
#   - Python executes TriggerSoftware after PLC rising edge
#   - Same physical camera can have multiple roles
#       Example: serial 250500042 -> innerwall(main) + bead(bead)
#   - One CameraActor per physical camera, so duplicate serial is never opened twice
# =========================================================

from arena_api.system import system
from arena_api.buffer import BufferFactory

import ctypes
import time
import threading
import queue
import concurrent.futures
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, List, Any, Tuple
PLC_IO_LOCK = threading.RLock()
import numpy as np

try:
    import snap7
    from snap7.util import get_bool
except Exception:
    snap7 = None
    get_bool = None


# =========================================================
# ENV LOADER
# =========================================================

def _project_root() -> Path:
    try:
        return Path(__file__).resolve().parents[2]
    except Exception:
        return Path.cwd()


def _load_env_file() -> Dict[str, str]:
    env_path = _project_root() / ".env"
    data: Dict[str, str] = {}

    try:
        if env_path.exists():
            with open(env_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    key, value = line.split("=", 1)
                    data[key.strip()] = value.strip().strip('"').strip("'")
    except Exception as e:
        print(f"[WARN] Could not load .env from {env_path}: {e}")

    return data


_ENV = _load_env_file()


def _env_str(key: str, default: str = "") -> str:
    value = _ENV.get(key, "")
    if value is None:
        return default
    value = str(value).strip()
    return value if value else default


def _env_int(key: str, default: int) -> int:
    try:
        value = _ENV.get(key, "")
        if value is None or str(value).strip() == "":
            return int(default)
        return int(float(str(value).strip()))
    except Exception:
        return int(default)


def _env_float(key: str, default: float) -> float:
    try:
        value = _ENV.get(key, "")
        if value is None or str(value).strip() == "":
            return float(default)
        return float(str(value).strip())
    except Exception:
        return float(default)


def _env_bool(key: str, default: bool = False) -> bool:
    value = _ENV.get(key, "")
    if value is None or str(value).strip() == "":
        return bool(default)
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def _side_key(side_name: str, field: str) -> str:
    return f"CAM_{side_name.upper()}_{field}"


def _side_or_global_str(side_name: str, field: str, global_key: str, default: str) -> str:
    value = _env_str(_side_key(side_name, field), "")
    return value if value != "" else _env_str(global_key, default)


def _side_or_global_int(side_name: str, field: str, global_key: str, default: int) -> int:
    value = _env_str(_side_key(side_name, field), "")
    if value != "":
        try:
            return int(float(value))
        except Exception:
            return int(default)
    return _env_int(global_key, default)


def _side_or_global_float(side_name: str, field: str, global_key: str, default: float) -> float:
    value = _env_str(_side_key(side_name, field), "")
    if value != "":
        try:
            return float(value)
        except Exception:
            return float(default)
    return _env_float(global_key, default)


def _side_or_global_bool(side_name: str, field: str, global_key: str, default: bool) -> bool:
    value = _env_str(_side_key(side_name, field), "")
    if value != "":
        return str(value).strip().lower() in ("1", "true", "yes", "on")
    return _env_bool(global_key, default)


# =========================================================
# GLOBAL CONFIG
# =========================================================

CAMERA_ROLE_ORDER = [
    ("sidewall1", "CAM_SIDEWALL1_SERIAL"),
    ("sidewall2", "CAM_SIDEWALL2_SERIAL"),
    ("innerwall", "CAM_INNERWALL_SERIAL"),
    ("tread", "CAM_TREAD_SERIAL"),
    ("bead", "CAM_BEAD_SERIAL"),
]

# accepted: plc_software / software / free
# hardware is intentionally not used for this final PLC-software flow.
TRIGGER_MODE = _env_str("CAM_TRIGGER_MODE", "plc_software").strip().lower()
if TRIGGER_MODE == "plc":
    TRIGGER_MODE = "plc_software"

TRIGGER_SELECTOR = _env_str("CAM_TRIGGER_SELECTOR", "AcquisitionStart")
TRIGGER_SOURCE = _env_str("CAM_TRIGGER_SOURCE", "Software")
TRIGGER_ACTIVATION = _env_str("CAM_TRIGGER_ACTIVATION", "RisingEdge")

PLC_IP = _env_str("PLC_IP", "192.168.10.1")
PLC_RACK = _env_int("PLC_RACK", 0)
PLC_SLOT = _env_int("PLC_SLOT", 1)

MAIN_TRIGGER_DB = _env_int("LIVE_MAIN_TRIGGER_DB", _env_int("CAPTURE_TRIGGER_DB", 74))
MAIN_TRIGGER_BYTE = _env_int("LIVE_MAIN_TRIGGER_BYTE", _env_int("NEW_SKU_CAPTURE_TRIGGER_BYTE", 0))
MAIN_TRIGGER_BIT = _env_int("LIVE_MAIN_TRIGGER_BIT", _env_int("NEW_SKU_CAPTURE_TRIGGER_BIT", 3))

BEAD_TRIGGER_DB = _env_int("LIVE_BEAD_TRIGGER_DB", 74)
BEAD_TRIGGER_BYTE = _env_int("LIVE_BEAD_TRIGGER_BYTE", 86)
BEAD_TRIGGER_BIT = _env_int("LIVE_BEAD_TRIGGER_BIT", 0)

# Final standalone uses 0.005 so bead edge is caught quickly.
PLC_POLL_DELAY_SEC = _env_float("LIVE_PLC_POLL_DELAY_SEC", _env_float("NEW_SKU_CAPTURE_POLL_DELAY_SEC", 0.005))
PLC_HIGH_LOG_EVERY_SEC = _env_float("LIVE_PLC_HIGH_LOG_EVERY_SEC", 1.0)

BUFFER_TIMEOUT_MS = _env_int("CAM_BUFFER_TIMEOUT_MS", 300000)
FLUSH_COUNT = _env_int("CAM_FLUSH_COUNT", 16)
PACKET_SIZE = _env_int("CAM_PACKET_SIZE", 9000)
PACKET_DELAY = _env_int("CAM_PACKET_DELAY", 1000)

AFTER_TRIGGER_DELAY_SEC = _env_float("CAM_AFTER_TRIGGER_DELAY_SEC", 0.02)
AFTER_ACQ_STOP_DELAY_SEC = _env_float("CAM_AFTER_ACQ_STOP_DELAY_SEC", 0.15)
AFTER_STOP_STREAM_DELAY_SEC = _env_float("CAM_AFTER_STOP_STREAM_DELAY_SEC", 0.20)
AFTER_START_STREAM_DELAY_SEC = _env_float("CAM_AFTER_START_STREAM_DELAY_SEC", 0.25)

PARALLEL = _env_bool("CAM_PARALLEL_CAPTURE", True)

# Shared inner/bead behavior from final standalone.
# True = bead role uses CAM_INNERWALL_SERIAL, so 250500042 opens only once.
SHARED_INNER_BEAD = _env_bool("CAM_SHARED_INNER_BEAD", True)
SHARED_INNER_BEAD_SERIAL = _env_str("CAM_INNERWALL_SERIAL", "250500042")

FAST_RESET_SHARED_CAMERA = _env_bool("CAM_FAST_RESET_SHARED_CAMERA", False)
FAST_RESET_SLEEP_SEC = _env_float("CAM_FAST_RESET_SLEEP_SEC", 0.05)
FAST_FLUSH_COUNT = _env_int("CAM_FAST_FLUSH_COUNT", 32)
FAST_FLUSH_TIMEOUT_MS = _env_int("CAM_FAST_FLUSH_TIMEOUT_MS", 2)
MAX_ALLOWED_BEAD_TRIGGER_DELAY_MS = _env_float("CAM_MAX_ALLOWED_BEAD_TRIGGER_DELAY_MS", 75.0)
VERBOSE_CONFIG_LOGS = _env_bool("CAM_VERBOSE_CONFIG_LOGS", False)


# =========================================================
# LOGGING
# =========================================================

def log(msg: str) -> None:
    from datetime import datetime
    ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    print(f"{ts} | {msg}", flush=True)


# =========================================================
# CAMERA ROLE CONFIG FROM .env
# =========================================================

def _side_group(side_name: str) -> str:
    default_group = "bead" if side_name == "bead" else "main"
    return _side_or_global_str(side_name, "GROUP", "CAM_GROUP", default_group).strip().lower()


def _role_enabled(side_name: str) -> bool:
    return _side_or_global_bool(side_name, "ENABLED", "CAM_ENABLED", True)


def _serial_for_side(side_name: str, serial_key: str) -> str:
    if side_name == "bead" and SHARED_INNER_BEAD:
        return _env_str("CAM_INNERWALL_SERIAL", "")
    return _env_str(serial_key, "")


def get_camera_role_config() -> List[Dict[str, Any]]:
    """
    One entry per logical role/side.
    If CAM_SHARED_INNER_BEAD=True, innerwall and bead will both have same serial.
    """
    configs: List[Dict[str, Any]] = []

    for side_name, serial_key in CAMERA_ROLE_ORDER:
        if not _role_enabled(side_name):
            continue

        serial = _serial_for_side(side_name, serial_key)
        if not serial:
            continue

        cfg = {
            "side": side_name,
            "serial": str(serial),
            "group": _side_group(side_name),

            "width": _side_or_global_int(side_name, "WIDTH", "CAM_WIDTH", 4096),
            "camera_height": _side_or_global_int(side_name, "CAMERA_HEIGHT", "CAM_CAMERA_HEIGHT", 14000),
            "final_height": _side_or_global_int(side_name, "FINAL_HEIGHT", "CAM_FINAL_HEIGHT", 42000),
            "pixel_format": _side_or_global_str(side_name, "PIXEL_FORMAT", "CAM_PIXEL_FORMAT", "Mono16"),
            "num_stream_buffers": _side_or_global_int(side_name, "STREAM_BUFFERS", "CAM_STREAM_BUFFERS", 16),

            "exposure_auto_limit_auto": _side_or_global_str(side_name, "EXPOSURE_AUTO_LIMIT_AUTO", "CAM_EXPOSURE_AUTO_LIMIT_AUTO", "Off"),
            "exposure_time": _side_or_global_float(side_name, "EXPOSURE_TIME", "CAM_EXPOSURE_TIME", 120.0),
            "gain": _side_or_global_float(side_name, "GAIN", "CAM_GAIN", 24.0),

            "acquisition_line_rate_enable": _side_or_global_bool(side_name, "ACQUISITION_LINE_RATE_ENABLE", "CAM_ACQUISITION_LINE_RATE_ENABLE", True),
            "acquisition_line_rate": _side_or_global_float(side_name, "ACQUISITION_LINE_RATE", "CAM_ACQUISITION_LINE_RATE", 8169.0),
            "acquisition_mode": _side_or_global_str(side_name, "ACQUISITION_MODE", "CAM_ACQUISITION_MODE", "Continuous"),
        }

        configs.append(cfg)

    return configs


def get_camera_to_side_map() -> Dict[str, str]:
    """
    Backward-compatible map. When one serial has multiple roles, this returns
    the first role for that serial. Use get_camera_roles_by_serial() for full map.
    """
    out: Dict[str, str] = {}
    for item in get_camera_role_config():
        out.setdefault(str(item["serial"]), item["side"])
    return out


def get_side_to_camera_map() -> Dict[str, str]:
    return {item["side"]: str(item["serial"]) for item in get_camera_role_config()}


def get_camera_roles_by_serial() -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for item in get_camera_role_config():
        grouped.setdefault(str(item["serial"]), []).append(item)
    return grouped


def get_physical_camera_config() -> List[Dict[str, Any]]:
    """
    One entry per physical camera serial.
    The first role in CAMERA_ROLE_ORDER decides camera node settings.
    This prevents duplicate opening of shared serial 250500042.
    """
    role_configs = get_camera_role_config()
    physical: Dict[str, Dict[str, Any]] = {}

    for role in role_configs:
        serial = str(role["serial"])
        if serial not in physical:
            physical[serial] = dict(role)
            physical[serial]["camera_name"] = role["side"]
            physical[serial]["roles"] = []
        physical[serial]["roles"].append({
            "name": role["side"],
            "group": role["group"],
            "enabled": True,
        })

    return list(physical.values())


CAMERA_ROLE_CONFIG = get_camera_role_config()
CAMERA_SERIALS = list({item["serial"] for item in CAMERA_ROLE_CONFIG})
NUM_CAMERAS = len(CAMERA_SERIALS)


# =========================================================
# PLC HELPERS
# =========================================================

def _plc_is_connected(plc_obj: Any) -> bool:
    try:
        if plc_obj is None:
            return False
        if hasattr(plc_obj, "get_connected"):
            return bool(plc_obj.get_connected())
        if hasattr(plc_obj, "get_cpu_state"):
            plc_obj.get_cpu_state()
            return True
    except Exception:
        return False
    return plc_obj is not None


def _extract_snap7_client(plc_interface: Any) -> Any:
    if plc_interface is None:
        return None

    for attr in ("client", "plc", "plc_client", "_client"):
        try:
            obj = getattr(plc_interface, attr, None)
            if obj is not None:
                return obj
        except Exception:
            pass

    return plc_interface


def _create_temp_plc_client() -> Any:
    if snap7 is None:
        raise RuntimeError("python-snap7 not installed. Install with: pip install python-snap7")
    plc = snap7.client.Client()
    plc.connect(PLC_IP, PLC_RACK, PLC_SLOT)
    if hasattr(plc, "get_connected") and not plc.get_connected():
        raise RuntimeError(f"PLC connection failed: {PLC_IP}")
    return plc

def _read_plc_bool(plc_client, db, byte, bit):
    """
    snap7.Client is not thread-safe.
    MAIN trigger and BEAD trigger may use same Test Mode PLC connection.
    So every db_read must be protected with one lock.
    """
    with PLC_IO_LOCK:
        data = plc_client.db_read(db, byte, 1)

    return get_bool(data, 0, bit)


def wait_plc_fresh_rising_edge(
    plc_interface: Any,
    db: int,
    byte: int,
    bit: int,
    label: str,
    stop_event: Optional[threading.Event] = None,
) -> Optional[float]:
    """
    Dedicated PLC edge wait for camera trigger.

    Do not reuse Test Mode PLC client here.
    GUI/Test Mode/Component Health may also touch that client.
    Live trigger waiting must use its own snap7 connection.
    """
    tag = f"DB{db}.DBX{byte}.{bit}"

    def _connect_client():
        plc = _create_temp_plc_client()
        log(f"[{label} PLC] dedicated PLC connected: {PLC_IP}")
        return plc

    def _disconnect_client(plc):
        try:
            if plc is not None:
                plc.disconnect()
        except Exception:
            pass

    plc_client = None

    try:
        plc_client = _connect_client()

        def safe_read_state():
            nonlocal plc_client
            last_err = None

            for attempt in range(3):
                try:
                    return _read_plc_bool(plc_client, db, byte, bit)
                except Exception as e:
                    last_err = e
                    log(
                        f"[{label} PLC][WARN] read failed {tag} "
                        f"attempt={attempt + 1}/3 | {e}"
                    )

                    _disconnect_client(plc_client)
                    plc_client = None
                    time.sleep(0.25)

                    try:
                        plc_client = _connect_client()
                    except Exception as ce:
                        last_err = ce
                        log(
                            f"[{label} PLC][WARN] reconnect failed "
                            f"attempt={attempt + 1}/3 | {ce}"
                        )
                        time.sleep(0.5)

            log(f"[{label} PLC][ERROR] read failed after retries: {last_err}")
            return None

        log(f"[{label}] PLC {tag} WAIT_LOW")

        last_log_time = 0.0

        while True:
            if stop_event is not None and stop_event.is_set():
                log(f"[{label}] PLC {tag} STOPPED_WHILE_WAIT_LOW")
                return None

            state = safe_read_state()

            if state is None:
                time.sleep(0.5)
                continue

            if not state:
                log(f"[{label}] PLC {tag} LOW_READY")
                break

            now = time.time()
            if now - last_log_time >= PLC_HIGH_LOG_EVERY_SEC:
                log(f"[{label}] PLC {tag} still HIGH, waiting reset LOW...")
                last_log_time = now

            time.sleep(PLC_POLL_DELAY_SEC)

        log(f"[{label}] PLC {tag} WAIT_HIGH")

        while True:
            if stop_event is not None and stop_event.is_set():
                log(f"[{label}] PLC {tag} STOPPED_WHILE_WAIT_HIGH")
                return None

            state = safe_read_state()

            if state is None:
                time.sleep(0.5)
                continue

            if state:
                stable = True

                for _ in range(2):
                    time.sleep(0.02)
                    state2 = safe_read_state()

                    if state2 is None or not state2:
                        stable = False
                        break

                if not stable:
                    log(f"[{label}] PLC {tag} HIGH_GLITCH_IGNORED")
                    continue

                edge_ts = time.perf_counter()
                log(f"[{label}] PLC {tag} HIGH_EDGE_STABLE")
                return edge_ts

            time.sleep(PLC_POLL_DELAY_SEC)

    finally:
        _disconnect_client(plc_client)
        log(f"[{label} PLC] dedicated PLC disconnected")


# =========================================================
# LINE SCAN PHYSICAL CAMERA
# =========================================================

class LineScanCamera:
    def __init__(
        self,
        serial_number: str,
        camera_name: str,
        roles: List[Dict[str, Any]],
        width: int = 4096,
        camera_height: int = 14000,
        final_height: int = 42000,
        pixel_format: str = "Mono16",
        num_stream_buffers: int = 16,
        exposure_auto_limit_auto: str = "Off",
        exposure_time: float = 120.0,
        gain: float = 24.0,
        acquisition_line_rate_enable: bool = True,
        acquisition_line_rate: float = 8169.0,
        acquisition_mode: str = "Continuous",
    ):
        self.serial_number = str(serial_number)
        self.camera_name = camera_name
        self.roles = roles

        self.width = int(width)
        self.camera_height = int(camera_height)
        self.final_height = int(final_height)
        self.pixel_format = pixel_format
        self.num_stream_buffers = int(num_stream_buffers)

        self.exposure_auto_limit_auto = exposure_auto_limit_auto
        self.exposure_time = float(exposure_time)
        self.gain = float(gain)

        self.acquisition_line_rate_enable = bool(acquisition_line_rate_enable)
        self.acquisition_line_rate = float(acquisition_line_rate) if acquisition_line_rate not in (None, "") else 0.0
        self.acquisition_mode = acquisition_mode

        self.device = None
        self.nodemap = None
        self.is_streaming = False
        self.is_connected = False

        self._stop_event = threading.Event()
        self._capture_lock = threading.Lock()

    # -----------------------------------------------------
    # NODE HELPERS
    # -----------------------------------------------------
    def _set_node(self, name: str, value: Any, verbose: Optional[bool] = None) -> bool:
        if verbose is None:
            verbose = VERBOSE_CONFIG_LOGS
        try:
            if self.nodemap is None:
                if verbose:
                    log(f"  [{self.serial_number}] {name}: nodemap not ready")
                return False
            node = self.nodemap.get_node(name)
            if node and node.is_writable:
                node.value = value
                if verbose:
                    log(f"  [{self.serial_number}] {name}: {node.value}")
                return True
            if verbose:
                log(f"  [{self.serial_number}] {name}: not writable / not found")
            return False
        except Exception as e:
            log(f"  [{self.serial_number}] {name} not set: {e}")
            return False

    def _get_node_value(self, name: str, default: Any = None) -> Any:
        try:
            if self.nodemap is None:
                return default
            node = self.nodemap.get_node(name)
            if node and node.is_readable:
                return node.value
            if node:
                return node.value
        except Exception:
            pass
        return default

    def _execute_node(self, name: str) -> bool:
        try:
            if self.nodemap is None:
                return False
            node = self.nodemap.get_node(name)
            if node:
                node.execute()
                return True
        except Exception as e:
            log(f"  [{self.serial_number}] EXEC_FAIL {name}: {e}")
        return False

    # -----------------------------------------------------
    # BUFFER HELPERS
    # -----------------------------------------------------
    def _convert_buffer(self, buffer) -> np.ndarray:
        copied = BufferFactory.copy(buffer)
        try:
            width = copied.width
            height = copied.height
            total_bytes = len(copied.data)
            c_arr = (ctypes.c_ubyte * total_bytes).from_address(ctypes.addressof(copied.pbytes))
            np_arr = np.ctypeslib.as_array(c_arr)
            bytes_per_pixel = total_bytes // (width * height)

            if bytes_per_pixel == 2:
                img = np_arr.view(np.uint16).reshape(height, width)
            else:
                img = np_arr.reshape(height, width)

            return img.copy()
        finally:
            BufferFactory.destroy(copied)

    def flush_buffers(self, max_count: int = FLUSH_COUNT, timeout_ms: int = 100, log_it: bool = True) -> int:
        if not self.is_streaming or self.device is None:
            return 0

        flushed = 0
        for _ in range(max_count):
            try:
                buf = self.device.get_buffer(timeout=timeout_ms)
                self.device.requeue_buffer(buf)
                flushed += 1
            except Exception:
                break

        if log_it:
            log(f"[{self.camera_name}/{self.serial_number}] FLUSH buffers={flushed}")
        return flushed

    def _get_buffer_interruptible(self, role_tag: str, timeout_ms: int = 500):
        while not self._stop_event.is_set():
            try:
                return self.device.get_buffer(timeout=timeout_ms)
            except Exception:
                continue
        raise RuntimeError(f"[{role_tag}] stop requested while waiting for buffer")

    # -----------------------------------------------------
    # CONNECTION
    # -----------------------------------------------------
    def connect_only(self) -> None:
        if self.is_connected and self.device is not None and self.nodemap is not None:
            log(f"[{self.serial_number}] Already connected")
            return

        target_info = None
        for info in system.device_infos:
            if str(info.get("serial")) == str(self.serial_number):
                target_info = info
                break

        if target_info is None:
            raise RuntimeError(f"Camera serial {self.serial_number} not found in Arena device list")

        devices = system.create_device([target_info])
        self.device = devices[0]
        self.nodemap = self.device.nodemap

        actual_serial = self._get_node_value("DeviceSerialNumber", self.serial_number)
        self.is_streaming = False
        self.is_connected = True

        role_txt = ", ".join([f"{r['name']}:{r['group']}" for r in self.roles])
        log("--------------------------------------------------")
        log(f"[{self.serial_number}] Camera connected ONLY")
        log(f"Camera name : {self.camera_name}")
        log(f"Roles       : {role_txt}")
        log(f"Actual serial: {actual_serial}")
        log("No camera configuration applied in Test Mode.")
        log("--------------------------------------------------")

    # -----------------------------------------------------
    # CONFIGURATION
    # -----------------------------------------------------
    def _configure_stream_nodes(self) -> None:
        try:
            tl = self.device.tl_stream_nodemap
        except Exception:
            tl = None

        if tl is None:
            return

        def set_tl(name: str, value: Any):
            try:
                node = tl.get_node(name)
                if node and node.is_writable:
                    node.value = value
                    if VERBOSE_CONFIG_LOGS:
                        log(f"  [{self.serial_number}] TL {name}: {node.value}")
            except Exception as e:
                log(f"  [{self.serial_number}] TL {name} not set: {e}")

        set_tl("StreamAutoNegotiatePacketSize", True)
        set_tl("StreamPacketResendEnable", True)
        set_tl("StreamBufferHandlingMode", "OldestFirst")

    def _configure_trigger(self) -> None:
        mode = TRIGGER_MODE

        if mode in ("software", "plc_software"):
            self._set_node("TriggerMode", "Off")
            self._set_node("TriggerSelector", TRIGGER_SELECTOR or "AcquisitionStart")
            self._set_node("TriggerSource", TRIGGER_SOURCE or "Software")
            self._set_node("TriggerActivation", TRIGGER_ACTIVATION)
            self._set_node("TriggerMode", "On")
            return

        if mode == "free":
            self._set_node("TriggerMode", "Off")
            return

        raise RuntimeError(
            f"Invalid CAM_TRIGGER_MODE={TRIGGER_MODE}. This final app file supports plc_software/software/free."
        )

    def configure_for_live(self) -> None:
        if not self.is_connected or self.device is None or self.nodemap is None:
            self.connect_only()

        if self.is_streaming:
            self.stop_stream()

        role_txt = ", ".join([f"{r['name']}:{r['group']}" for r in self.roles])
        log("--------------------------------------------------")
        log(f"[{self.serial_number}] Applying LIVE camera configuration")
        log(f"Camera name : {self.camera_name}")
        log(f"Roles       : {role_txt}")
        log("--------------------------------------------------")

        self._configure_stream_nodes()

        self._set_node("TriggerMode", "Off")

        self._set_node("Width", self.width)
        self._set_node("Height", self.camera_height)
        self._set_node("PixelFormat", self.pixel_format)
        self._set_node("AcquisitionMode", self.acquisition_mode)

        if self.acquisition_line_rate_enable:
            ok_enable = self._set_node("AcquisitionLineRateEnable", True)
            ok_rate = self._set_node("AcquisitionLineRate", self.acquisition_line_rate)
            if ok_enable or ok_rate:
                safe_exposure = min(
                    self.exposure_time,
                    0.90 * (1_000_000.0 / max(float(self.acquisition_line_rate), 1.0)),
                )
            else:
                safe_exposure = self.exposure_time
        else:
            log(f"  [{self.serial_number}] AcquisitionLineRate skipped")
            safe_exposure = self.exposure_time

        self._set_node("ExposureAutoLimitAuto", self.exposure_auto_limit_auto)
        time.sleep(0.02)
        self._set_node("ExposureTime", safe_exposure)
        self._set_node("Gain", self.gain)

        self._set_node("GevSCPSPacketSize", getattr(self, "packet_size", PACKET_SIZE))
        self._set_node("GevSCPD", getattr(self, "packet_delay", PACKET_DELAY))

        self._configure_trigger()

        log(f"[{self.serial_number}] FINAL SETTINGS")
        for node_name in [
            "DeviceSerialNumber", "Width", "Height", "PixelFormat", "AcquisitionMode",
            "AcquisitionLineRateEnable", "AcquisitionLineRate", "ExposureTime", "Gain",
            "TriggerSelector", "TriggerSource", "TriggerActivation", "TriggerMode",
            "LineStatus", "GevSCPSPacketSize", "GevSCPD",
        ]:
            log(f"  {node_name}: {self._get_node_value(node_name, '-')}")

    # -----------------------------------------------------
    # STREAM CONTROL
    # -----------------------------------------------------
    def start_stream(self) -> None:
        if not self.is_connected or self.device is None:
            raise RuntimeError(f"[{self.serial_number}] Camera not connected")

        if self.is_streaming:
            return

        log(f"[{self.serial_number}] Starting stream with {self.num_stream_buffers} buffers")
        self.device.start_stream(self.num_stream_buffers)
        self.is_streaming = True
        self._stop_event.clear()

    def stop_stream(self) -> None:
        if self.device is not None and self.is_streaming:
            try:
                self._stop_event.set()
                self.device.stop_stream()
                log(f"[{self.serial_number}] Stream stopped")
            except Exception as e:
                log(f"[WARN] [{self.serial_number}] Error stopping stream: {e}")
            finally:
                self.is_streaming = False

    def reset_acquisition_for_next_cycle(self, role_name: str) -> None:
        if TRIGGER_MODE not in ("software", "plc_software"):
            return

        role_tag = role_name.upper()
        reset_start = time.perf_counter()

        self._execute_node("AcquisitionStop")

        # Optional fast reset only for shared inner/bead camera.
        if FAST_RESET_SHARED_CAMERA and self.serial_number == SHARED_INNER_BEAD_SERIAL:
            time.sleep(FAST_RESET_SLEEP_SEC)

            flushed = 0
            empty_hits = 0
            max_reset_ms = 1200.0

            while True:
                elapsed_ms = (time.perf_counter() - reset_start) * 1000.0
                if elapsed_ms > max_reset_ms:
                    break

                try:
                    buf = self.device.get_buffer(timeout=FAST_FLUSH_TIMEOUT_MS)
                    self.device.requeue_buffer(buf)
                    flushed += 1
                    empty_hits = 0
                except Exception:
                    empty_hits += 1
                    if empty_hits >= 3:
                        break

                if flushed >= FAST_FLUSH_COUNT:
                    break

            reset_ms = (time.perf_counter() - reset_start) * 1000.0
            log(
                f"[{role_tag}] CAMERA_READY serial={self.serial_number} "
                f"fast_reset_ms={reset_ms:.1f} flushed={flushed}"
            )
            return

        time.sleep(AFTER_ACQ_STOP_DELAY_SEC)

        try:
            self.stop_stream()
        except Exception as e:
            log(f"[{role_tag}] stop_stream warning: {e}")

        time.sleep(AFTER_STOP_STREAM_DELAY_SEC)

        try:
            self.start_stream()
        except Exception as e:
            log(f"[{role_tag}] start_stream warning: {e}")

        time.sleep(AFTER_START_STREAM_DELAY_SEC)

        flushed = self.flush_buffers(log_it=False)
        reset_ms = (time.perf_counter() - reset_start) * 1000.0
        log(
            f"[{role_tag}] CAMERA_READY serial={self.serial_number} "
            f"normal_reset_ms={reset_ms:.1f} flushed={flushed}"
        )

    # -----------------------------------------------------
    # CAPTURE
    # -----------------------------------------------------
    def capture_role_image(self, task: "CaptureTask") -> np.ndarray:
        if not self.is_streaming:
            raise RuntimeError(f"[{self.serial_number}] Stream not running")

        with self._capture_lock:
            role_name = task.role_name
            role_tag = role_name.upper()
            image_index = task.image_index

            if TRIGGER_MODE in ("software", "plc_software"):
                if role_name == "bead":
                    pre_flushed = self.flush_buffers(max_count=8, timeout_ms=2, log_it=False)
                    if pre_flushed > 0:
                        log(f"[BEAD] PRE_TRIGGER_FLUSH serial={self.serial_number} flushed={pre_flushed}")

                trigger_before = time.perf_counter()
                delay_from_plc_ms = (trigger_before - task.plc_edge_ts) * 1000.0

                late_msg = ""
                if role_name == "bead" and delay_from_plc_ms > MAX_ALLOWED_BEAD_TRIGGER_DELAY_MS:
                    late_msg = " LATE_TRIGGER"

                log(
                    f"[{role_tag}] TRIGGER_SOFTWARE serial={self.serial_number} "
                    f"img={image_index} plc_to_trigger_ms={delay_from_plc_ms:.1f}{late_msg}"
                )

                self._execute_node("TriggerSoftware")
                time.sleep(AFTER_TRIGGER_DELAY_SEC)

            elif TRIGGER_MODE == "free":
                log(f"[{role_tag}] FREE_CAPTURE serial={self.serial_number} img={image_index}")

            else:
                raise RuntimeError(f"Unsupported CAM_TRIGGER_MODE for this file: {TRIGGER_MODE}")

            full_img = np.zeros((self.final_height, self.width), dtype=np.uint16)
            current_row = 0
            chunk_id = 0
            expected_chunks = int(np.ceil(self.final_height / max(self.camera_height, 1)))
            start_time = time.perf_counter()

            try:
                while current_row < self.final_height:
                    if self._stop_event.is_set():
                        raise RuntimeError(f"[{role_tag}] stop requested during capture")

                    buffer = self._get_buffer_interruptible(role_tag, timeout_ms=500)
                    try:
                        frame = self._convert_buffer(buffer)
                    finally:
                        self.device.requeue_buffer(buffer)

                    if frame.ndim != 2:
                        raise RuntimeError(f"Unexpected frame shape: {frame.shape}")

                    h, w = frame.shape
                    if w != self.width:
                        log(f"[{role_tag}] WIDTH_WARNING got={w} expected={self.width}")

                    copy_h = min(h, self.final_height - current_row)
                    full_img[current_row:current_row + copy_h, :] = frame[:copy_h, :]
                    current_row += copy_h
                    chunk_id += 1

                    log(
                        f"[{role_tag}] CHUNK {chunk_id}/{expected_chunks} "
                        f"rows={current_row}/{self.final_height}"
                    )

                elapsed = time.perf_counter() - start_time
                log(
                    f"[{role_tag}] STITCH_DONE serial={self.serial_number} "
                    f"img={image_index} rows={current_row}/{self.final_height} time={elapsed:.2f}s"
                )
                return full_img

            finally:
                self.reset_acquisition_for_next_cycle(role_name)

    # -----------------------------------------------------
    # CLOSE
    # -----------------------------------------------------
    def stop_and_close(self) -> None:
        log(f"[{self.serial_number}] Closing camera")
        self.stop_stream()
        self.is_connected = False
        self.device = None
        self.nodemap = None


# =========================================================
# CAMERA ACTOR - one actor per physical camera
# =========================================================

@dataclass
class CaptureTask:
    role_name: str
    group: str
    image_index: int
    plc_edge_ts: float
    submit_ts: float
    done_event: threading.Event
    error: List[str]
    result: List[Optional[np.ndarray]]


class CameraActor:
    def __init__(self, camera: LineScanCamera):
        self.camera = camera
        self.serial = camera.serial_number
        self.roles = camera.roles

        self.q: "queue.Queue[Optional[CaptureTask]]" = queue.Queue()
        self.thread: Optional[threading.Thread] = None

        self.ready_event = threading.Event()
        self.error: Optional[Exception] = None

        self.state_lock = threading.Lock()
        self.state = "STARTING"

    def set_state(self, state: str) -> None:
        with self.state_lock:
            self.state = state

    def is_ready(self) -> bool:
        with self.state_lock:
            return self.state == "READY" and self.q.empty()

    def start(self) -> None:
        if self.thread is not None and self.thread.is_alive():
            return

        self.ready_event.clear()
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
            result=[],
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

        if self.thread is not None:
            self.thread.join(timeout=2.0)

        if self.thread is not None and self.thread.is_alive():
            log(f"[STOP] camera actor still alive serial={self.serial}")

    def _run(self) -> None:
        try:
            self.camera.configure_for_live()
            self.camera.start_stream()
            self.camera.flush_buffers(log_it=False)

            self.set_state("READY")
            log(f"[READY] serial={self.serial} camera_ready=True")
            self.ready_event.set()

            while True:
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

                    img = self.camera.capture_role_image(task)
                    task.result.append(img)

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

def _profile_bool(value, default=False) -> bool:
    if isinstance(value, bool):
        return value

    if value is None or str(value).strip() == "":
        return bool(default)

    return str(value).strip().lower() in ("1", "true", "yes", "on", "y")
# =========================================================
# MULTI-CAMERA MANAGER
# =========================================================

class MultiCameraManager:
    """
    Test Mode:
        manager.connect_all()
        manager.set_plc_interface(plc_client_or_wrapper)

    Live Mode:
        manager.start_all_streams()
        manager.capture_all()

    Returns images by side/role name:
        {
          "sidewall1": img,
          "sidewall2": img,
          "tread": img,
          "innerwall": img,
          "bead": img,
        }
    """

    def __init__(self, plc_interface: Any = None):
        self.plc_interface = plc_interface
        self._streams_started = False
        self._stop_event = threading.Event()
        self._capture_index = 0

        self.role_config = get_camera_role_config()
        self.physical_config = get_physical_camera_config()

        self.camera_to_side = get_camera_to_side_map()
        self.side_to_camera = get_side_to_camera_map()
        self.camera_roles_by_serial = get_camera_roles_by_serial()

        if not self.role_config:
            raise RuntimeError(
                "No camera serials configured in .env. Set CAM_SIDEWALL1_SERIAL, "
                "CAM_SIDEWALL2_SERIAL, CAM_INNERWALL_SERIAL, CAM_TREAD_SERIAL, CAM_BEAD_SERIAL."
            )

        self.cameras: List[LineScanCamera] = []
        self.actors: List[CameraActor] = []

        for item in self.physical_config:
            cam = LineScanCamera(
                serial_number=item["serial"],
                camera_name=item.get("camera_name", item["serial"]),
                roles=item.get("roles", []),
                width=item["width"],
                camera_height=item["camera_height"],
                final_height=item["final_height"],
                pixel_format=item["pixel_format"],
                num_stream_buffers=item["num_stream_buffers"],
                exposure_auto_limit_auto=item["exposure_auto_limit_auto"],
                exposure_time=item["exposure_time"],
                gain=item["gain"],
                acquisition_line_rate_enable=item["acquisition_line_rate_enable"],
                acquisition_line_rate=item["acquisition_line_rate"],
                acquisition_mode=item["acquisition_mode"],
            )
            self.cameras.append(cam)

    def set_plc_interface(self, plc_interface: Any) -> None:
        self.plc_interface = plc_interface

    def apply_camera_profile(self, profile: Dict[str, Any]) -> None:
        """
        Apply SKU-wise camera profile to already-created LineScanCamera objects.

        Important:
        - Test Mode still only connects cameras.
        - Live Mode calls this before start_all_streams().
        - For current testing, serials should match .env serials.
        """

        if not isinstance(profile, dict):
            raise ValueError("camera profile must be a dict")

        sku_name = profile.get("sku_name", "-")
        cameras_cfg = profile.get("cameras", {}) or {}

        if not cameras_cfg:
            raise ValueError(f"No cameras found in camera profile for SKU={sku_name}")

        log("=" * 60)
        log(f"[CAMERA PROFILE] Applying SKU camera profile | SKU={sku_name}")
        log("=" * 60)

        # Refresh side/serial maps from profile
        for side_name, cfg in cameras_cfg.items():
            if not isinstance(cfg, dict):
                continue

            serial = str(cfg.get("serial", "")).strip()
            side_name = str(side_name).strip().lower()

            if serial:
                self.side_to_camera[side_name] = serial
                self.camera_to_side.setdefault(serial, side_name)

        for cam in self.cameras:
            selected_cfg = None
            selected_side = None

            # Match using logical role name: sidewall1, sidewall2, tread, innerwall, bead
            for role in getattr(cam, "roles", []):
                role_name = str(role.get("name", "")).strip().lower()

                cfg = cameras_cfg.get(role_name)

                if not isinstance(cfg, dict):
                    continue

                profile_serial = str(cfg.get("serial", "")).strip()

                if profile_serial and profile_serial != str(cam.serial_number):
                    log(
                        f"[CAMERA PROFILE][WARN] serial mismatch for role={role_name} | "
                        f"profile_serial={profile_serial} | connected_serial={cam.serial_number}. "
                        f"Using connected camera object for this test."
                    )

                role["enabled"] = _profile_bool(cfg.get("enabled", True), True)
                role["group"] = str(cfg.get("group", role.get("group", "main"))).strip().lower()

                if selected_cfg is None and role["enabled"]:
                    selected_cfg = cfg
                    selected_side = role_name

            if selected_cfg is None:
                log(f"[CAMERA PROFILE][WARN] No enabled profile role matched serial={cam.serial_number}")
                continue

            cam.width = int(selected_cfg.get("width", cam.width))

            # Device Page may save "height"; live camera code uses "camera_height"
            cam.camera_height = int(
                selected_cfg.get(
                    "camera_height",
                    selected_cfg.get("height", cam.camera_height),
                )
            )

            cam.final_height = int(selected_cfg.get("final_height", cam.final_height))
            cam.pixel_format = str(selected_cfg.get("pixel_format", cam.pixel_format))
            cam.num_stream_buffers = int(
                selected_cfg.get("num_stream_buffers", cam.num_stream_buffers)
            )

            cam.exposure_auto_limit_auto = str(
                selected_cfg.get("exposure_auto_limit_auto", cam.exposure_auto_limit_auto)
            )
            cam.exposure_time = float(selected_cfg.get("exposure_time", cam.exposure_time))
            cam.gain = float(selected_cfg.get("gain", cam.gain))

            cam.acquisition_line_rate_enable = _profile_bool(
                selected_cfg.get(
                    "acquisition_line_rate_enable",
                    cam.acquisition_line_rate_enable,
                ),
                cam.acquisition_line_rate_enable,
            )

            cam.acquisition_line_rate = float(
                selected_cfg.get("acquisition_line_rate", cam.acquisition_line_rate) or 0.0
            )

            cam.acquisition_mode = str(
                selected_cfg.get("acquisition_mode", cam.acquisition_mode)
            )

            # Optional per-profile packet settings
            cam.packet_size = int(selected_cfg.get("packet_size", PACKET_SIZE))
            cam.packet_delay = int(selected_cfg.get("packet_delay", PACKET_DELAY))

            log(
                f"[CAMERA PROFILE] Applied | side={selected_side} | serial={cam.serial_number} | "
                f"width={cam.width} | height={cam.camera_height} | final_height={cam.final_height} | "
                f"pixel={cam.pixel_format} | exposure={cam.exposure_time} | gain={cam.gain} | "
                f"line_rate_enable={cam.acquisition_line_rate_enable} | "
                f"line_rate={cam.acquisition_line_rate} | packet={cam.packet_size}/{cam.packet_delay}"
            )

        log("[CAMERA PROFILE] Apply completed")
    def connect_all(self, fail_fast: bool = False) -> bool:
        log("=" * 60)
        log(f"Connecting {len(self.cameras)} unique Lucid camera(s)")
        log(f"Trigger Mode: {TRIGGER_MODE.upper()}")
        log(f"Shared inner/bead: {SHARED_INNER_BEAD} | serial={SHARED_INNER_BEAD_SERIAL}")
        log("Camera Role Mapping:")
        for serial, roles in self.camera_roles_by_serial.items():
            role_txt = ", ".join([f"{r['side']}:{r['group']}" for r in roles])
            log(f"  {serial} -> {role_txt}")
        log("=" * 60)

        for cam in self.cameras:
            try:
                cam.connect_only()
            except Exception as e:
                cam.is_connected = False
                cam.device = None
                cam.nodemap = None
                log(f"[CAMERA][ERROR] serial={cam.serial_number} failed: {e}")
                if fail_fast:
                    raise

        connected = [cam.serial_number for cam in self.cameras if cam.is_connected]
        missing = [cam.serial_number for cam in self.cameras if not cam.is_connected]

        log(f"[CAMERA] Connected cameras: {connected}")
        log(f"[CAMERA] Missing/failed cameras: {missing}")

        if not connected:
            raise RuntimeError("No configured Lucid cameras connected")

        return len(missing) == 0

    def start_all_streams(self) -> bool:
        log("=" * 60)
        log("Configuring and starting all camera streams for LIVE")
        log(f"Trigger Mode: {TRIGGER_MODE.upper()}")
        log(f"Main PLC tag: DB{MAIN_TRIGGER_DB}.DBX{MAIN_TRIGGER_BYTE}.{MAIN_TRIGGER_BIT}")
        log(f"Bead PLC tag: DB{BEAD_TRIGGER_DB}.DBX{BEAD_TRIGGER_BYTE}.{BEAD_TRIGGER_BIT}")
        log(f"PLC poll delay: {PLC_POLL_DELAY_SEC}s")
        log("=" * 60)

        self._stop_event.clear()
        self.actors = []
        started = []
        failed = []

        for cam in self.cameras:
            try:
                if not cam.is_connected:
                    cam.connect_only()
                actor = CameraActor(cam)
                actor.start()
                self.actors.append(actor)
                started.append(cam.serial_number)
            except Exception as e:
                failed.append(cam.serial_number)
                log(f"[CAMERA][ERROR] live configure/start failed | serial={cam.serial_number} | {e}")
                traceback.print_exc()

        self._streams_started = len(started) > 0
        log(f"[CAMERA] Streams started: {started}")
        log(f"[CAMERA] Streams failed : {failed}")

        if not started:
            raise RuntimeError("No camera streams started")

        return len(failed) == 0

    def stop_all_streams(self) -> None:
        log("=" * 60)
        log("Stopping all camera streams")
        log("=" * 60)
        self._stop_event.set()

        for actor in self.actors:
            try:
                actor.stop()
            except Exception:
                pass

        self.actors = []

        for cam in self.cameras:
            cam.stop_stream()

        self._streams_started = False
        log("All camera streams stopped")

    def _build_role_targets(
        self,
        group_name: str,
        sides_to_capture: Optional[List[str]] = None,
    ) -> List[Tuple[CameraActor, str]]:
        targets: List[Tuple[CameraActor, str]] = []

        active_sides = set(sides_to_capture or [])

        for actor in self.actors:
            for role in actor.roles:
                if not role.get("enabled", True):
                    continue

                role_name = role.get("name")
                role_group = role.get("group")

                if role_group != group_name:
                    continue

                if active_sides and role_name not in active_sides:
                    continue

                targets.append((actor, role_name))

        return targets

    def _wait_all_tasks(self, tasks: List[CaptureTask], label: str, cycle: int) -> Dict[str, Optional[np.ndarray]]:
        results: Dict[str, Optional[np.ndarray]] = {}

        for task in tasks:
            while not task.done_event.is_set():
                if self._stop_event.is_set():
                    break
                task.done_event.wait(timeout=0.1)

        errors = []
        for task in tasks:
            if task.error:
                results[task.role_name] = None
                errors.append(f"role={task.role_name}, image={task.image_index}, error={task.error[0]}")
            else:
                results[task.role_name] = task.result[0] if task.result else None

        if errors:
            log(f"[{label}] CYCLE_DONE cycle={cycle} status=ERROR")
            for e in errors:
                log(f"[{label}] {e}")
        else:
            log(f"[{label}] CYCLE_DONE cycle={cycle} status=OK")

        return results

    def _capture_group_after_edge(
        self,
        group_name: str,
        targets: List[Tuple[CameraActor, str]],
        plc_edge_ts: float,
        cycle: int,
    ) -> Dict[str, Optional[np.ndarray]]:
        if not targets:
            return {}

        log(f"[{group_name.upper()}] RELEASE cycle={cycle}")

        tasks: List[CaptureTask] = []

        for actor, role_name in targets:
            if group_name == "bead":
                ready = actor.is_ready()
                log(
                    f"[BEAD] EDGE_CHECK cycle={cycle} serial={actor.serial} "
                    f"camera_ready_at_edge={ready} queue_size={actor.q.qsize()}"
                )

            task = actor.submit(
                role_name=role_name,
                group=group_name,
                image_index=cycle,
                plc_edge_ts=plc_edge_ts,
            )
            tasks.append(task)

        return self._wait_all_tasks(tasks, group_name.upper(), cycle)

    def _wait_then_capture_group(
        self,
        group_name: str,
        targets: List[Tuple[CameraActor, str]],
        db: int,
        byte: int,
        bit: int,
        cycle: int,
    ) -> Dict[str, Optional[np.ndarray]]:
        if not targets:
            return {}

        log(f"[{group_name.upper()}] WAIT_TRIGGER cycle={cycle}")

        edge_ts = wait_plc_fresh_rising_edge(
            plc_interface=self.plc_interface,
            db=db,
            byte=byte,
            bit=bit,
            label=group_name.upper(),
            stop_event=self._stop_event,
        )

        if edge_ts is None:
            return {role_name: None for _, role_name in targets}

        return self._capture_group_after_edge(group_name, targets, edge_ts, cycle)

    def _software_capture_once(
        self,
        cycle: int,
        sides_to_capture: Optional[List[str]] = None,
    ) -> Dict[str, Optional[np.ndarray]]:
        active_sides = set(sides_to_capture or [])

        all_targets: List[Tuple[CameraActor, str, str]] = []

        for actor in self.actors:
            for role in actor.roles:
                if not role.get("enabled", True):
                    continue

                role_name = role.get("name")

                if active_sides and role_name not in active_sides:
                    continue

                all_targets.append((actor, role_name, role["group"]))

        fake_edge_ts = time.perf_counter()
        tasks: List[CaptureTask] = []

        for actor, role_name, group_name in all_targets:
            tasks.append(
                actor.submit(
                    role_name,
                    group_name,
                    cycle,
                    plc_edge_ts=fake_edge_ts,
                )
            )

        return self._wait_all_tasks(tasks, "SOFTWARE", cycle)

    def capture_all(
        self,
        sides_to_capture: Optional[List[str]] = None,
    ) -> Dict[str, Optional[np.ndarray]]:
        """
        Returns side/role keyed images.

        sides_to_capture controls camera capture only.
        AI sides are controlled separately by cycle_engine.py.
        """
        if not self._streams_started:
            raise RuntimeError("Camera streams are not started. Call start_all_streams() first")

        if sides_to_capture is None:
            sides_to_capture = [
                "sidewall1",
                "sidewall2",
                "innerwall",
                "tread",
                "bead",
            ]

        active_capture_sides = set(sides_to_capture)

        log(f"[CAPTURE] active capture sides: {sorted(active_capture_sides)}")

        self._capture_index += 1
        cycle = self._capture_index

        results: Dict[str, Optional[np.ndarray]] = {}

        if TRIGGER_MODE == "plc_software":
            main_targets = self._build_role_targets(
                "main",
                sides_to_capture=sides_to_capture,
            )

            bead_targets = self._build_role_targets(
                "bead",
                sides_to_capture=sides_to_capture,
            )

            log(
                f"[CAPTURE] PLC_SOFTWARE cycle={cycle} started | "
                f"main_targets={[name for _, name in main_targets]} | "
                f"bead_targets={[name for _, name in bead_targets]}"
            )

            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                futures = []

                if main_targets:
                    futures.append(
                        pool.submit(
                            self._wait_then_capture_group,
                            "main",
                            main_targets,
                            MAIN_TRIGGER_DB,
                            MAIN_TRIGGER_BYTE,
                            MAIN_TRIGGER_BIT,
                            cycle,
                        )
                    )
                else:
                    log("[MAIN] skipped because no main-side capture requested")

                if bead_targets:
                    futures.append(
                        pool.submit(
                            self._wait_then_capture_group,
                            "bead",
                            bead_targets,
                            BEAD_TRIGGER_DB,
                            BEAD_TRIGGER_BYTE,
                            BEAD_TRIGGER_BIT,
                            cycle,
                        )
                    )
                else:
                    log("[BEAD] skipped because bead capture is not requested")

                for future in concurrent.futures.as_completed(futures):
                    try:
                        results.update(future.result())
                    except Exception:
                        traceback.print_exc()

            log(
                f"[CAPTURE] PLC_SOFTWARE cycle={cycle} completed | "
                f"keys={list(results.keys())}"
            )
            return results

        if TRIGGER_MODE in ("software", "free"):
            log(f"[CAPTURE] {TRIGGER_MODE.upper()} cycle={cycle} started")

            results.update(
                self._software_capture_once(
                    cycle,
                    sides_to_capture=sides_to_capture,
                )
            )

            log(
                f"[CAPTURE] {TRIGGER_MODE.upper()} cycle={cycle} completed | "
                f"keys={list(results.keys())}"
            )
            return results

        raise RuntimeError(f"Unsupported CAM_TRIGGER_MODE for this application file: {TRIGGER_MODE}")

    def close_all(self) -> None:
        self.stop_all_streams()
        for cam in self.cameras:
            cam.stop_and_close()
        try:
            system.destroy_device()
        except Exception as e:
            log(f"[WARN] system.destroy_device: {e}")


__all__ = [
    "TRIGGER_MODE",
    "get_camera_role_config",
    "get_camera_to_side_map",
    "get_side_to_camera_map",
    "get_camera_roles_by_serial",
    "get_physical_camera_config",
    "LineScanCamera",
    "MultiCameraManager",
    "wait_plc_fresh_rising_edge",
]
