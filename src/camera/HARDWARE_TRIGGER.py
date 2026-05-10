from arena_api.system import system
from arena_api.buffer import BufferFactory

import ctypes
import numpy as np
import concurrent.futures
import traceback
import threading
from typing import Optional, Dict
from pathlib import Path


# =========================================================
# ENV LOADER
# =========================================================

def _project_root():
    """
    HARDWARE_TRIGGER.py location:
        src/camera/HARDWARE_TRIGGER.py

    Project root:
        two levels above src/camera
    """
    try:
        return Path(__file__).resolve().parents[2]
    except Exception:
        return Path.cwd()


def _load_env_file():
    env_path = _project_root() / ".env"
    data = {}

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


def _env_str(key, default=""):
    value = _ENV.get(key, "")

    if value is None:
        return default

    value = str(value).strip()

    if value == "":
        return default

    return value


def _env_int(key, default):
    value = _ENV.get(key, "")

    try:
        if value is None or str(value).strip() == "":
            return int(default)
        return int(float(str(value).strip()))
    except Exception:
        return int(default)


def _env_float(key, default):
    value = _ENV.get(key, "")

    try:
        if value is None or str(value).strip() == "":
            return float(default)
        return float(str(value).strip())
    except Exception:
        return float(default)


def _env_bool(key, default=False):
    value = _ENV.get(key, "")

    if value is None or str(value).strip() == "":
        return bool(default)

    value = str(value).strip().lower()

    return value in ("1", "true", "yes", "on")


def _side_key(side_name, field):
    return f"CAM_{side_name.upper()}_{field}"


def _side_or_global_str(side_name, field, global_key, default):
    side_value = _env_str(_side_key(side_name, field), "")
    if side_value != "":
        return side_value
    return _env_str(global_key, default)


def _side_or_global_int(side_name, field, global_key, default):
    side_value = _env_str(_side_key(side_name, field), "")
    if side_value != "":
        try:
            return int(float(side_value))
        except Exception:
            return int(default)
    return _env_int(global_key, default)


def _side_or_global_float(side_name, field, global_key, default):
    side_value = _env_str(_side_key(side_name, field), "")
    if side_value != "":
        try:
            return float(side_value)
        except Exception:
            return float(default)
    return _env_float(global_key, default)


def _side_or_global_bool(side_name, field, global_key, default):
    side_value = _env_str(_side_key(side_name, field), "")
    if side_value != "":
        return str(side_value).strip().lower() in ("1", "true", "yes", "on")
    return _env_bool(global_key, default)


# =========================================================
# CAMERA ROLE CONFIG
# =========================================================

CAMERA_ROLE_ORDER = [
    ("sidewall1", "CAM_SIDEWALL1_SERIAL"),
    ("sidewall2", "CAM_SIDEWALL2_SERIAL"),
    ("innerwall", "CAM_INNERWALL_SERIAL"),
    ("tread", "CAM_TREAD_SERIAL"),
    ("bead", "CAM_BEAD_SERIAL"),
]


def get_camera_role_config():
    """
    Reads all camera serials and node parameters from .env.

    Global values:
        CAM_WIDTH
        CAM_CAMERA_HEIGHT
        CAM_FINAL_HEIGHT
        CAM_PIXEL_FORMAT
        CAM_EXPOSURE_AUTO_LIMIT_AUTO
        CAM_EXPOSURE_TIME
        CAM_GAIN
        CAM_ACQUISITION_LINE_RATE_ENABLE
        CAM_ACQUISITION_LINE_RATE
        CAM_ACQUISITION_MODE
        CAM_STREAM_BUFFERS

    Side-specific override example:
        CAM_SIDEWALL1_EXPOSURE_TIME
        CAM_TREAD_GAIN
    """

    configs = []

    for side_name, serial_key in CAMERA_ROLE_ORDER:
        serial = _env_str(serial_key, "")

        if not serial:
            continue

        cfg = {
            "side": side_name,
            "serial": serial,

            # Camera image size / buffer
            "width": _side_or_global_int(side_name, "WIDTH", "CAM_WIDTH", 4096),
            "camera_height": _side_or_global_int(side_name, "CAMERA_HEIGHT", "CAM_CAMERA_HEIGHT", 14000),
            "final_height": _side_or_global_int(side_name, "FINAL_HEIGHT", "CAM_FINAL_HEIGHT", 42000),
            "pixel_format": _side_or_global_str(side_name, "PIXEL_FORMAT", "CAM_PIXEL_FORMAT", "Mono16"),
            "num_stream_buffers": _side_or_global_int(side_name, "STREAM_BUFFERS", "CAM_STREAM_BUFFERS", 16),

            # Exposure / gain
            "exposure_auto_limit_auto": _side_or_global_str(
                side_name,
                "EXPOSURE_AUTO_LIMIT_AUTO",
                "CAM_EXPOSURE_AUTO_LIMIT_AUTO",
                "Off",
            ),
            "exposure_time": _side_or_global_float(
                side_name,
                "EXPOSURE_TIME",
                "CAM_EXPOSURE_TIME",
                200.0,
            ),
            "gain": _side_or_global_float(
                side_name,
                "GAIN",
                "CAM_GAIN",
                24.0,
            ),

            # Acquisition
            "acquisition_line_rate_enable": _side_or_global_bool(
                side_name,
                "ACQUISITION_LINE_RATE_ENABLE",
                "CAM_ACQUISITION_LINE_RATE_ENABLE",
                True,
            ),
            "acquisition_line_rate": _side_or_global_float(
                side_name,
                "ACQUISITION_LINE_RATE",
                "CAM_ACQUISITION_LINE_RATE",
                4096.178266,
            ),
            "acquisition_mode": _side_or_global_str(
                side_name,
                "ACQUISITION_MODE",
                "CAM_ACQUISITION_MODE",
                "Continuous",
            ),
        }

        configs.append(cfg)

    return configs


def get_camera_to_side_map():
    return {
        item["serial"]: item["side"]
        for item in get_camera_role_config()
    }


def get_side_to_camera_map():
    return {
        item["side"]: item["serial"]
        for item in get_camera_role_config()
    }


# =========================================================
# GLOBAL CONTROLS FROM .env
# =========================================================

PARALLEL = True

CAMERA_ROLE_CONFIG = get_camera_role_config()
CAMERA_SERIALS = [item["serial"] for item in CAMERA_ROLE_CONFIG]
NUM_CAMERAS = len(CAMERA_SERIALS)

# software = PLC software trigger controls capture
# hardware = physical trigger signal to camera
TRIGGER_MODE = _env_str("CAM_TRIGGER_MODE", "software").lower()

TRIGGER_SELECTOR = _env_str("CAM_TRIGGER_SELECTOR", "FrameStart")
TRIGGER_SOURCE = _env_str("CAM_TRIGGER_SOURCE", "Line0")
TRIGGER_ACTIVATION = _env_str("CAM_TRIGGER_ACTIVATION", "RisingEdge")

PLC_TRIGGER_POLL_INTERVAL = _env_float("PLC_TRIGGER_POLL_INTERVAL", 0.01)

DEFAULT_CONFIG = dict(
    width=_env_int("CAM_WIDTH", 4096),
    camera_height=_env_int("CAM_CAMERA_HEIGHT", 14000),
    final_height=_env_int("CAM_FINAL_HEIGHT", 42000),
    pixel_format=_env_str("CAM_PIXEL_FORMAT", "Mono16"),
    num_stream_buffers=_env_int("CAM_STREAM_BUFFERS", 16),

    exposure_auto_limit_auto=_env_str("CAM_EXPOSURE_AUTO_LIMIT_AUTO", "Off"),
    exposure_time=_env_float("CAM_EXPOSURE_TIME", 200.0),
    gain=_env_float("CAM_GAIN", 24.0),

    acquisition_line_rate_enable=_env_bool("CAM_ACQUISITION_LINE_RATE_ENABLE", True),
    acquisition_line_rate=_env_float("CAM_ACQUISITION_LINE_RATE", 4096.178266),
    acquisition_mode=_env_str("CAM_ACQUISITION_MODE", "Continuous"),
)


# =========================================================
# LINE SCAN CAMERA
# =========================================================

class LineScanCamera:
    def __init__(
        self,
        side_name=None,
        serial_number=None,

        width=DEFAULT_CONFIG["width"],
        camera_height=DEFAULT_CONFIG["camera_height"],
        final_height=DEFAULT_CONFIG["final_height"],
        pixel_format=DEFAULT_CONFIG["pixel_format"],
        num_stream_buffers=DEFAULT_CONFIG["num_stream_buffers"],

        exposure_auto_limit_auto=DEFAULT_CONFIG["exposure_auto_limit_auto"],
        exposure_time=DEFAULT_CONFIG["exposure_time"],
        gain=DEFAULT_CONFIG["gain"],

        acquisition_line_rate_enable=DEFAULT_CONFIG["acquisition_line_rate_enable"],
        acquisition_line_rate=DEFAULT_CONFIG["acquisition_line_rate"],
        acquisition_mode=DEFAULT_CONFIG["acquisition_mode"],

        trigger_mode=TRIGGER_MODE,
        trigger_selector=TRIGGER_SELECTOR,
        trigger_source=TRIGGER_SOURCE,
        trigger_activation=TRIGGER_ACTIVATION,
    ):
        self.side_name = side_name
        self.serial_number = serial_number

        self.width = width
        self.camera_height = camera_height
        self.final_height = final_height
        self.pixel_format = pixel_format
        self.num_stream_buffers = num_stream_buffers

        self.exposure_auto_limit_auto = exposure_auto_limit_auto
        self.exposure_time = exposure_time
        self.gain = gain

        self.acquisition_line_rate_enable = acquisition_line_rate_enable
        self.acquisition_line_rate = acquisition_line_rate
        self.acquisition_mode = acquisition_mode

        self.trigger_mode = trigger_mode
        self.trigger_selector = trigger_selector
        self.trigger_source = trigger_source
        self.trigger_activation = trigger_activation

        self.device = None
        self.nodemap = None
        self.is_streaming = False
        self.is_connected = False

        self._stop_event = threading.Event()
        self._capture_lock = threading.Lock()

    # -----------------------------------------------------
    # BUFFER CONVERSION
    # -----------------------------------------------------
    def _convert_buffer(self, buffer):
        copied = BufferFactory.copy(buffer)

        try:
            width = copied.width
            height = copied.height
            total_bytes = len(copied.data)

            c_arr = (ctypes.c_ubyte * total_bytes).from_address(
                ctypes.addressof(copied.pbytes)
            )

            np_arr = np.ctypeslib.as_array(c_arr)

            bytes_per_pixel = total_bytes // (width * height)

            if bytes_per_pixel == 2:
                img = np_arr.view(np.uint16).reshape(height, width)
            else:
                img = np_arr.reshape(height, width)

            return img.copy()

        finally:
            BufferFactory.destroy(copied)

    # -----------------------------------------------------
    # NODE HELPERS
    # -----------------------------------------------------
    def _set_node(self, name, value):
        """
        Safe Lucid Arena node setter.
        If a node does not exist or is not writable, it will not crash the app.
        """
        try:
            if self.nodemap is None:
                print(f"  [{self.serial_number}] {name}: nodemap not ready")
                return False

            node = self.nodemap.get_node(name)

            if node and node.is_writable:
                node.value = value
                print(f"  [{self.serial_number}] {name}: {node.value}")
                return True

            print(f"  [{self.serial_number}] {name}: not writable / not found")
            return False

        except Exception as e:
            print(f"  [{self.serial_number}] {name} not set: {e}")
            return False

    def _get_node_value(self, name, default=None):
        try:
            if self.nodemap is None:
                return default

            node = self.nodemap.get_node(name)

            if node:
                return node.value

            return default

        except Exception:
            return default

    # -----------------------------------------------------
    # SELECT DEVICE BY SERIAL
    # -----------------------------------------------------
    def _select_device(self, devices):
        if not devices:
            raise RuntimeError("No Lucid camera found")

        if self.serial_number is None:
            return devices[0]

        for dev in devices:
            try:
                nm = dev.nodemap
                serial_node = nm.get_node("DeviceSerialNumber")

                if serial_node and str(serial_node.value) == str(self.serial_number):
                    return dev

            except Exception:
                continue

        raise RuntimeError(f"Camera with serial {self.serial_number} not found")

    # -----------------------------------------------------
    # TRIGGER CONFIG
    # -----------------------------------------------------
    def _configure_trigger(self):
        if self.trigger_mode == "hardware":
            print(f"  [{self.serial_number}] Configuring HARDWARE trigger...")

            self._set_node("TriggerSelector", self.trigger_selector)
            self._set_node("TriggerMode", "On")
            self._set_node("TriggerSource", self.trigger_source)
            self._set_node("TriggerActivation", self.trigger_activation)

            print(
                f"  [{self.serial_number}] Hardware trigger configured "
                f"(selector={self.trigger_selector}, source={self.trigger_source}, activation={self.trigger_activation})"
            )

        else:
            print(f"  [{self.serial_number}] Configuring SOFTWARE trigger...")

            self._set_node("TriggerSelector", self.trigger_selector)
            self._set_node("TriggerMode", "Off")

            print(f"  [{self.serial_number}] Software trigger configured")

    # -----------------------------------------------------
    # CONNECT + CONFIGURE CAMERA
    # -----------------------------------------------------
    def connect_and_configure(self):
        if self.is_connected:
            print(f"[{self.serial_number}] Already connected.")
            return

        devices = system.create_device()

        self.device = self._select_device(devices)
        self.nodemap = self.device.nodemap

        actual_serial = self._get_node_value("DeviceSerialNumber", self.serial_number)

        print("")
        print("--------------------------------------------------")
        print(f" [{self.serial_number}] Camera connected")
        print(f" Role: {self.side_name}")
        print(f" Actual Serial: {actual_serial}")
        print("--------------------------------------------------")

        # ---------------- Image format ----------------
        self._set_node("Width", self.width)
        self._set_node("Height", self.camera_height)
        self._set_node("PixelFormat", self.pixel_format)

        # ---------------- Exposure / gain ----------------
        self._set_node("ExposureAutoLimitAuto", self.exposure_auto_limit_auto)
        self._set_node("ExposureTime", self.exposure_time)
        self._set_node("Gain", self.gain)

        # ---------------- Acquisition ----------------
        self._set_node("AcquisitionLineRateEnable", self.acquisition_line_rate_enable)
        self._set_node("AcquisitionLineRate", self.acquisition_line_rate)
        self._set_node("AcquisitionMode", self.acquisition_mode)

        # ---------------- Trigger ----------------
        self._configure_trigger()

        self.is_streaming = False
        self.is_connected = True

        print(
            f" [{self.serial_number}] Camera configured | "
            f"side={self.side_name} | "
            f"size={self.width}x{self.camera_height} | "
            f"final_height={self.final_height} | "
            f"exposure={self.exposure_time} | "
            f"gain={self.gain} | "
            f"line_rate={self.acquisition_line_rate} | "
            f"trigger={self.trigger_mode}"
        )

    # -----------------------------------------------------
    # STREAM CONTROL
    # -----------------------------------------------------
    def start_stream(self):
        if not self.is_connected or self.device is None:
            raise RuntimeError(f"[{self.serial_number}] Camera not connected.")

        if self.is_streaming:
            return

        mode_str = (
            "HARDWARE trigger waiting"
            if self.trigger_mode == "hardware"
            else "SOFTWARE / continuous streaming"
        )

        print(f" [{self.serial_number}] Starting stream - {mode_str}...")
        self.device.start_stream(self.num_stream_buffers)
        self.is_streaming = True
        self._stop_event.clear()

    def stop_stream(self):
        if self.device is not None and self.is_streaming:
            try:
                self._stop_event.set()
                self.device.stop_stream()
                self.is_streaming = False
                print(f" [{self.serial_number}] Stream stopped")
            except Exception as e:
                print(f"[WARN] [{self.serial_number}] Error stopping stream: {e}")

    # -----------------------------------------------------
    # CAPTURE
    # -----------------------------------------------------
    def capture_stitched_image(self):
        """
        Captures one stitched image from this camera stream.
        Uses:
            camera_height = individual buffer height
            final_height  = final stitched image height
        """
        if not self.is_streaming:
            raise RuntimeError(f"[{self.serial_number}] Stream not running.")

        with self._capture_lock:
            full_img = np.zeros((self.final_height, self.width), dtype=np.uint16)
            current_row = 0

            if self.trigger_mode == "hardware":
                print(f" [{self.serial_number}] Waiting for HARDWARE trigger signal...")
            else:
                print(f" [{self.serial_number}] Capturing stitched image...")

            while current_row < self.final_height:
                if self._stop_event.is_set():
                    return self.serial_number, None

                buffer = self.device.get_buffer()

                try:
                    frame = self._convert_buffer(buffer)

                    if frame.ndim != 2:
                        raise RuntimeError(f"Unexpected frame shape: {frame.shape}")

                    h, w = frame.shape

                    if w != self.width:
                        raise RuntimeError(f"Width mismatch: got {w}, expected {self.width}")

                    remaining = self.final_height - current_row
                    lines_to_copy = min(h, remaining)

                    full_img[current_row:current_row + lines_to_copy, :] = frame[:lines_to_copy, :]
                    current_row += lines_to_copy

                    print(f"  [{self.serial_number}] rows captured: {current_row}/{self.final_height}")

                finally:
                    self.device.requeue_buffer(buffer)

            print(f" [{self.serial_number}] Stitch complete: {full_img.shape}")

            return self.serial_number, full_img

    # -----------------------------------------------------
    # CLOSE
    # -----------------------------------------------------
    def stop_and_close(self):
        print(f" [{self.serial_number}] Closing camera...")

        self.stop_stream()

        self.is_connected = False
        self.device = None
        self.nodemap = None

        try:
            system.destroy_device()
            print(f" [{self.serial_number}] Camera destroyed")
        except Exception as e:
            print(f"[WARN] [{self.serial_number}] destroy_device: {e}")


# =========================================================
# MULTI-CAMERA MANAGER
# =========================================================

class MultiCameraManager:
    """
    Manages all Lucid line-scan cameras configured in .env.

    Test Mode calls:
        manager.connect_all()

    Live mode calls:
        manager.start_all_streams()
        manager.capture_all()
    """

    def __init__(self, plc_interface=None):
        self.cameras: list[LineScanCamera] = []
        self.plc_interface = plc_interface
        self._streams_started = False

        self.camera_role_config = get_camera_role_config()
        self.camera_to_side = get_camera_to_side_map()
        self.side_to_camera = get_side_to_camera_map()

        if not self.camera_role_config:
            raise RuntimeError(
                "No camera serials configured in .env. "
                "Please set CAM_SIDEWALL1_SERIAL, CAM_SIDEWALL2_SERIAL, CAM_INNERWALL_SERIAL, CAM_TREAD_SERIAL, CAM_BEAD_SERIAL."
            )

        for item in self.camera_role_config:
            cam = LineScanCamera(
                side_name=item["side"],
                serial_number=item["serial"],

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

                trigger_mode=TRIGGER_MODE,
                trigger_selector=TRIGGER_SELECTOR,
                trigger_source=TRIGGER_SOURCE,
                trigger_activation=TRIGGER_ACTIVATION,
            )

            self.cameras.append(cam)

    def set_plc_interface(self, plc_interface):
        self.plc_interface = plc_interface

    def connect_all(self, fail_fast=False):
        mode_str = "HARDWARE" if TRIGGER_MODE == "hardware" else "SOFTWARE / PLC"

        print("")
        print("=" * 60)
        print(f"Connecting {len(self.cameras)} Lucid camera(s)")
        print(f"Trigger Mode: {mode_str}")
        print("Camera Role Mapping:")

        for serial, side in self.camera_to_side.items():
            print(f"  {serial} -> {side}")

        print("=" * 60)

        for cam in self.cameras:
            try:
                cam.connect_and_configure()

            except Exception as e:
                cam.is_connected = False
                cam.device = None
                cam.nodemap = None

                print(
                    f"[CAMERA][ERROR] "
                    f"side={cam.side_name} | serial={cam.serial_number} | failed: {e}"
                )

                if fail_fast:
                    raise

        connected = [
            f"{cam.side_name}:{cam.serial_number}"
            for cam in self.cameras
            if cam.is_connected
        ]

        missing = [
            f"{cam.side_name}:{cam.serial_number}"
            for cam in self.cameras
            if not cam.is_connected
        ]

        print("")
        print("[CAMERA] Connected cameras:", connected)
        print("[CAMERA] Missing/failed cameras:", missing)

        if not connected:
            raise RuntimeError("No configured Lucid cameras connected.")

        print("")
        return len(missing) == 0

    def stop_all_streams(self):
        print("")
        print("=" * 60)
        print("Stopping all camera streams...")
        print("=" * 60)

        for cam in self.cameras:
            cam.stop_stream()

        self._streams_started = False

        print("All camera streams stopped.")
        print("")

    def capture_all(self) -> Dict[str, np.ndarray]:
        """
        Capture from all configured cameras in parallel.
        Returns:
            {
                serial_number: image_array
            }
        """
        results: Dict[str, Optional[np.ndarray]] = {}

        def _task(cam: LineScanCamera):
            return cam.capture_stitched_image()

        max_workers = max(1, len(self.cameras))

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
            future_map = {
                pool.submit(_task, cam): cam
                for cam in self.cameras
            }

            for future in concurrent.futures.as_completed(future_map):
                cam = future_map[future]

                try:
                    serial, img = future.result()
                    results[serial] = img

                    if img is not None:
                        print(f" [{serial}] image ready — shape {img.shape}")
                    else:
                        print(f" [{serial}] image is None")

                except Exception:
                    results[cam.serial_number] = None
                    print(f" [{cam.serial_number}] capture FAILED:")
                    traceback.print_exc()

        return results

    def close_all(self):
        print("")
        print("=" * 60)
        print("Closing all cameras...")
        print("=" * 60)

        self.stop_all_streams()

        for cam in self.cameras:
            cam.stop_and_close()

        print("All cameras closed.")
        print("")