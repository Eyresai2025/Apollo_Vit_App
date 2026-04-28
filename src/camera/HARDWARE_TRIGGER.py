from arena_api.system import system
from arena_api.buffer import BufferFactory
import ctypes
import numpy as np
import concurrent.futures
import traceback
import threading
import time
from typing import Optional, Dict, Any, Callable
from datetime import datetime
import os

# ─────────────────────────────────────────────
#  GLOBAL CONTROLS
# ─────────────────────────────────────────────

PARALLEL = True          # True  → capture all cameras at the same time
                         # False → capture cameras one after another

NUM_CAMERAS = 5          # How many cameras to manage

# Trigger mode selection
# "software" → PLC tag triggers capture (internal trigger, camera streams continuously)
# "hardware" → Physical hardware trigger signal to camera
TRIGGER_MODE = "software"  # Options: "software", "hardware"

# Hardware trigger settings (used only when TRIGGER_MODE = "hardware")
TRIGGER_SOURCE = "Line1"
TRIGGER_ACTIVATION = "RisingEdge"

# Software trigger settings
PLC_TRIGGER_POLL_INTERVAL = 0.01  # Seconds between PLC tag reads

# Serial numbers for each camera slot (index 0-4)
CAMERA_SERIALS = [
    "244802149",          # Camera 0
    "244802163",          # Camera 1
    "251102086",          # Camera 2
    "251401655",          # Camera 3
    "251300826",          # Camera 4
]

# Per-camera overrides
CAMERA_OVERRIDES = {
    # 0: {"gain_db": 20.0},
}

# ─────────────────────────────────────────────
#  SHARED DEFAULTS
# ─────────────────────────────────────────────

DEFAULT_CONFIG = dict(
    width              = 4096,
    camera_height      = 14000,
    final_height       = 42000,
    exposure_us        = 200.0,
    gain_db            = 24.0,
    line_rate          = 4096.178266,
    pixel_format       = "Mono16",
    num_stream_buffers = 16,
)

# ─────────────────────────────────────────────

class LineScanCamera:
    def __init__(
        self,
        width              = DEFAULT_CONFIG["width"],
        camera_height      = DEFAULT_CONFIG["camera_height"],
        final_height       = DEFAULT_CONFIG["final_height"],
        exposure_us        = DEFAULT_CONFIG["exposure_us"],
        gain_db            = DEFAULT_CONFIG["gain_db"],
        line_rate          = DEFAULT_CONFIG["line_rate"],
        pixel_format       = DEFAULT_CONFIG["pixel_format"],
        num_stream_buffers = DEFAULT_CONFIG["num_stream_buffers"],
        serial_number      = None,
        trigger_mode       = TRIGGER_MODE,
        trigger_source     = TRIGGER_SOURCE,
        trigger_activation = TRIGGER_ACTIVATION,
    ):
        self.width              = width
        self.camera_height      = camera_height
        self.final_height       = final_height
        self.exposure_us        = exposure_us
        self.gain_db            = gain_db
        self.line_rate          = line_rate
        self.pixel_format       = pixel_format
        self.num_stream_buffers = num_stream_buffers
        self.serial_number      = serial_number
        self.trigger_mode       = trigger_mode
        self.trigger_source     = trigger_source
        self.trigger_activation = trigger_activation

        self.device      = None
        self.nodemap     = None
        self.is_streaming = False
        self.is_connected = False
        
        self._stop_event = threading.Event()
        self._capture_lock = threading.Lock()

    def _convert_buffer(self, buffer):
        copied = BufferFactory.copy(buffer)
        try:
            width       = copied.width
            height      = copied.height
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

    def _set_node(self, name, value):
        try:
            node = self.nodemap.get_node(name)
            if node and node.is_writable:
                node.value = value
                print(f"  [{self.serial_number}] {name}: {node.value}")
            else:
                print(f"  [{self.serial_number}] {name}: not writable / not found")
        except Exception as e:
            print(f"  [{self.serial_number}] {name} not set: {e}")

    def _select_device(self, devices):
        if not devices:
            raise RuntimeError("No camera found")

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

    def _configure_trigger(self):
        """Configure trigger based on mode"""
        if self.trigger_mode == "hardware":
            print(f"  [{self.serial_number}] Configuring HARDWARE trigger...")
            self._set_node("TriggerSelector", "FrameStart")
            self._set_node("TriggerMode", "On")
            self._set_node("TriggerSource", self.trigger_source)
            self._set_node("TriggerActivation", self.trigger_activation)
            print(f"  [{self.serial_number}] Hardware trigger configured (source: {self.trigger_source})")
        else:
            print(f"  [{self.serial_number}] Configuring SOFTWARE trigger (continuous streaming)...")
            self._set_node("TriggerMode", "Off")
            self._set_node("TriggerSelector", "FrameStart")
            print(f"  [{self.serial_number}] Software trigger configured")

    def connect_and_configure(self):
        if self.is_connected:
            print(f"[{self.serial_number}] Already connected.")
            return

        devices      = system.create_device()
        self.device  = self._select_device(devices)
        self.nodemap = self.device.nodemap
        print(f" [{self.serial_number}] Camera connected")

        self._set_node("Width",                   self.width)
        self._set_node("Height",                  self.camera_height)
        self._set_node("PixelFormat",             self.pixel_format)
        self._set_node("ExposureAutoLimitAuto",   "Off")
        self._set_node("ExposureTime",            self.exposure_us)
        self._set_node("Gain",                    self.gain_db)
        self._set_node("AcquisitionLineRateEnable", True)
        self._set_node("AcquisitionLineRate",     self.line_rate)
        self._set_node("AcquisitionMode",         "Continuous")
        
        self._configure_trigger()

        self.is_streaming = False
        self.is_connected = True
        print(f" [{self.serial_number}] Camera configured ({self.trigger_mode} trigger)")

    def start_stream(self):
        """Start streaming (for both modes)"""
        if not self.is_connected or self.device is None:
            raise RuntimeError(f"[{self.serial_number}] Camera not connected.")

        if self.is_streaming:
            return

        mode_str = "HARDWARE trigger (waiting for signal)" if self.trigger_mode == "hardware" else "SOFTWARE trigger (continuous)"
        print(f" [{self.serial_number}] Starting stream - {mode_str}...")
        self.device.start_stream(self.num_stream_buffers)
        self.is_streaming = True
        self._stop_event.clear()

    def capture_stitched_image(self):
        """
        Capture one complete stitched image from running stream.
        For SOFTWARE mode: immediately captures from continuous stream
        For HARDWARE mode: waits for hardware trigger signal, then captures
        """
        if not self.is_streaming:
            raise RuntimeError(f"[{self.serial_number}] Stream not running.")

        with self._capture_lock:
            full_img    = np.zeros((self.final_height, self.width), dtype=np.uint16)
            current_row = 0

            if self.trigger_mode == "hardware":
                print(f"📸 [{self.serial_number}] Waiting for HARDWARE trigger signal...")
            else:
                print(f"📸 [{self.serial_number}] Capturing stitched image...")

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

                    remaining    = self.final_height - current_row
                    lines_to_copy = min(h, remaining)
                    full_img[current_row:current_row + lines_to_copy, :] = frame[:lines_to_copy, :]
                    current_row += lines_to_copy

                    print(f"  [{self.serial_number}] rows captured: {current_row}/{self.final_height}")
                finally:
                    self.device.requeue_buffer(buffer)

            print(f" [{self.serial_number}] Stitch complete: {full_img.shape}")
            return self.serial_number, full_img

    def stop_stream(self):
        """Stop the camera stream"""
        if self.device is not None and self.is_streaming:
            try:
                self._stop_event.set()
                self.device.stop_stream()
                self.is_streaming = False
                print(f" [{self.serial_number}] Stream stopped")
            except Exception as e:
                print(f"[WARN] [{self.serial_number}] Error stopping stream: {e}")

    def stop_and_close(self):
        print(f" [{self.serial_number}] Closing camera…")
        self.stop_stream()
        self.is_connected = False
        self.device       = None
        self.nodemap      = None
        try:
            system.destroy_device()
            print(f" [{self.serial_number}] Camera destroyed")
        except Exception as e:
            print(f"[WARN] [{self.serial_number}] destroy_device: {e}")


# ─────────────────────────────────────────────
#  MULTI-CAMERA MANAGER
# ─────────────────────────────────────────────

class MultiCameraManager:
    """
    Manages NUM_CAMERAS LineScanCamera instances.
    Supports both SOFTWARE (PLC) and HARDWARE trigger modes.
    
    SOFTWARE mode: Cameras stream continuously, capture triggered externally (PLC)
    HARDWARE mode: Cameras wait for physical trigger signal
    """

    def __init__(self, plc_interface=None):
        self.cameras: list[LineScanCamera] = []
        self.plc_interface = plc_interface
        self._streams_started = False

        for i in range(NUM_CAMERAS):
            cfg    = {**DEFAULT_CONFIG}
            cfg.update(CAMERA_OVERRIDES.get(i, {}))
            serial = CAMERA_SERIALS[i] if i < len(CAMERA_SERIALS) else None
            self.cameras.append(LineScanCamera(serial_number=serial, **cfg))

    def set_plc_interface(self, plc_interface):
        """Set the PLC interface"""
        self.plc_interface = plc_interface

    def connect_all(self):
        """Connect and configure every camera"""
        mode_str = "HARDWARE" if TRIGGER_MODE == "hardware" else "SOFTWARE (PLC)"
        print(f"\n{'='*50}")
        print(f"Connecting {NUM_CAMERAS} camera(s) - Trigger Mode: {mode_str}")
        print(f"{'='*50}")
        for cam in self.cameras:
            cam.connect_and_configure()
        print("All cameras connected.\n")

    def start_all_streams(self):
        """Start streams on all cameras"""
        if self._streams_started:
            return
            
        mode_str = "hardware trigger (waiting for signal)" if TRIGGER_MODE == "hardware" else "continuous streaming"
        print(f"\n{'='*50}")
        print(f"Starting streams on {NUM_CAMERAS} camera(s) - {mode_str}")
        print(f"{'='*50}")
        for cam in self.cameras:
            cam.start_stream()
        self._streams_started = True
        print("All camera streams started.\n")

    def stop_all_streams(self):
        """Stop streams on all cameras"""
        print(f"\n{'='*50}")
        print("Stopping all camera streams...")
        print(f"{'='*50}")
        for cam in self.cameras:
            cam.stop_stream()
        self._streams_started = False
        print("All camera streams stopped.\n")

    def capture_all(self) -> Dict[str, np.ndarray]:
        """
        Capture from all cameras (parallel).
        For SOFTWARE mode: captures immediately from streams
        For HARDWARE mode: each camera waits for its hardware trigger
        """
        results: Dict[str, Optional[np.ndarray]] = {}

        def _task(cam: LineScanCamera):
            return cam.capture_stitched_image()

        with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_CAMERAS) as pool:
            future_map = {pool.submit(_task, cam): cam for cam in self.cameras}
            for future in concurrent.futures.as_completed(future_map):
                cam = future_map[future]
                try:
                    serial, img = future.result()
                    results[serial] = img
                    if img is not None:
                        print(f" [{serial}] image ready — shape {img.shape}")
                except Exception:
                    results[cam.serial_number] = None
                    print(f" [{cam.serial_number}] capture FAILED:")
                    traceback.print_exc()

        return results

    def close_all(self):
        """Disconnect every camera cleanly."""
        print(f"\n{'='*50}")
        print("Closing all cameras…")
        print(f"{'='*50}")
        self.stop_all_streams()
        for cam in self.cameras:
            cam.stop_and_close()
        print(" All cameras closed.\n")