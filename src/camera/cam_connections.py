from arena_api.system import system
from arena_api.buffer import BufferFactory
import ctypes
import numpy as np
import concurrent.futures
import traceback

# ─────────────────────────────────────────────
#  GLOBAL CONTROLS  ←  edit these to tune behaviour
# ─────────────────────────────────────────────

PARALLEL = True          # True  → capture all cameras at the same time
                         # False → capture cameras one after another

NUM_CAMERAS = 5          # How many cameras to manage

# Serial numbers for each camera slot (index 0-4).
# Set a value to None to auto-select (first found).
CAMERA_SERIALS = [
    "12345001",          # Camera 0
    "12345002",          # Camera 1
    "12345003",          # Camera 2
    "12345004",          # Camera 3
    "12345005",          # Camera 4
]

# Per-camera overrides.  Keys must match LineScanCamera.__init__ kwargs.
# Omit a camera index to use the shared defaults below.
CAMERA_OVERRIDES = {
    # 0: {"gain_db": 20.0},   # example: camera 0 uses different gain
}

# ─────────────────────────────────────────────
#  SHARED DEFAULTS  (applied to every camera)
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

        self.device      = None
        self.nodemap     = None
        self.is_streaming = False
        self.is_connected = False

    # ── internal helpers ───────────────────────────────────────────────

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

    # ── public API ─────────────────────────────────────────────────────

    def connect_and_configure(self):
        if self.is_connected:
            print(f"[{self.serial_number}] Already connected.")
            return

        devices      = system.create_device()
        self.device  = self._select_device(devices)
        self.nodemap = self.device.nodemap
        print(f"✅ [{self.serial_number}] Camera connected")

        self._set_node("Width",                   self.width)
        self._set_node("Height",                  self.camera_height)
        self._set_node("PixelFormat",             self.pixel_format)
        self._set_node("ExposureAutoLimitAuto",   "Off")
        self._set_node("ExposureTime",            self.exposure_us)
        self._set_node("Gain",                    self.gain_db)
        self._set_node("AcquisitionLineRateEnable", True)
        self._set_node("AcquisitionLineRate",     self.line_rate)
        self._set_node("AcquisitionMode",         "Continuous")
        self._set_node("TriggerMode",             "Off")

        self.is_streaming = False
        self.is_connected = True
        print(f"✅ [{self.serial_number}] Camera configured")

    def capture_one_stitched_image(self):
        """
        Start a fresh stream → stitch one full image → stop stream.
        Returns (serial_number, image_ndarray).
        """
        if not self.is_connected or self.device is None:
            raise RuntimeError(
                f"[{self.serial_number}] Camera not connected. "
                "Call connect_and_configure() first."
            )

        if self.is_streaming:
            print(f"[WARN] [{self.serial_number}] Stream already running — stopping first.")
            try:
                self.device.stop_stream()
            except Exception as e:
                print(f"[WARN] [{self.serial_number}] stop_stream failed: {e}")
            self.is_streaming = False

        print(f"🚀 [{self.serial_number}] Starting stream ({self.num_stream_buffers} buffers)…")
        self.device.start_stream(self.num_stream_buffers)
        self.is_streaming = True

        try:
            full_img    = np.zeros((self.final_height, self.width), dtype=np.uint16)
            current_row = 0

            while current_row < self.final_height:
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

            print(f"✅ [{self.serial_number}] Stitch complete: {full_img.shape}")
            return self.serial_number, full_img

        finally:
            try:
                self.device.stop_stream()
                print(f"✅ [{self.serial_number}] Stream stopped")
            except Exception as e:
                print(f"[WARN] [{self.serial_number}] Error stopping stream: {e}")
            self.is_streaming = False

    def stop_and_close(self):
        print(f"🛑 [{self.serial_number}] Closing camera…")

        if self.device is not None and self.is_streaming:
            try:
                self.device.stop_stream()
                print(f"✅ [{self.serial_number}] Stream stopped")
            except Exception as e:
                print(f"[WARN] [{self.serial_number}] stop_stream: {e}")

        self.is_streaming = False
        self.is_connected = False
        self.device       = None
        self.nodemap      = None

        try:
            system.destroy_device()
            print(f"✅ [{self.serial_number}] Camera destroyed")
        except Exception as e:
            print(f"[WARN] [{self.serial_number}] destroy_device: {e}")


# ─────────────────────────────────────────────
#  MULTI-CAMERA MANAGER
# ─────────────────────────────────────────────

class MultiCameraManager:
    """
    Manages NUM_CAMERAS LineScanCamera instances.

    Usage
    -----
        manager = MultiCameraManager()
        manager.connect_all()
        results = manager.capture_all()   # dict: {serial → np.ndarray}
        manager.close_all()
    """

    def __init__(self):
        self.cameras: list[LineScanCamera] = []

        for i in range(NUM_CAMERAS):
            cfg    = {**DEFAULT_CONFIG}                    # shared defaults
            cfg.update(CAMERA_OVERRIDES.get(i, {}))        # per-camera overrides
            serial = CAMERA_SERIALS[i] if i < len(CAMERA_SERIALS) else None
            self.cameras.append(LineScanCamera(serial_number=serial, **cfg))

    # ── connect ────────────────────────────────────────────────────────

    def connect_all(self):
        """Connect and configure every camera (always sequential — SDK requirement)."""
        print(f"\n{'='*50}")
        print(f"Connecting {NUM_CAMERAS} camera(s)…")
        print(f"{'='*50}")
        for cam in self.cameras:
            cam.connect_and_configure()
        print("✅ All cameras connected.\n")

    # ── capture ────────────────────────────────────────────────────────

    def capture_all(self) -> dict:
        """
        Capture one stitched image from every camera.

        Returns
        -------
        dict[str, np.ndarray]
            {serial_number: image}  — one entry per camera.
            Failed cameras are still present; their value is None.
        """
        print(f"\n{'='*50}")
        print(f"Capturing from {NUM_CAMERAS} camera(s) | PARALLEL={PARALLEL}")
        print(f"{'='*50}\n")

        results: dict[str, np.ndarray | None] = {}

        if PARALLEL:
            results = self._capture_parallel()
        else:
            results = self._capture_sequential()

        ok      = sum(1 for v in results.values() if v is not None)
        failed  = NUM_CAMERAS - ok
        print(f"\n✅ Capture done — {ok}/{NUM_CAMERAS} succeeded, {failed} failed.")
        return results

    def _capture_parallel(self) -> dict:
        results = {}

        def _task(cam: LineScanCamera):
            return cam.capture_one_stitched_image()

        with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_CAMERAS) as pool:
            future_map = {pool.submit(_task, cam): cam for cam in self.cameras}

            for future in concurrent.futures.as_completed(future_map):
                cam = future_map[future]
                try:
                    serial, img = future.result()
                    results[serial] = img
                    print(f"✅ [{serial}] image ready — shape {img.shape}")
                except Exception:
                    serial = cam.serial_number
                    results[serial] = None
                    print(f"❌ [{serial}] capture FAILED:")
                    traceback.print_exc()

        return results

    def _capture_sequential(self) -> dict:
        results = {}
        for cam in self.cameras:
            try:
                serial, img = cam.capture_one_stitched_image()
                results[serial] = img
                print(f"✅ [{serial}] image ready — shape {img.shape}")
            except Exception:
                serial = cam.serial_number
                results[serial] = None
                print(f"❌ [{serial}] capture FAILED:")
                traceback.print_exc()
        return results

    # ── close ──────────────────────────────────────────────────────────

    def close_all(self):
        """Disconnect every camera cleanly."""
        print(f"\n{'='*50}")
        print("Closing all cameras…")
        print(f"{'='*50}")
        for cam in self.cameras:
            cam.stop_and_close()
        print("✅ All cameras closed.\n")


# ─────────────────────────────────────────────
#  QUICK SMOKE-TEST  (run this file directly)
# ─────────────────────────────────────────────

if __name__ == "__main__":
    manager = MultiCameraManager()

    try:
        manager.connect_all()
        images = manager.capture_all()

        # images is: { "12345001": np.ndarray, "12345002": np.ndarray, … }
        for serial, img in images.items():
            if img is not None:
                print(f"  Camera {serial} → image shape: {img.shape}, dtype: {img.dtype}")
            else:
                print(f"  Camera {serial} → FAILED (None)")

    finally:
        manager.close_all()