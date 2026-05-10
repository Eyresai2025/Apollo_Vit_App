from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
import ctypes
import time

import numpy as np
import cv2


@dataclass
class CameraInfo:
    serial: str
    model: str
    ip: str
    status: str


class ArenaCameraManager:
    def __init__(self):
        self.system = None
        self.devices = {}
        self.arena_available = False
        self.current_settings_by_serial = {}
        self.streaming_serials = set()

        try:
            from arena_api.system import system
            self.system = system
            self.arena_available = True
            print("[ARENA] Arena SDK loaded")
        except Exception as e:
            print("[ARENA] Arena SDK not available:", e)
            self.arena_available = False

    # ---------------------------------------------------------------------
    # Camera discovery
    # ---------------------------------------------------------------------
    def refresh_cameras(self):
        camera_list = []

        if not self.arena_available:
            print("[ARENA] Cannot refresh. Arena SDK not available.")
            return camera_list

        try:
            self.close_all()

            devices = self.system.create_device()
            self.devices.clear()

            for dev in devices:
                serial = str(self._get_node_value(dev, "DeviceSerialNumber", "-"))
                model = str(self._get_node_value(dev, "DeviceModelName", "-"))
                ip_raw = self._get_node_value(dev, "GevCurrentIPAddress", "-")
                ip = self._format_ip(ip_raw)

                self.devices[serial] = dev

                camera_list.append(
                    CameraInfo(
                        serial=serial,
                        model=model,
                        ip=ip,
                        status="Connected"
                    )
                )

        except Exception as e:
            print("[ARENA] refresh_cameras error:", e)

        return camera_list

    def get_device(self, serial: str):
        return self.devices.get(str(serial))

    def _format_ip(self, value):
        try:
            if isinstance(value, int):
                return ".".join(str((value >> shift) & 255) for shift in [24, 16, 8, 0])
            return str(value)
        except Exception:
            return str(value)

    def _get_node_value(self, dev, node_name, default=None):
        try:
            node = dev.nodemap.get_node(node_name)
            return node.value
        except Exception:
            return default

    # ---------------------------------------------------------------------
    # Safe helpers
    # ---------------------------------------------------------------------
    def _force_stop_stream(self, serial: str):
        """
        Always try to stop acquisition before changing Width/Height/PixelFormat/TriggerMode.
        This is required because these nodes are not writable during acquisition.
        """
        dev = self.get_device(serial)

        if dev is None:
            return

        try:
            dev.stop_stream()
            print(f"[ARENA] Force stop stream for {serial}")
        except Exception as e:
            # This warning is acceptable if stream was not running.
            print(f"[ARENA] force stop warning for {serial}: {e}")

        self.streaming_serials.discard(str(serial))
        time.sleep(0.2)

    def _set_node(self, nm, node_name, value, required=False):
        try:
            node = nm.get_node(node_name)

            # Check writability where Arena exposes it.
            try:
                if hasattr(node, "is_writable") and not node.is_writable:
                    msg = f"[ARENA] {node_name} is not writable"
                    print(msg)

                    if required:
                        raise RuntimeError(msg)

                    return False
            except Exception:
                pass

            node.value = value
            print(f"[ARENA] SET {node_name} = {value}")
            return True

        except Exception as e:
            msg = f"[ARENA] {'REQUIRED FAILED' if required else 'SKIP'} {node_name}: {e}"
            print(msg)

            if required:
                raise RuntimeError(msg)

            return False

    # ---------------------------------------------------------------------
    # Apply settings
    # ---------------------------------------------------------------------
    def apply_settings(self, serial: str, settings: dict, mode: str = None):
        """
        mode:
            preview_free_run = image quality checking, TriggerMode Off
            hardware         = production Line0 trigger settings
        """

        serial = str(serial)
        dev = self.get_device(serial)

        if dev is None:
            return False, f"Camera {serial} not connected"

        if mode is None:
            mode = "hardware" if settings.get("use_hardware_trigger", True) else "preview_free_run"

        try:
            # Critical: stop acquisition before changing Width/Height/PixelFormat.
            self._force_stop_stream(serial)

            nm = dev.nodemap

            width = int(settings.get("width", 4096))
            height = int(settings.get("height", 6000))
            pixel_format = settings.get("pixel_format", "Mono16")

            exposure_us = float(settings.get("exposure_time", 150.0))
            gain_db = float(settings.get("gain", 0.0))
            line_rate = float(settings.get("acquisition_line_rate", 4096.0))

            # Use 1500 as safe default. Use 9000 only after Jumbo Frames are enabled in Windows NIC.
            packet_size = int(settings.get("packet_size", 1500))

            # Trigger must be off before geometry/network changes.
            self._set_node(nm, "TriggerMode", "Off")
            time.sleep(0.05)

            # Geometry
            self._set_node(nm, "Width", width, required=True)
            self._set_node(nm, "Height", height, required=True)
            self._set_node(nm, "PixelFormat", pixel_format, required=True)

            # Exposure / gain
            self._set_node(nm, "ExposureAuto", "Off")
            self._set_node(nm, "ExposureTime", exposure_us)

            self._set_node(nm, "GainAuto", "Off")
            self._set_node(nm, "Gain", gain_db)

            # Line rate
            self._set_node(nm, "AcquisitionLineRateEnable", True)
            self._set_node(nm, "AcquisitionLineRate", line_rate)

            # Acquisition
            self._set_node(nm, "AcquisitionMode", "Continuous", required=True)

            # Network packet size
            self._set_node(nm, "GevSCPSPacketSize", packet_size)

            if mode == "preview_free_run":
                self._set_node(nm, "TriggerMode", "Off", required=True)

                print("[ARENA] Applied SOFTWARE/FREE-RUN preview settings")
                self.current_settings_by_serial[serial] = dict(settings)
                self.current_settings_by_serial[serial]["packet_size"] = packet_size
                return True, "Software/free-run preview settings applied"

            # Hardware trigger production mode
            self._set_node(
                nm,
                "LineSelector",
                settings.get("line_selector", "Line0"),
                required=True
            )

            self._set_node(
                nm,
                "LineMode",
                settings.get("line_mode", "Input"),
                required=True
            )

            self._set_node(
                nm,
                "LineSource",
                settings.get("line_source", "Off")
            )

            # Use FrameStart for line-scan trigger unless you specifically confirm AcquisitionStart is required.
            trigger_selector = settings.get("trigger_selector", "FrameStart")
            if trigger_selector == "AcquisitionStart":
                trigger_selector = "FrameStart"

            self._set_node(
                nm,
                "TriggerSelector",
                trigger_selector,
                required=True
            )

            self._set_node(
                nm,
                "TriggerSource",
                settings.get("trigger_source", "Line0"),
                required=True
            )

            self._set_node(
                nm,
                "TriggerActivation",
                settings.get("trigger_activation", "RisingEdge"),
                required=True
            )

            self._set_node(
                nm,
                "TriggerMode",
                "On",
                required=True
            )

            print("[ARENA] Applied HARDWARE TRIGGER Line0 settings")
            self.current_settings_by_serial[serial] = dict(settings)
            self.current_settings_by_serial[serial]["packet_size"] = packet_size
            self.current_settings_by_serial[serial]["trigger_selector"] = trigger_selector
            return True, "Hardware trigger settings applied"

        except Exception as e:
            return False, str(e)

    # ---------------------------------------------------------------------
    # Live preview
    # ---------------------------------------------------------------------
    def start_live_stream(self, serial: str, settings: dict, mode: str):
        serial = str(serial)
        dev = self.get_device(serial)

        if dev is None:
            raise RuntimeError(f"Camera {serial} not connected")

        # Always stop first, even if streaming_serials does not know.
        self._force_stop_stream(serial)

        ok, msg = self.apply_settings(serial, settings, mode=mode)

        if not ok:
            raise RuntimeError(msg)

        try:
            packet_size = int(settings.get("packet_size", 1500))
            print(f"[ARENA] Starting stream for {serial} with packet_size={packet_size}")

            dev.start_stream()
            self.streaming_serials.add(serial)

            print(f"[ARENA] Stream started for {serial} | mode={mode}")

        except Exception as e:
            self.streaming_serials.discard(serial)

            # Try to unlock camera after failed start_stream.
            try:
                dev.stop_stream()
            except Exception:
                pass

            raise RuntimeError(str(e))

    def stop_live_stream(self, serial: str):
        serial = str(serial)
        dev = self.get_device(serial)

        if dev is None:
            self.streaming_serials.discard(serial)
            return

        try:
            dev.stop_stream()
            print(f"[ARENA] Stream stopped for {serial}")
        except Exception as e:
            print(f"[ARENA] stop_stream warning for {serial}: {e}")

        self.streaming_serials.discard(serial)
        time.sleep(0.1)

    def get_live_frame(self, serial: str, timeout=1000):
        serial = str(serial)
        dev = self.get_device(serial)

        if dev is None:
            raise RuntimeError(f"Camera {serial} not connected")

        buffer = dev.get_buffer(timeout=timeout)

        try:
            img = self._copy_buffer_to_numpy(buffer, serial)
        finally:
            dev.requeue_buffer(buffer)

        return img

    # ---------------------------------------------------------------------
    # Capture one image
    # ---------------------------------------------------------------------
    def capture_one_image(
        self,
        serial: str,
        settings: dict,
        mode: str,
        save_dir="media/device_test_captures",
        timeout=8000
    ):
        serial = str(serial)
        dev = self.get_device(serial)

        if dev is None:
            raise RuntimeError(f"Camera {serial} not connected")

        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        if serial in self.streaming_serials:
            raise RuntimeError("Stop live preview before Capture One Image.")

        self.start_live_stream(serial, settings, mode=mode)

        frame = None

        try:
            frame = self.get_live_frame(serial, timeout=timeout)
            line_count = frame.shape[0]

        finally:
            self.stop_live_stream(serial)

        if frame is None:
            raise RuntimeError("Capture failed. No frame received.")

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        image_path = save_dir / f"test_capture_{serial}_{mode}_{ts}.png"

        cv2.imwrite(str(image_path), frame)

        return str(image_path), line_count

    # ---------------------------------------------------------------------
    # Buffer conversion
    # ---------------------------------------------------------------------
    def _copy_buffer_to_numpy(self, buffer, serial: str):
        settings = self.current_settings_by_serial.get(str(serial), {})
        pixel_format = settings.get("pixel_format", "Mono16")

        width = int(buffer.width)
        height = int(buffer.height)
        size = width * height

        if pixel_format == "Mono16":
            dtype = np.uint16
            ctype = ctypes.c_uint16
        else:
            dtype = np.uint8
            ctype = ctypes.c_ubyte

        try:
            arr = np.asarray(buffer.data)

            if arr.size >= size:
                arr = arr[:size].astype(dtype, copy=True)
                return arr.reshape((height, width))

        except Exception:
            pass

        try:
            raw = bytes(buffer.data)
            arr = np.frombuffer(raw, dtype=dtype)

            if arr.size >= size:
                arr = arr[:size].copy()
                return arr.reshape((height, width))

        except Exception:
            pass

        try:
            ptr = ctypes.cast(buffer.pdata, ctypes.POINTER(ctype))
            arr = np.ctypeslib.as_array(ptr, shape=(size,))
            arr = arr.copy()
            return arr.reshape((height, width))

        except Exception as e:
            raise RuntimeError(f"Could not convert Arena buffer to numpy: {e}")

    def close_all(self):
        print("[ARENA] Closing all Device Page cameras...")

        for serial in list(self.streaming_serials):
            try:
                self.stop_live_stream(serial)
            except Exception as e:
                print(f"[ARENA] stop stream failed for {serial}: {e}")

        self.streaming_serials.clear()

        for serial, dev in list(self.devices.items()):
            try:
                dev.stop_stream()
                print(f"[ARENA] stop_stream done for {serial}")
            except Exception:
                pass

        try:
            if self.arena_available and self.system is not None and self.devices:
                self.system.destroy_device()
                print("[ARENA] system.destroy_device() done")
        except Exception as e:
            print(f"[ARENA] destroy_device warning: {e}")

        self.devices.clear()
        self.current_settings_by_serial.clear()

        print("[ARENA] Device Page camera cleanup completed")