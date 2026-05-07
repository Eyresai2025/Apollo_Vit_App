from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
import ctypes

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
            for serial in list(self.streaming_serials):
                self.stop_live_stream(serial)

            try:
                if self.devices:
                    self.system.destroy_device()
                    self.devices.clear()
            except Exception as e:
                print("[ARENA] destroy_device warning:", e)

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
    # Safe node setter
    # ---------------------------------------------------------------------
    def _set_node(self, nm, node_name, value, required=False):
        try:
            node = nm.get_node(node_name)
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
            "preview_free_run" = image quality checking, TriggerMode Off
            "hardware"         = production Line0 trigger settings
        """

        dev = self.get_device(serial)

        if dev is None:
            return False, f"Camera {serial} not connected"

        if mode is None:
            mode = "hardware" if settings.get("use_hardware_trigger", True) else "preview_free_run"

        try:
            nm = dev.nodemap

            width = int(settings.get("width", 4096))
            height = int(settings.get("height", 6000))
            pixel_format = settings.get("pixel_format", "Mono16")

            exposure_us = float(settings.get("exposure_time", 150.0))
            gain_db = float(settings.get("gain", 0.0))
            line_rate = float(settings.get("acquisition_line_rate", 4096.0))
            packet_size = int(settings.get("packet_size", 9000))

            # Always stop trigger before changing trigger-related nodes
            self._set_node(nm, "TriggerMode", "Off")

            # Geometry
            self._set_node(nm, "Width", width, required=True)
            self._set_node(nm, "Height", height, required=True)
            self._set_node(nm, "PixelFormat", pixel_format, required=True)

            # Exposure / gain manual
            self._set_node(nm, "ExposureAuto", "Off")
            self._set_node(nm, "ExposureTime", exposure_us)

            self._set_node(nm, "GainAuto", "Off")
            self._set_node(nm, "Gain", gain_db)

            # Line rate
            self._set_node(nm, "AcquisitionLineRateEnable", True)
            self._set_node(nm, "AcquisitionLineRate", line_rate)

            # Continuous acquisition
            self._set_node(nm, "AcquisitionMode", "Continuous", required=True)

            # Network packet size
            self._set_node(nm, "GevSCPSPacketSize", packet_size)

            if mode == "preview_free_run":
                # Image quality checking mode.
                # No hardware trigger required.
                self._set_node(nm, "TriggerMode", "Off", required=True)

                print("[ARENA] Applied SOFTWARE/FREE-RUN preview settings")
                self.current_settings_by_serial[str(serial)] = dict(settings)
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

            self._set_node(
                nm,
                "TriggerSelector",
                settings.get("trigger_selector", "AcquisitionStart"),
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
            self.current_settings_by_serial[str(serial)] = dict(settings)
            return True, "Hardware trigger settings applied"

        except Exception as e:
            return False, str(e)

    # ---------------------------------------------------------------------
    # Live preview
    # ---------------------------------------------------------------------
    def start_live_stream(self, serial: str, settings: dict, mode: str):
        dev = self.get_device(serial)

        if dev is None:
            raise RuntimeError(f"Camera {serial} not connected")

        if serial in self.streaming_serials:
            self.stop_live_stream(serial)

        ok, msg = self.apply_settings(serial, settings, mode=mode)

        if not ok:
            raise RuntimeError(msg)

        dev.start_stream()
        self.streaming_serials.add(serial)

        print(f"[ARENA] Stream started for {serial} | mode={mode}")

    def stop_live_stream(self, serial: str):
        dev = self.get_device(serial)

        if dev is None:
            return

        try:
            dev.stop_stream()
            print(f"[ARENA] Stream stopped for {serial}")
        except Exception as e:
            print(f"[ARENA] stop_stream warning for {serial}: {e}")

        self.streaming_serials.discard(serial)

    def get_live_frame(self, serial: str, timeout=1000):
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
        dev = self.get_device(serial)

        if dev is None:
            raise RuntimeError(f"Camera {serial} not connected")

        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        was_streaming = serial in self.streaming_serials

        if was_streaming:
            raise RuntimeError("Stop live preview before Capture One Image.")

        self.start_live_stream(serial, settings, mode=mode)

        try:
            frame = self.get_live_frame(serial, timeout=timeout)
            line_count = frame.shape[0]

        finally:
            self.stop_live_stream(serial)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        image_path = save_dir / f"test_capture_{serial}_{mode}_{ts}.png"

        cv2.imwrite(str(image_path), frame)

        return str(image_path), line_count

    # ---------------------------------------------------------------------
    # Buffer conversion
    # ---------------------------------------------------------------------
    def _copy_buffer_to_numpy(self, buffer, serial: str):
        """
        Converts Arena buffer to numpy.
        Supports Mono8 and Mono16.
        """

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

        # Method 1: buffer.data
        try:
            arr = np.asarray(buffer.data)

            if arr.size >= size:
                arr = arr[:size].astype(dtype, copy=True)
                return arr.reshape((height, width))

        except Exception:
            pass

        # Method 2: bytes(buffer.data)
        try:
            raw = bytes(buffer.data)
            arr = np.frombuffer(raw, dtype=dtype)

            if arr.size >= size:
                arr = arr[:size].copy()
                return arr.reshape((height, width))

        except Exception:
            pass

        # Method 3: buffer.pdata pointer
        try:
            ptr = ctypes.cast(buffer.pdata, ctypes.POINTER(ctype))
            arr = np.ctypeslib.as_array(ptr, shape=(size,))
            arr = arr.copy()
            return arr.reshape((height, width))

        except Exception as e:
            raise RuntimeError(f"Could not convert Arena buffer to numpy: {e}")
        
    def close_all(self):
        """
        Gracefully stop all live streams and release Arena camera handles.
        Call this when leaving Device page or closing app.
        """

        print("[ARENA] Closing all Device Page cameras...")

        # Stop all active streams first
        for serial in list(self.streaming_serials):
            try:
                self.stop_live_stream(serial)
            except Exception as e:
                print(f"[ARENA] stop stream failed for {serial}: {e}")

        self.streaming_serials.clear()

        # Try stopping any camera that may still be open
        for serial, dev in list(self.devices.items()):
            try:
                dev.stop_stream()
                print(f"[ARENA] stop_stream done for {serial}")
            except Exception:
                pass

        # Release Arena device handles
        try:
            if self.arena_available and self.system is not None and self.devices:
                self.system.destroy_device()
                print("[ARENA] system.destroy_device() done")
        except Exception as e:
            print(f"[ARENA] destroy_device warning: {e}")

        self.devices.clear()
        self.current_settings_by_serial.clear()

        print("[ARENA] Device Page camera cleanup completed")