from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
import json
import os
import time

import numpy as np
import cv2


@dataclass
class LaserInfo:
    laser_id: str
    laser_name: str
    model: str
    interface: str
    status: str


class TeledyneLaserManager:
    def __init__(self):
        self.connected_lasers = {}
        self.current_settings_by_laser = {}
        self.streaming_lasers = set()

        self.harvester = None
        self.acquirers = {}

        self._load_env_file()

        self.cti_path = os.environ.get("TELEDYNE_CTI_PATH", "").strip()
        mock_env = os.environ.get("TELEDYNE_LASER_MOCK", "False").strip().lower()
        self.mock_mode = mock_env in ("1", "true", "yes")

        if self.mock_mode:
            print("[LASER] Running in MOCK mode")
            return

        try:
            from harvesters.core import Harvester

            if not self.cti_path or not os.path.exists(self.cti_path):
                raise RuntimeError(
                    f"Invalid TELEDYNE_CTI_PATH:\n{self.cti_path}\n"
                    "Set correct Sapera/Teledyne .cti path in .env"
                )

            self.harvester = Harvester()
            self.harvester.add_file(self.cti_path)
            self.harvester.update()

            print(f"[LASER] CTI loaded: {self.cti_path}")
            print(f"[LASER] Device count: {len(self.harvester.device_info_list)}")

        except Exception as e:
            print(f"[LASER] Real backend init failed: {e}")
            print("[LASER] Falling back to MOCK mode")
            self.mock_mode = True

    # ------------------------------------------------------------
    # ENV loader fallback
    # ------------------------------------------------------------
    def _load_env_file(self):
        env_path = Path(".env")
        if not env_path.exists():
            return

        try:
            with open(env_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue

                    key, value = line.split("=", 1)
                    key = key.strip()
                    value = value.strip().strip('"').strip("'")

                    if key and key not in os.environ:
                        os.environ[key] = value
        except Exception as e:
            print(f"[LASER] .env load skipped: {e}")

    # ------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------
    def refresh_lasers(self):
        self.close_all()

        if self.mock_mode:
            return self._mock_refresh_lasers()

        laser_list = []

        try:
            self.harvester.update()

            for idx, dev_info in enumerate(self.harvester.device_info_list):
                serial = str(
                    getattr(dev_info, "serial_number", "")
                    or getattr(dev_info, "id_", "")
                    or idx
                )

                model = str(
                    getattr(dev_info, "model", "")
                    or "Teledyne DALSA Z-Trak"
                )

                display_name = str(
                    getattr(dev_info, "display_name", "")
                    or getattr(dev_info, "user_defined_name", "")
                    or serial
                )

                laser_id = serial
                laser_name = display_name

                info = LaserInfo(
                    laser_id=laser_id,
                    laser_name=laser_name,
                    model=model,
                    interface="GenTL/Sapera",
                    status="Connected"
                )

                self.connected_lasers[laser_id] = {
                    "index": idx,
                    "info": info,
                    "dev_info": dev_info,
                }

                laser_list.append(info)

            print(f"[LASER] refresh_lasers found {len(laser_list)} device(s)")
            return laser_list

        except Exception as e:
            print(f"[LASER] refresh_lasers failed: {e}")
            return laser_list

    def _mock_refresh_lasers(self):
        dummy = [
            LaserInfo("LASER_SW1", "Laser_SW1", "Teledyne DALSA Z-Trak", "Mock", "Connected"),
            LaserInfo("LASER_SW2", "Laser_SW2", "Teledyne DALSA Z-Trak", "Mock", "Connected"),
            LaserInfo("LASER_TREAD", "Laser_Tread", "Teledyne DALSA Z-Trak", "Mock", "Connected"),
        ]

        for d in dummy:
            self.connected_lasers[d.laser_id] = d

        return dummy

    # ------------------------------------------------------------
    # Open / close acquirer
    # ------------------------------------------------------------
    def _get_or_create_acquirer(self, laser_id):
        if laser_id in self.acquirers:
            return self.acquirers[laser_id]

        if laser_id not in self.connected_lasers:
            raise RuntimeError(f"Laser {laser_id} not found. Click Refresh Lasers first.")

        index = self.connected_lasers[laser_id]["index"]

        try:
            ia = self.harvester.create(index)
        except Exception:
            ia = self.harvester.create_image_acquirer(list_index=index)

        self.acquirers[laser_id] = ia
        print(f"[LASER] Opened acquirer for {laser_id}")
        return ia

    def _close_acquirer(self, laser_id):
        ia = self.acquirers.pop(laser_id, None)

        if ia is None:
            return

        try:
            ia.stop()
        except Exception:
            pass

        try:
            ia.destroy()
        except Exception:
            pass

        print(f"[LASER] Closed acquirer for {laser_id}")

    # ------------------------------------------------------------
    # Node helpers
    # ------------------------------------------------------------
    def _get_nodemap(self, ia):
        return ia.remote_device.node_map

    def _get_node_object(self, nm, name):
        try:
            return nm.get_node(name)
        except Exception:
            pass

        try:
            return getattr(nm, name)
        except Exception:
            pass

        return None

    def _set_node(self, nm, names, value, required=False):
        if isinstance(names, str):
            names = [names]

        last_error = None

        for name in names:
            try:
                node = self._get_node_object(nm, name)
                if node is None:
                    continue

                node.value = value
                print(f"[LASER] SET {name} = {value}")
                return True

            except Exception as e:
                last_error = e

        msg = f"Could not set {names} = {value}. Last error: {last_error}"

        if required:
            raise RuntimeError("[LASER] REQUIRED FAILED: " + msg)

        print("[LASER] SKIP:", msg)
        return False

    def _get_node_value(self, nm, names, default=None):
        if isinstance(names, str):
            names = [names]

        for name in names:
            try:
                node = self._get_node_object(nm, name)
                if node is not None:
                    return node.value
            except Exception:
                pass

        return default

    def _execute_node(self, nm, names, required=False):
        if isinstance(names, str):
            names = [names]

        last_error = None

        for name in names:
            try:
                node = self._get_node_object(nm, name)
                if node is None:
                    continue

                node.execute()
                print(f"[LASER] EXECUTE {name}")
                return True
            except Exception as e:
                last_error = e

        msg = f"Could not execute {names}. Last error: {last_error}"

        if required:
            raise RuntimeError("[LASER] REQUIRED EXECUTE FAILED: " + msg)

        print("[LASER] SKIP:", msg)
        return False

    def _load_user_set(self, nm, user_set):
        if not user_set:
            return

        try:
            self._set_node(nm, "UserSetSelector", user_set)
            self._execute_node(nm, "UserSetLoad")
            print(f"[LASER] Loaded user set: {user_set}")
        except Exception as e:
            print(f"[LASER] UserSet load skipped: {e}")

    # ------------------------------------------------------------
    # Apply settings from GUI
    # ------------------------------------------------------------
    def apply_settings(self, laser_id: str, settings: dict):
        if laser_id not in self.connected_lasers:
            return False, f"Laser {laser_id} not connected"

        try:
            self.current_settings_by_laser[laser_id] = dict(settings)

            if self.mock_mode:
                print(f"[LASER] MOCK apply settings for {laser_id}: {settings}")
                return True, "Laser settings applied in mock mode"

            ia = self._get_or_create_acquirer(laser_id)
            nm = self._get_nodemap(ia)

            # Optional UserSet load.
            # If unchecked in GUI, this will skip and apply all settings directly.
            if bool(settings.get("use_user_set", False)):
                self._load_user_set(nm, settings.get("user_set", "UserSet1"))
            else:
                print("[LASER] Skipping UserSet load. Applying all settings directly from GUI.")

            # Stop trigger before changing settings
            self._set_node(nm, "TriggerMode", "Off")

            # Output mode: Sherlock/Z-Expert equivalent
            self._set_node(
                nm,
                ["DeviceOutputType", "DeviceScanType", "Scan3dOutputMode"],
                settings.get("device_output_type", "Linescan3D")
            )

            self._set_node(
                nm,
                ["Scan3dDataType", "Scan3dCoordinateFormat", "Device3DDataType"],
                settings.get("scan3d_data_type", "UniformX Z")
            )

            profiles_per_scan = int(settings.get("profiles_per_scan", 1))
            roi_width = int(settings.get("roi_width", 4096))
            roi_x_start = int(settings.get("roi_x_start", 0))

            # ROI / width
            self._set_node(nm, "Width", roi_width)
            self._set_node(nm, ["OffsetX", "RegionOffsetX"], roi_x_start)

            # Profiles per scan: node names may differ by Z-Trak model
            self._set_node(
                nm,
                ["ProfilesPerScan", "AcquisitionFrameCount", "Height"],
                profiles_per_scan
            )

            # Exposure
            self._set_node(nm, "ExposureAuto", "Off")
            self._set_node(
                nm,
                ["ExposureTime", "ExposureTimeAbs"],
                float(settings.get("exposure", 100.0))
            )

            # Scan rate
            self._set_node(nm, "AcquisitionLineRateEnable", True)
            self._set_node(
                nm,
                ["AcquisitionLineRate", "AcquisitionFrameRate"],
                float(settings.get("scan_rate", 4000.0))
            )

            # Threshold and averaging if available
            self._set_node(
                nm,
                ["Threshold", "DetectionThreshold", "LaserThreshold"],
                float(settings.get("threshold", 50.0))
            )

            self._set_node(
                nm,
                ["ProfileAveraging", "Averaging", "TemporalAveraging"],
                int(settings.get("profile_averaging", 1))
            )

            # Packet size
            self._set_node(
                nm,
                "GevSCPSPacketSize",
                int(settings.get("packet_size", 9000))
            )

            # Continuous acquisition
            self._set_node(nm, "AcquisitionMode", "Continuous")

            # Trigger
            trigger_mode = settings.get("trigger_mode", "Off")

            if trigger_mode == "Off":
                self._set_node(nm, "TriggerMode", "Off", required=True)
            else:
                self._set_node(nm, "TriggerMode", "Off")
                self._set_node(nm, "TriggerSelector", "FrameStart")
                self._set_node(nm, "TriggerSource", settings.get("trigger_source", "Line0"))
                self._set_node(nm, "TriggerActivation", settings.get("trigger_activation", "RisingEdge"))
                self._set_node(nm, "TriggerMode", "On", required=True)

            return True, "Real laser settings applied from GUI"

        except Exception as e:
            return False, str(e)

    # ------------------------------------------------------------
    # Start / stop stream
    # ------------------------------------------------------------
    def start_live_stream(self, laser_id: str, settings: dict):
        ok, msg = self.apply_settings(laser_id, settings)

        if not ok:
            raise RuntimeError(msg)

        if self.mock_mode:
            self.streaming_lasers.add(laser_id)
            return

        ia = self._get_or_create_acquirer(laser_id)

        try:
            ia.start()
        except Exception:
            ia.start_acquisition()

        self.streaming_lasers.add(laser_id)
        print(f"[LASER] Stream started for {laser_id}")

    def stop_live_stream(self, laser_id: str):
        if self.mock_mode:
            self.streaming_lasers.discard(laser_id)
            return

        ia = self.acquirers.get(laser_id)

        if ia is not None:
            try:
                ia.stop()
            except Exception:
                try:
                    ia.stop_acquisition()
                except Exception:
                    pass

        self.streaming_lasers.discard(laser_id)
        print(f"[LASER] Stream stopped for {laser_id}")

    # ------------------------------------------------------------
    # Fetch profile/range data
    # ------------------------------------------------------------
    def get_live_profile(self, laser_id: str, timeout=1000):
        if laser_id not in self.connected_lasers:
            raise RuntimeError(f"Laser {laser_id} not connected")

        settings = self.current_settings_by_laser.get(laser_id, {})

        if self.mock_mode:
            time.sleep(0.03)
            return self._generate_dummy_profile(settings)

        range_map = self.get_live_range_map(laser_id, timeout=timeout)

        # For live display, show middle profile line
        if range_map.ndim == 2:
            row = range_map.shape[0] // 2
            z = range_map[row, :]
        else:
            z = range_map.reshape(-1)

        x = self._build_x_axis(len(z), laser_id, settings)
        profile = np.stack([x, z.astype(np.float32)], axis=1)
        return profile

    def get_live_range_map(self, laser_id: str, timeout=1000):
        if laser_id not in self.connected_lasers:
            raise RuntimeError(f"Laser {laser_id} not connected")

        settings = self.current_settings_by_laser.get(laser_id, {})

        if self.mock_mode:
            return self._generate_dummy_range_map(settings)

        ia = self._get_or_create_acquirer(laser_id)

        timeout_sec = max(0.1, float(timeout) / 1000.0)

        with ia.fetch(timeout=timeout_sec) as buffer:
            range_map = self._buffer_to_range_map(buffer, laser_id, settings)

        return range_map

    def _buffer_to_range_map(self, buffer, laser_id, settings):
        payload = buffer.payload

        if not hasattr(payload, "components") or len(payload.components) == 0:
            raise RuntimeError("No payload components returned from laser")

        component = payload.components[0]
        data = np.asarray(component.data).copy()

        width = int(getattr(component, "width", 0) or settings.get("roi_width", 4096))
        height = int(getattr(component, "height", 0) or settings.get("profiles_per_scan", 1))

        if height <= 0:
            height = int(settings.get("profiles_per_scan", 1))

        if width <= 0:
            width = int(data.size / max(height, 1))

        required = width * height

        if data.size < required:
            raise RuntimeError(
                f"Laser buffer too small. Got {data.size}, expected {required}, "
                f"width={width}, height={height}"
            )

        raw = data[:required].reshape(height, width)

        z_scale, z_offset = self._read_z_scale_offset(laser_id)
        z = raw.astype(np.float32) * float(z_scale) + float(z_offset)

        invalid_value = str(settings.get("invalid_value", "")).strip()

        if invalid_value:
            try:
                invalid_value_float = float(invalid_value)
                z[raw == invalid_value_float] = np.nan
            except Exception:
                pass

        return z

    def _build_x_axis(self, n, laser_id, settings):
        x_scale, x_offset = self._read_x_scale_offset(laser_id)
        user_x_scale = float(settings.get("x_scale", 1.0))

        x = np.arange(n, dtype=np.float32) * float(x_scale) * user_x_scale + float(x_offset)
        return x

    def _read_x_scale_offset(self, laser_id):
        return self._read_scan3d_scale_offset(
            laser_id,
            coordinate="CoordinateA",
            default_scale=1.0
        )

    def _read_z_scale_offset(self, laser_id):
        return self._read_scan3d_scale_offset(
            laser_id,
            coordinate="CoordinateC",
            default_scale=1.0
        )

    def _read_scan3d_scale_offset(self, laser_id, coordinate, default_scale=1.0):
        try:
            ia = self.acquirers.get(laser_id)

            if ia is None:
                return default_scale, 0.0

            nm = self._get_nodemap(ia)

            self._set_node(nm, "Scan3dCoordinateSelector", coordinate)

            scale = self._get_node_value(nm, "Scan3dCoordinateScale", default_scale)
            offset = self._get_node_value(nm, "Scan3dCoordinateOffset", 0.0)

            return float(scale), float(offset)

        except Exception:
            return default_scale, 0.0

    # ------------------------------------------------------------
    # Capture and save
    # ------------------------------------------------------------
    def capture_one_profile(self, laser_id: str, settings: dict, save_dir="media/laser_test_captures"):
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        was_streaming = laser_id in self.streaming_lasers

        if not was_streaming:
            self.start_live_stream(laser_id, settings)

        try:
            profiles_per_scan = int(settings.get("profiles_per_scan", 1))

            if profiles_per_scan > 1:
                range_map = self.get_live_range_map(laser_id, timeout=8000)
                profile = self._range_map_to_middle_profile(range_map, laser_id, settings)
                preview_image = self.range_map_to_preview_image(range_map)
            else:
                profile = self.get_live_profile(laser_id, timeout=8000)
                range_map = profile[:, 1].reshape(1, -1)
                preview_image = self.profile_to_preview_image(
                    profile,
                    x_scale=float(settings.get("x_scale", 1.0)),
                    z_scale=float(settings.get("z_scale", 1.0)),
                )

            metrics = self.compute_quality_metrics(profile)

        finally:
            if not was_streaming:
                self.stop_live_stream(laser_id)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")

        profile_npy_path = save_dir / f"{laser_id}_{ts}_profile.npy"
        range_npy_path = save_dir / f"{laser_id}_{ts}_range_map.npy"
        csv_path = save_dir / f"{laser_id}_{ts}_profile.csv"
        png_path = save_dir / f"{laser_id}_{ts}_preview.png"
        json_path = save_dir / f"{laser_id}_{ts}_metrics.json"

        np.save(str(profile_npy_path), profile)
        np.save(str(range_npy_path), range_map)
        np.savetxt(str(csv_path), profile, delimiter=",", header="x,z", comments="")
        cv2.imwrite(str(png_path), preview_image)

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=4)

        return {
            "npy_path": str(profile_npy_path),
            "range_npy_path": str(range_npy_path),
            "csv_path": str(csv_path),
            "png_path": str(png_path),
            "json_path": str(json_path),
            "metrics": metrics,
        }

    def _range_map_to_middle_profile(self, range_map, laser_id, settings):
        row = range_map.shape[0] // 2
        z = range_map[row, :]
        x = self._build_x_axis(len(z), laser_id, settings)
        return np.stack([x, z.astype(np.float32)], axis=1)

    # ------------------------------------------------------------
    # Quality metrics
    # ------------------------------------------------------------
    def compute_quality_metrics(self, profile: np.ndarray) -> dict:
        if profile is None or len(profile) == 0:
            return {
                "valid_points_percent": 0.0,
                "missing_points_percent": 100.0,
                "outlier_points_percent": 100.0,
                "z_range": 0.0,
                "snr_score": 0.0,
                "decision": "REJECT",
                "reason": "Empty profile",
            }

        z = profile[:, 1]
        total = len(z)

        valid_mask = np.isfinite(z)
        valid_count = int(np.sum(valid_mask))
        missing_count = total - valid_count

        if valid_count <= 5:
            return {
                "valid_points_percent": round(100.0 * valid_count / total, 2),
                "missing_points_percent": round(100.0 * missing_count / total, 2),
                "outlier_points_percent": 100.0,
                "z_range": 0.0,
                "snr_score": 0.0,
                "decision": "REJECT",
                "reason": "Too few valid points",
            }

        z_valid = z[valid_mask]
        z_median = float(np.median(z_valid))
        z_mad = float(np.median(np.abs(z_valid - z_median)) + 1e-6)

        outlier_mask = np.abs(z_valid - z_median) > 6.0 * z_mad
        outlier_count = int(np.sum(outlier_mask))

        z_range = float(np.max(z_valid) - np.min(z_valid))
        noise = float(np.std(z_valid - self._smooth_1d(z_valid, k=21)) + 1e-6)
        signal = float(np.std(z_valid) + 1e-6)
        snr_score = signal / noise

        missing_percent = 100.0 * missing_count / total
        valid_percent = 100.0 * valid_count / total
        outlier_percent = 100.0 * outlier_count / max(valid_count, 1)

        decision = "ACCEPT"
        reason = "OK"

        if missing_percent > 15:
            decision = "REJECT"
            reason = "Too many missing points"
        elif outlier_percent > 10:
            decision = "REJECT"
            reason = "Too many outliers"
        elif z_range <= 1e-6:
            decision = "REJECT"
            reason = "Flat profile / no height variation"
        elif snr_score < 1.5:
            decision = "REJECT"
            reason = "Low profile quality"

        return {
            "valid_points_percent": round(valid_percent, 2),
            "missing_points_percent": round(missing_percent, 2),
            "outlier_points_percent": round(outlier_percent, 2),
            "z_range": round(z_range, 4),
            "snr_score": round(float(snr_score), 3),
            "decision": decision,
            "reason": reason,
        }

    def _smooth_1d(self, arr, k=21):
        if len(arr) < k:
            return arr

        k = max(3, int(k))

        if k % 2 == 0:
            k += 1

        kernel = np.ones(k, dtype=np.float32) / float(k)
        return np.convolve(arr, kernel, mode="same")

    # ------------------------------------------------------------
    # Preview drawing
    # ------------------------------------------------------------
    def profile_to_preview_image(self, profile, width=900, height=420, x_scale=1.0, z_scale=1.0):
        canvas = np.zeros((height, width, 3), dtype=np.uint8)

        if profile is None or len(profile) == 0:
            cv2.putText(canvas, "No Profile", (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            return canvas

        x = profile[:, 0].astype(np.float32) * float(x_scale)
        z = profile[:, 1].astype(np.float32) * float(z_scale)

        valid = np.isfinite(x) & np.isfinite(z)

        if np.sum(valid) < 2:
            cv2.putText(canvas, "Invalid Profile", (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            return canvas

        x = x[valid]
        z = z[valid]

        x_min, x_max = float(np.min(x)), float(np.max(x))
        z_min, z_max = float(np.min(z)), float(np.max(z))

        if abs(x_max - x_min) < 1e-6:
            x_max = x_min + 1.0

        if abs(z_max - z_min) < 1e-6:
            z_max = z_min + 1.0

        px = ((x - x_min) / (x_max - x_min) * (width - 40) + 20).astype(np.int32)
        py = (height - 20 - ((z - z_min) / (z_max - z_min) * (height - 40))).astype(np.int32)

        points = np.stack([px, py], axis=1)

        for i in range(1, len(points)):
            cv2.line(canvas, tuple(points[i - 1]), tuple(points[i]), (0, 255, 0), 1)

        cv2.putText(canvas, "2D Laser Profile", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(canvas, f"Z Range: {z_max - z_min:.2f}", (20, height - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

        return canvas

    def range_map_to_preview_image(self, range_map, width=900, height=420):
        z = np.asarray(range_map, dtype=np.float32)

        valid = np.isfinite(z)

        if np.sum(valid) < 5:
            canvas = np.zeros((height, width, 3), dtype=np.uint8)
            cv2.putText(canvas, "Invalid Range Map", (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            return canvas

        z_min = float(np.nanmin(z))
        z_max = float(np.nanmax(z))

        if abs(z_max - z_min) < 1e-6:
            z_max = z_min + 1.0

        display = ((z - z_min) / (z_max - z_min) * 255.0)
        display[~valid] = 0
        display = np.clip(display, 0, 255).astype(np.uint8)

        display = cv2.resize(display, (width, height), interpolation=cv2.INTER_AREA)
        color = cv2.applyColorMap(display, cv2.COLORMAP_JET)

        cv2.putText(color, "3D Height / Range Map", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(color, f"Z Range: {z_max - z_min:.2f}", (20, height - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

        return color

    # ------------------------------------------------------------
    # Dummy mode
    # ------------------------------------------------------------
    def _generate_dummy_profile(self, settings):
        n = int(settings.get("roi_width", 4096))
        n = max(300, min(n, 4096))

        x = np.linspace(0, 100, n).astype(np.float32)

        z = (
            20.0
            + 4.0 * np.sin(x / 9.0)
            + 1.5 * np.sin(x / 2.7)
            + np.random.normal(0, 0.15, size=n)
        ).astype(np.float32)

        missing_count = int(n * 0.02)

        if missing_count > 0:
            idx = np.random.choice(n, missing_count, replace=False)
            z[idx] = np.nan

        return np.stack([x, z], axis=1)

    def _generate_dummy_range_map(self, settings):
        profiles = int(settings.get("profiles_per_scan", 1))
        width = int(settings.get("roi_width", 4096))

        profiles = max(1, min(profiles, 3000))
        width = max(300, min(width, 4096))

        x = np.linspace(0, 100, width).astype(np.float32)
        rows = []

        for i in range(profiles):
            z = (
                20.0
                + 4.0 * np.sin(x / 9.0 + i * 0.01)
                + 1.5 * np.sin(x / 2.7)
                + np.random.normal(0, 0.15, size=width)
            ).astype(np.float32)
            rows.append(z)

        return np.asarray(rows, dtype=np.float32)

    # ------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------
    def close_all(self):
        for laser_id in list(self.streaming_lasers):
            try:
                self.stop_live_stream(laser_id)
            except Exception:
                pass

        self.streaming_lasers.clear()

        for laser_id in list(self.acquirers.keys()):
            try:
                self._close_acquirer(laser_id)
            except Exception:
                pass

        self.acquirers.clear()
        self.connected_lasers.clear()
        self.current_settings_by_laser.clear()

        try:
            if self.harvester is not None:
                self.harvester.reset()
        except Exception:
            pass

        print("[LASER] Laser manager cleanup completed")