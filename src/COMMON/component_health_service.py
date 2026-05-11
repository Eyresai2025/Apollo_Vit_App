# src/COMMON/component_health_service.py

import os
import time
import shutil
from pathlib import Path


class ComponentHealthService:
    """
    Lightweight Live Page health monitor.

    IMPORTANT:
    - Does NOT connect cameras.
    - Does NOT configure cameras.
    - Does NOT reconnect PLC.
    - Does NOT load AI models.
    - Does NOT run Test Mode check.
    - Only reads existing state from full_hardware_check.py and lightweight system info.
    """

    def __init__(self, media_path, env_path=None):
        self.media_path = Path(media_path)
        self.env_path = Path(env_path) if env_path else self.media_path.parent / ".env"
        self.env = self._load_env_file(self.env_path)

        self.deployment = self.env.get("DEPLOYMENT", "False")

        self.storage_min_free_gb = self._env_float("STORAGE_MIN_FREE_GB", 20.0)

        self.app_ok_db = self._env_int("APP_OK_DB", 100)
        self.app_ok_byte = self._env_int("APP_OK_BYTE", 0)
        self.app_ok_bit = self._env_int("APP_OK_BIT", 4)

        self.plc_mode_db = self._env_int("PLC_MODE_DB", 100)
        self.plc_mode_byte = self._env_int("PLC_MODE_BYTE", 2)
        self.plc_mode_size = self._env_int("PLC_MODE_SIZE", 2)
        self.require_laser = self._env_bool("REQUIRE_LASER", False)
        self._storage_cache = None
        self._storage_cache_time = 0
        self._storage_cache_interval_sec = 30

        self._gpu_cache = None
        self._gpu_cache_time = 0
        self._gpu_cache_interval_sec = 10

    # ------------------------------------------------------------
    # ENV
    # ------------------------------------------------------------
    def _load_env_file(self, env_path):
        data = {}

        try:
            if env_path and Path(env_path).exists():
                with open(env_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith("#") or "=" not in line:
                            continue
                        key, val = line.split("=", 1)
                        data[key.strip()] = val.strip().strip('"').strip("'")
        except Exception:
            pass

        return data

    def _env_int(self, key, default):
        try:
            value = self.env.get(key, "")
            if value is None or str(value).strip() == "":
                return int(default)
            return int(float(str(value).strip()))
        except Exception:
            return int(default)

    def _env_float(self, key, default):
        try:
            value = self.env.get(key, "")
            if value is None or str(value).strip() == "":
                return float(default)
            return float(str(value).strip())
        except Exception:
            return float(default)
    def _env_bool(self, key, default=False):
        value = self.env.get(key, "")

        if value is None or str(value).strip() == "":
            return bool(default)

        return str(value).strip().lower() in ("1", "true", "yes", "y", "on")
    # ------------------------------------------------------------
    # PLC SMALL READ HELPERS
    # ------------------------------------------------------------
    def _read_db_bit(self, client, db_number, byte_index, bit_index):
        data = client.db_read(db_number, byte_index, 1)
        return bool(data[0] & (1 << bit_index))

    def _read_db_word(self, client, db_number, byte_index, size=2):
        data = client.db_read(db_number, byte_index, size)

        if not data:
            return None

        if len(data) >= 2:
            return int.from_bytes(bytes(data[:2]), byteorder="big", signed=False)

        return int(data[0])

    def _decode_mode(self, mode_value):
        """
        PLC mode mapping from PLC team:
            0 = UNKNOWN
            1 = MANUAL
            2 = AUTO
            3 = FAULT
            4 = TEACHING
        """

        try:
            mode_value = int(mode_value)
        except Exception:
            return "UNKNOWN"

        mode_map = {
            0: "UNKNOWN",
            1: "MANUAL",
            2: "AUTO",
            3: "FAULT",
            4: "TEACHING",
        }

        return mode_map.get(mode_value, "UNKNOWN")

    # ------------------------------------------------------------
    # GPU
    # ------------------------------------------------------------
    def _check_gpu(self):
        now = time.time()

        if self._gpu_cache is not None and (now - self._gpu_cache_time) < self._gpu_cache_interval_sec:
            return self._gpu_cache

        result = {
            "ok": False,
            "title": "GPU",
            "text": "CUDA not available",
            "detail": {},
        }

        try:
            import torch

            if torch.cuda.is_available():
                name = torch.cuda.get_device_name(0)
                result["ok"] = True
                result["text"] = f"OK - {name}"
                result["detail"] = {
                    "cuda": True,
                    "device_name": name,
                }
            else:
                result["text"] = "CUDA not available"

        except Exception as e:
            result["text"] = f"GPU check error: {e}"

        self._gpu_cache = result
        self._gpu_cache_time = now
        return result

    # ------------------------------------------------------------
    # STORAGE
    # ------------------------------------------------------------
    def _check_storage(self):
        now = time.time()

        if self._storage_cache is not None and (now - self._storage_cache_time) < self._storage_cache_interval_sec:
            return self._storage_cache

        result = {
            "ok": False,
            "title": "Storage",
            "text": "Storage check failed",
            "detail": {},
        }

        try:
            usage = shutil.disk_usage(str(self.media_path))
            free_gb = usage.free / (1024 ** 3)

            result["ok"] = free_gb >= self.storage_min_free_gb
            result["text"] = f"{free_gb:.1f} GB free"
            result["detail"] = {
                "free_gb": round(free_gb, 2),
                "min_required_gb": self.storage_min_free_gb,
            }

        except Exception as e:
            result["text"] = f"Storage error: {e}"

        self._storage_cache = result
        self._storage_cache_time = now
        return result

    # ------------------------------------------------------------
    # TEST MODE STATE
    # ------------------------------------------------------------
    def _get_hardware_state(self):
        try:
            from src.COMMON.full_hardware_check import get_hardware_state
            return get_hardware_state()
        except Exception:
            return {
                "ready": False,
                "last_result": None,
                "plc_client": None,
                "multi_cam": None,
            }

    # ------------------------------------------------------------
    # PLC HEALTH
    # ------------------------------------------------------------
    def _check_plc(self, state):
        result = {
            "ok": False,
            "title": "PLC",
            "text": "Not checked",
            "detail": {},
        }

        last_result = state.get("last_result") or {}
        client = state.get("plc_client")

        if str(self.deployment) != "True":
            if last_result:
                result["ok"] = bool(last_result.get("plc_ok", False))
                result["text"] = "Demo OK" if result["ok"] else "Demo not ready"
            else:
                result["text"] = "Demo not checked"
            return result

        if client is None:
            result["text"] = "No PLC client"
            return result

        try:
            connected = bool(client.get_connected())
            result["ok"] = connected
            result["text"] = "Connected" if connected else "Disconnected"
            result["detail"]["connected"] = connected

        except Exception as e:
            result["text"] = f"PLC error: {e}"

        return result

    # ------------------------------------------------------------
    # APP OK BIT
    # ------------------------------------------------------------
    def _check_app_ok_bit(self, state):
        result = {
            "ok": False,
            "title": "App OK",
            "text": "Not verified",
            "detail": {},
        }

        last_result = state.get("last_result") or {}
        details = last_result.get("details", {}) if isinstance(last_result, dict) else {}
        app_detail = details.get("application_ok_bit", {})
        client = state.get("plc_client")

        address = f"DB{self.app_ok_db}.DBX{self.app_ok_byte}.{self.app_ok_bit}"
        result["detail"]["address"] = address

        if str(self.deployment) != "True":
            verified = app_detail.get("verified", app_detail.get("sent", False))
            result["ok"] = bool(verified)
            result["text"] = "Demo verified" if result["ok"] else "Demo not verified"
            return result

        if client is None:
            result["text"] = "No PLC client"
            return result

        try:
            value = self._read_db_bit(
                client,
                self.app_ok_db,
                self.app_ok_byte,
                self.app_ok_bit,
            )

            result["ok"] = bool(value)
            result["text"] = f"{address} = {value}"
            result["detail"]["read_back_value"] = value

        except Exception as e:
            result["text"] = f"Read failed: {e}"

        return result

    # ------------------------------------------------------------
    # PLC MODE
    # ------------------------------------------------------------
    def _check_machine_mode(self, state):
        result = {
            "mode": "UNKNOWN",
            "mode_ok": False,
            "text": "Mode: UNKNOWN",
        }

        client = state.get("plc_client")

        if str(self.deployment) != "True":
            last_result = state.get("last_result")
            if last_result:
                result["mode"] = "DEMO"
                result["mode_ok"] = True
                result["text"] = "Mode: DEMO"
            return result

        if client is None:
            return result

        try:
            raw_mode = self._read_db_word(
                client,
                self.plc_mode_db,
                self.plc_mode_byte,
                self.plc_mode_size,
            )

            if raw_mode is None:
                return result

            mode_text = self._decode_mode(raw_mode)

            print(
                f"[PLC MODE] Raw DB{self.plc_mode_db}.DBW{self.plc_mode_byte} = "
                f"{raw_mode} -> {mode_text}"
            )

            result["mode"] = mode_text
            result["mode_ok"] = mode_text in ("MANUAL", "AUTO", "TEACHING")
            result["text"] = f"Mode: {mode_text}"

        except Exception:
            pass

        return result

    # ------------------------------------------------------------
    # CAMERA HEALTH
    # ------------------------------------------------------------
    def _check_cameras(self, state):
        result = {
            "ok": False,
            "title": "Cameras",
            "text": "Not checked",
            "detail": {},
        }

        last_result = state.get("last_result") or {}
        details = last_result.get("details", {}) if isinstance(last_result, dict) else {}
        camera_detail = details.get("camera", {})
        camera_status = camera_detail.get("camera_status", [])

        multi_cam = state.get("multi_cam")

        # Prefer live object state if available
        if multi_cam is not None and hasattr(multi_cam, "cameras"):
            try:
                total = len(multi_cam.cameras)
                connected = 0

                for cam in multi_cam.cameras:
                    if getattr(cam, "is_connected", False):
                        connected += 1

                result["ok"] = total > 0 and connected == total
                result["text"] = f"{connected}/{total} connected"
                result["detail"] = {
                    "source": "multi_cam object",
                    "connected": connected,
                    "total": total,
                }
                return result

            except Exception as e:
                result["text"] = f"Camera object check error: {e}"

        # Fallback to last Test Mode result
        if camera_status:
            total = len(camera_status)
            connected = sum(1 for c in camera_status if c.get("connected"))

            result["ok"] = total > 0 and connected == total
            result["text"] = f"{connected}/{total} connected"
            result["detail"] = {
                "source": "last Test Mode result",
                "connected": connected,
                "total": total,
            }
            return result

        return result

    # ------------------------------------------------------------
    # LASER HEALTH
    # ------------------------------------------------------------
    def _check_laser(self, state):
        result = {
            "ok": False,
            "title": "Laser",
            "text": "Not checked",
            "detail": {},
        }

        last_result = state.get("last_result") or {}
        details = last_result.get("details", {}) if isinstance(last_result, dict) else {}
        laser_detail = details.get("laser", {})

        if not laser_detail:
            return result

        connected = bool(laser_detail.get("connected", False))
        message = laser_detail.get("message", "-")

        result["ok"] = connected
        result["text"] = "Connected" if connected else "Not connected"
        result["detail"] = {
            "message": message,
        }

        return result

    # ------------------------------------------------------------
    # MAIN
    # ------------------------------------------------------------
    def get_health(self, inspection_running=False):
        state = self._get_hardware_state()

        plc = self._check_plc(state)
        app_ok = self._check_app_ok_bit(state)
        cameras = self._check_cameras(state)
        laser = self._check_laser(state)
        gpu = self._check_gpu()
        storage = self._check_storage()
        mode = self._check_machine_mode(state)

        # All items shown in Live Page component health
        items = {
            "plc": plc,
            "cameras": cameras,
            "laser": laser,
            "gpu": gpu,
            "storage": storage,
            "app_ok": app_ok,
        }

        # Only these items decide System READY / NOT READY
        required_items = {
            "plc": plc,
            "cameras": cameras,
            "gpu": gpu,
            "storage": storage,
            "app_ok": app_ok,
        }

        # Laser should block READY only if REQUIRE_LASER=True
        if self.require_laser:
            required_items["laser"] = laser

        system_ok = all(item["ok"] for item in required_items.values())

        if inspection_running:
            system_text = "INSPECTION RUNNING"
        elif system_ok:
            system_text = "READY"
        else:
            system_text = "NOT READY"

        return {
            "timestamp": time.time(),
            "system_ok": system_ok,
            "system_text": system_text,
            "mode": mode,
            "items": items,
            "required_items": required_items,
        }