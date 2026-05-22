# src/COMMON/full_hardware_check.py

import os
import io
import time
import contextlib
from pathlib import Path
from datetime import datetime

from PyQt5.QtCore import QObject, QThread, pyqtSignal, pyqtSlot, Qt
from PyQt5.QtWidgets import QMessageBox


_HARDWARE_STATE = {
    "ready": False,
    "last_result": None,
    "plc_client": None,
    "multi_cam": None,
}


def is_hardware_ready():
    return bool(_HARDWARE_STATE.get("ready", False))


def get_hardware_state():
    return dict(_HARDWARE_STATE)


def reset_hardware_state():
    _HARDWARE_STATE["ready"] = False
    _HARDWARE_STATE["last_result"] = None
    _HARDWARE_STATE["plc_client"] = None
    _HARDWARE_STATE["multi_cam"] = None


def _load_env_file(env_path):
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


def _env_int(env, key, default):
    try:
        value = env.get(key, "")
        if value is None or str(value).strip() == "":
            return int(default)
        return int(float(str(value).strip()))
    except Exception:
        return int(default)


def _env_float(env, key, default):
    try:
        value = env.get(key, "")
        if value is None or str(value).strip() == "":
            return float(default)
        return float(str(value).strip())
    except Exception:
        return float(default)
    
def _env_bool(env, key, default=False):
    value = str(env.get(key, str(default))).strip().lower()
    return value in ("1", "true", "yes", "y", "on")


def _set_status(dot, txt, state, msg):
    colors = {
        "ok": "#2f9e44",
        "warn": "#ff9800",
        "err": "#e03131",
        "off": "#666666",
    }
    c = colors.get(state, "#666666")

    dot.setStyleSheet(f"""
        QLabel {{
            font: 900 16px 'Segoe UI';
            color: {c};
            border: none;
            background: transparent;
        }}
    """)

    txt.setStyleSheet(f"""
        QLabel {{
            font: 700 11px 'Segoe UI';
            color: {c};
            background: transparent;
            border: none;
        }}
    """)

    txt.setText(msg)


def _set_progress(test_page, state):
    if state == "running":
        color = "#ff9800"
        value = 35
        label = "System Status: RUNNING HARDWARE CHECK..."
    elif state == "ok":
        color = "#4CAF50"
        value = 100
        label = "System Status: READY FOR LIVE INSPECTION"
    else:
        color = "#e03131"
        value = 100
        label = "System Status: HARDWARE CHECK FAILED"

    test_page.pbar.setValue(value)
    test_page.pbar.setStyleSheet(f"""
        QProgressBar {{
            background:#eee;
            border-radius:5px;
            border:none;
        }}
        QProgressBar::chunk {{
            background:{color};
            border-radius:5px;
        }}
    """)
    test_page.p_label.setText(label)


class FullHardwareChecker:
    def __init__(self, media_path, light_feedback=None):
        self.media_path = Path(media_path)
        self.env_path = self.media_path.parent / ".env"
        self.env = _load_env_file(self.env_path)

        self.light_feedback = light_feedback or {}

        self.deployment = self.env.get("DEPLOYMENT", "False")
        self.plc_ip = self.env.get("PLC_IP", "")

        self.plc_type = self.env.get("PLC_TYPE", "Siemens S7-1500")
        self.plc_rack = _env_int(self.env, "PLC_RACK", 0)
        self.plc_slot = _env_int(self.env, "PLC_SLOT", 1)
        self.plc_retry_count = _env_int(self.env, "PLC_RETRY_COUNT", 3)
        self.plc_retry_delay_sec = _env_float(self.env, "PLC_RETRY_DELAY_SEC", 1.0)
        self.require_lights = _env_bool(self.env, "REQUIRE_LIGHTS", True)
        self.require_laser = _env_bool(self.env, "REQUIRE_LASER", False)
        self.app_ok_bit = {
            "db": _env_int(self.env, "APP_OK_DB", 100),
            "byte": _env_int(self.env, "APP_OK_BYTE", 0),
            "bit": _env_int(self.env, "APP_OK_BIT", 4),
        }


        self.camera_status_write_enabled = _env_bool(
            self.env,
            "CAMERA_STATUS_WRITE_ENABLED",
            True,
        )

        self.camera_status_required = _env_bool(
            self.env,
            "CAMERA_STATUS_REQUIRED_FOR_APP_OK",
            True,
        )

        self.camera_status_bits = {
            "sidewall1": {
                "db": _env_int(self.env, "PLC_SW1_CAMERA_STATUS_DB", 74),
                "byte": _env_int(self.env, "PLC_SW1_CAMERA_STATUS_BYTE", 86),
                "bit": _env_int(self.env, "PLC_SW1_CAMERA_STATUS_BIT", 1),
            },
            "sidewall2": {
                "db": _env_int(self.env, "PLC_SW2_CAMERA_STATUS_DB", 74),
                "byte": _env_int(self.env, "PLC_SW2_CAMERA_STATUS_BYTE", 86),
                "bit": _env_int(self.env, "PLC_SW2_CAMERA_STATUS_BIT", 2),
            },
            "tread": {
                "db": _env_int(self.env, "PLC_TREAD_CAMERA_STATUS_DB", 74),
                "byte": _env_int(self.env, "PLC_TREAD_CAMERA_STATUS_BYTE", 86),
                "bit": _env_int(self.env, "PLC_TREAD_CAMERA_STATUS_BIT", 3),
            },
            "innerwall": {
                "db": _env_int(self.env, "PLC_INNER_CAMERA_STATUS_DB", 74),
                "byte": _env_int(self.env, "PLC_INNER_CAMERA_STATUS_BYTE", 86),
                "bit": _env_int(self.env, "PLC_INNER_CAMERA_STATUS_BIT", 4),
            },
            "bead": {
                "db": _env_int(self.env, "PLC_BEAD_CAMERA_STATUS_DB", 74),
                "byte": _env_int(self.env, "PLC_BEAD_CAMERA_STATUS_BYTE", 86),
                "bit": _env_int(self.env, "PLC_BEAD_CAMERA_STATUS_BIT", 5),
            },
        }
    # --------------------------------------------------------
    # PLC
    # --------------------------------------------------------
    def _plc_detail_base(self):
        return {
            "plc_type": self.plc_type,
            "ip": self.plc_ip,
            "rack": self.plc_rack,
            "slot": self.plc_slot,
            "retry_count": self.plc_retry_count,
            "retry_delay_sec": self.plc_retry_delay_sec,
            "connected": False,
            "connected_on_attempt": "-",
            "last_error": "-",
        }

    def _connect_plc(self):
        if str(self.deployment) != "True":
            detail = self._plc_detail_base()
            detail["connected"] = "DEMO PASS"
            detail["demo_mode"] = True
            return True, None, "DEPLOYMENT=False. Demo PLC pass.", detail

        if not self.plc_ip:
            detail = self._plc_detail_base()
            detail["last_error"] = "PLC_IP missing in .env."
            return False, None, "PLC_IP missing in .env.", detail

        try:
            from snap7 import Client

            last_error = None

            for attempt in range(1, self.plc_retry_count + 1):
                try:
                    client = Client()
                    client.connect(self.plc_ip, self.plc_rack, self.plc_slot)

                    if client.get_connected():
                        detail = self._plc_detail_base()
                        detail["connected"] = True
                        detail["connected_on_attempt"] = attempt
                        return True, client, f"PLC connected on attempt {attempt}.", detail

                    last_error = f"Attempt {attempt}: snap7 client not connected"

                except Exception as e:
                    last_error = f"Attempt {attempt}: {e}"

                time.sleep(self.plc_retry_delay_sec)

            detail = self._plc_detail_base()
            detail["last_error"] = last_error
            return False, None, f"PLC connection failed after {self.plc_retry_count} attempts. {last_error}", detail

        except Exception as e:
            detail = self._plc_detail_base()
            detail["last_error"] = str(e)
            return False, None, f"PLC connection error: {e}", detail
        
    def _read_db_bit(self, client, db_number, byte_index, bit_index):
        data = client.db_read(db_number, byte_index, 1)
        return bool(data[0] & (1 << bit_index))

    def _write_db_bit(self, client, db_number, byte_index, bit_index, value=True):
        data = client.db_read(db_number, byte_index, 1)
        byte_val = data[0]

        if value:
            byte_val = byte_val | (1 << bit_index)
        else:
            byte_val = byte_val & ~(1 << bit_index)

        client.db_write(db_number, byte_index, bytes([byte_val]))

    def _send_application_ok_bit(self, client, checks_ok):
        address = f'DB{self.app_ok_bit["db"]}.DBX{self.app_ok_bit["byte"]}.{self.app_ok_bit["bit"]}'

        detail = {
            "address": address,
            "sent": False,
            "value_written": False,
            "read_back_value": False,
            "verified": False,
            "message": "-",
        }

        # Demo/local mode
        if str(self.deployment) != "True":
            detail["sent"] = True
            detail["value_written"] = "DEMO PASS"
            detail["read_back_value"] = "DEMO PASS"
            detail["verified"] = "DEMO PASS"
            detail["message"] = "DEPLOYMENT=False. Application OK bit demo pass."
            return True, detail

        # Production mode: if hardware checks failed, force App OK bit FALSE
        if not checks_ok:
            if client is None:
                detail["message"] = (
                    "Hardware checks failed. Application OK bit could not be cleared "
                    "because PLC client is None."
                )
                return False, detail

            try:
                db = self.app_ok_bit["db"]
                byte = self.app_ok_bit["byte"]
                bit = self.app_ok_bit["bit"]

                # 1. Write App OK bit FALSE
                self._write_db_bit(
                    client=client,
                    db_number=db,
                    byte_index=byte,
                    bit_index=bit,
                    value=False,
                )

                # 2. Small delay for PLC update cycle
                time.sleep(0.1)

                # 3. Read same bit back
                read_back = self._read_db_bit(
                    client=client,
                    db_number=db,
                    byte_index=byte,
                    bit_index=bit,
                )

                detail["sent"] = True
                detail["value_written"] = False
                detail["read_back_value"] = read_back
                detail["verified"] = not bool(read_back)

                if not read_back:
                    detail["message"] = (
                        f"Hardware checks failed. Application OK bit cleared and verified at {address}."
                    )
                    return False, detail

                detail["message"] = (
                    f"Hardware checks failed. Tried to clear Application OK bit, "
                    f"but read-back is still TRUE at {address}."
                )
                return False, detail

            except Exception as e:
                detail["message"] = (
                    f"Hardware checks failed. Failed to clear Application OK bit at {address}: {e}"
                )
                return False, detail

        if client is None:
            detail["message"] = "Application OK bit not sent because PLC client is None."
            return False, detail

        try:
            db = self.app_ok_bit["db"]
            byte = self.app_ok_bit["byte"]
            bit = self.app_ok_bit["bit"]

            # 1. Write App OK bit TRUE
            self._write_db_bit(
                client=client,
                db_number=db,
                byte_index=byte,
                bit_index=bit,
                value=True,
            )

            # 2. Small delay for PLC update cycle
            time.sleep(0.1)

            # 3. Read same bit back
            read_back = self._read_db_bit(
                client=client,
                db_number=db,
                byte_index=byte,
                bit_index=bit,
            )

            detail["sent"] = True
            detail["value_written"] = True
            detail["read_back_value"] = read_back
            detail["verified"] = bool(read_back)

            if read_back:
                detail["message"] = f"Application OK bit written and verified at {address}."
                return True, detail

            detail["message"] = f"Application OK bit write attempted but read-back is FALSE at {address}."
            return False, detail

        except Exception as e:
            detail["message"] = f"Failed to write/verify Application OK bit at {address}: {e}"
            return False, detail
    def _send_camera_status_bits(self, client, camera_status):
        """
        Writes each logical camera connection status to PLC.

        sidewall1 -> DB74.DBX86.1
        sidewall2 -> DB74.DBX86.2
        tread     -> DB74.DBX86.3
        innerwall -> DB74.DBX86.4
        bead      -> DB74.DBX86.5
        """

        detail = {
            "enabled": bool(self.camera_status_write_enabled),
            "sent": False,
            "verified": False,
            "items": [],
            "message": "-",
        }

        if not self.camera_status_write_enabled:
            detail["sent"] = True
            detail["verified"] = True
            detail["message"] = "Camera status PLC bit writing is disabled."
            return True, detail

        if str(self.deployment) != "True":
            detail["sent"] = True
            detail["verified"] = "DEMO PASS"
            detail["message"] = "DEPLOYMENT=False. Camera status bits demo pass."

            for side, addr in self.camera_status_bits.items():
                detail["items"].append({
                    "side": side,
                    "address": f"DB{addr['db']}.DBX{addr['byte']}.{addr['bit']}",
                    "value_written": "DEMO PASS",
                    "read_back_value": "DEMO PASS",
                    "verified": "DEMO PASS",
                })

            return True, detail

        if client is None:
            detail["message"] = "Camera status bits not sent because PLC client is None."
            return False, detail

        status_by_side = {
            str(item.get("side", "")).strip().lower(): bool(item.get("connected", False))
            for item in camera_status or []
        }

        try:
            all_verified = True

            for side, addr in self.camera_status_bits.items():
                value_to_write = bool(status_by_side.get(side, False))

                db = int(addr["db"])
                byte = int(addr["byte"])
                bit = int(addr["bit"])
                address = f"DB{db}.DBX{byte}.{bit}"

                self._write_db_bit(
                    client=client,
                    db_number=db,
                    byte_index=byte,
                    bit_index=bit,
                    value=value_to_write,
                )

                time.sleep(0.02)

                read_back = self._read_db_bit(
                    client=client,
                    db_number=db,
                    byte_index=byte,
                    bit_index=bit,
                )

                verified = bool(read_back) == bool(value_to_write)
                all_verified = all_verified and verified

                detail["items"].append({
                    "side": side,
                    "address": address,
                    "value_written": value_to_write,
                    "read_back_value": bool(read_back),
                    "verified": verified,
                })

                print(
                    f"[PLC][CAMERA_STATUS] {side} -> {address} "
                    f"write={value_to_write} read_back={bool(read_back)} verified={verified}"
                )

            detail["sent"] = True
            detail["verified"] = bool(all_verified)

            if all_verified:
                detail["message"] = "Camera status bits written and verified in PLC."
                return True, detail

            detail["message"] = "One or more camera status PLC bits failed read-back verification."
            return False, detail

        except Exception as e:
            detail["message"] = f"Failed to write/verify camera status PLC bits: {e}"
            return False, detail
    # --------------------------------------------------------
    # LIGHT MANUAL FEEDBACK
    # --------------------------------------------------------
    def _check_light_feedback(self):
        lights = {}
        for i in range(1, 6):
            key = f"light{i}"
            lights[key] = bool(self.light_feedback.get(key, False))

        all_ok = all(lights.values())

        detail = {
            "lights": lights,
            "all_lights_ok": all_ok,
            "source": "Operator checkbox feedback",
        }

        return all_ok, detail

    # --------------------------------------------------------
    # CAMERA
    # --------------------------------------------------------
    def _get_expected_camera_configs(self):
        try:
            from src.camera.HARDWARE_TRIGGER import (
                get_camera_role_config,
                TRIGGER_MODE,
                TRIGGER_SOURCE,
                TRIGGER_ACTIVATION,
            )

            configs = []
            for item in get_camera_role_config():
                cfg = dict(item)
                cfg["trigger_mode"] = TRIGGER_MODE
                cfg["trigger_source"] = TRIGGER_SOURCE
                cfg["trigger_activation"] = TRIGGER_ACTIVATION
                configs.append(cfg)

            return configs
        except Exception as e:
            print(f"[CAMERA][ERROR] Could not read expected camera config: {e}")
            return []

    def _print_camera_config_to_console(self, configs):
        print("\n" + "=" * 70)
        print("[TEST MODE] EXPECTED CAMERA CONFIGURATION FROM .env")
        print("=" * 70)

        for cfg in configs:
            print(
                f"side={cfg.get('side')} | serial={cfg.get('serial')} | "
                f"width={cfg.get('width')} | height={cfg.get('camera_height')} | "
                f"final_height={cfg.get('final_height')} | pixel={cfg.get('pixel_format')} | "
                f"exposure={cfg.get('exposure_time')} | gain={cfg.get('gain')} | "
                f"line_rate={cfg.get('acquisition_line_rate')} | "
                f"trigger={cfg.get('trigger_mode')} source={cfg.get('trigger_source')}"
            )

        print("=" * 70 + "\n")

    def _check_lucid_cameras(self, plc_client=None):
        expected_configs = self._get_expected_camera_configs()
        self._print_camera_config_to_console(expected_configs)

        expected_status = [
            {
                "side": cfg.get("side", ""),
                "serial": str(cfg.get("serial", "")),
                "connected": False,
                "message": "Not connected",
            }
            for cfg in expected_configs
        ]

        if str(self.deployment) != "True":
            for item in expected_status:
                item["connected"] = True
                item["message"] = "DEMO PASS"

            return True, None, "DEPLOYMENT=False. Demo camera pass.", {
                "camera_status": expected_status,
                "expected_configs": expected_configs,
                "camera_log": "DEMO PASS",
            }

        try:
            log_buffer = io.StringIO()

            with contextlib.redirect_stdout(log_buffer), contextlib.redirect_stderr(log_buffer):
                from src.camera.HARDWARE_TRIGGER import MultiCameraManager
                manager = MultiCameraManager(plc_interface=plc_client)
                manager.connect_all()

            connected_serials = set()

            for cam in manager.cameras:
                if bool(getattr(cam, "is_connected", False)):
                    connected_serials.add(str(getattr(cam, "serial_number", "")))

            camera_status = []
            for cfg in expected_configs:
                serial = str(cfg.get("serial", ""))
                ok = serial in connected_serials
                camera_status.append({
                    "side": cfg.get("side", ""),
                    "serial": serial,
                    "connected": ok,
                    "message": "Connected" if ok else "Not connected",
                })

            all_connected = all(x["connected"] for x in camera_status)

            return all_connected, manager, "Lucid camera check completed.", {
                "camera_status": camera_status,
                "expected_configs": expected_configs,
                "camera_log": log_buffer.getvalue(),
            }

        except Exception as e:
            error_msg = f"Lucid camera connection/configuration failed: {e}"
            print(f"[CAMERA][ERROR] {error_msg}")

            return False, None, error_msg, {
                "camera_status": expected_status,
                "expected_configs": expected_configs,
                "camera_log": error_msg,
            }

    # --------------------------------------------------------
    # LASER
    # --------------------------------------------------------
    def _check_teledyne_laser(self):
        sapera_path = self.env.get("SAPERA_CAMEXPERT_PATH", "")
        laser_config_file = self.env.get("LASER_CONFIG_FILE", "")

        possible_paths = []
        if sapera_path:
            possible_paths.append(sapera_path)

        possible_paths.extend([
            r"C:\Program Files\Teledyne DALSA\Sapera\CamExpert\CamExpert.exe",
            r"C:\Program Files\Teledyne DALSA\Sapera LT\CamExpert\CamExpert.exe",
            r"C:\Program Files\Teledyne DALSA\Sapera\Bin",
            r"C:\Program Files\Teledyne DALSA\Sapera LT\Bin",
        ])

        found_paths = [p for p in possible_paths if p and os.path.exists(p)]

        detail = {
            "sapera_env_path": sapera_path,
            "found_paths": found_paths,
            "laser_config_file": laser_config_file,
            "laser_config_exists": bool(laser_config_file and os.path.exists(laser_config_file)),
            "connected": False,
            "message": "",
        }

        print("\n" + "=" * 70)
        print("[TEST MODE] TELEDYNE / SAPERA LASER CHECK")
        print("=" * 70)
        print(f"SAPERA_CAMEXPERT_PATH = {sapera_path}")
        print(f"FOUND_PATHS = {found_paths}")
        print(f"LASER_CONFIG_FILE = {laser_config_file}")
        print("=" * 70 + "\n")

        if str(self.deployment) != "True":
            detail["connected"] = True
            detail["message"] = "DEMO PASS"
            return True, "DEPLOYMENT=False. Demo laser pass.", detail

        if not found_paths:
            detail["message"] = "Sapera/CamExpert path not found."
            return False, detail["message"], detail

        if not laser_config_file:
            detail["message"] = "LASER_CONFIG_FILE is empty. Laser hardware/config not verified."
            return False, detail["message"], detail

        if not os.path.exists(laser_config_file):
            detail["message"] = f"Laser config file not found: {laser_config_file}"
            return False, detail["message"], detail

        detail["connected"] = True
        detail["message"] = "Sapera path and laser config file found."
        return True, detail["message"], detail

    # --------------------------------------------------------
    # MAIN
    # --------------------------------------------------------
    def run_all_checks(self):
        result = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "deployment": str(self.deployment),
            "overall_ok": False,
            "plc_ok": False,
            "camera_ok": False,
            "laser_ok": False,
            "lights_ok": False,
            "app_ok_sent": False,
            "plc_client": None,
            "multi_cam": None,
            "messages": [],
            "details": {},
        }

        lights_ok, light_detail = self._check_light_feedback()
        result["lights_ok"] = lights_ok
        result["details"]["lights"] = light_detail

        if lights_ok:
            result["messages"].append("Operator confirmed all lights are working.")
        else:
            result["messages"].append("Operator did not confirm all lights are working.")

        plc_ok, plc_client, plc_msg, plc_detail = self._connect_plc()
        result["plc_ok"] = plc_ok
        result["plc_client"] = plc_client
        result["details"]["plc"] = plc_detail
        result["messages"].append(plc_msg)

        camera_ok, multi_cam, camera_msg, camera_detail = self._check_lucid_cameras(plc_client=plc_client)
        result["camera_ok"] = camera_ok
        result["multi_cam"] = multi_cam
        result["details"]["camera"] = camera_detail
        result["messages"].append(camera_msg)

        camera_status_bits_ok, camera_status_bits_detail = self._send_camera_status_bits(
            client=plc_client,
            camera_status=camera_detail.get("camera_status", []),
        )

        result["camera_status_bits_ok"] = camera_status_bits_ok
        result["details"]["camera_status_bits"] = camera_status_bits_detail
        result["messages"].append(camera_status_bits_detail.get("message", "-"))

        laser_ok, laser_msg, laser_detail = self._check_teledyne_laser()
        result["laser_ok"] = laser_ok
        result["details"]["laser"] = laser_detail
        result["messages"].append(laser_msg)

        lights_required_ok = result["lights_ok"] if self.require_lights else True
        laser_required_ok = result["laser_ok"] if self.require_laser else True

        camera_status_required_ok = (
            result.get("camera_status_bits_ok", False)
            if self.camera_status_required
            else True
        )

        checks_ok_before_app_bit = (
            lights_required_ok
            and result["plc_ok"]
            and result["camera_ok"]
            and camera_status_required_ok
            and laser_required_ok
        )

        if not self.require_lights:
            result["messages"].append("Light check is bypassed using REQUIRE_LIGHTS=False.")

        if not self.require_laser:
            result["messages"].append("Laser check is bypassed using REQUIRE_LASER=False.")

        app_ok_sent, app_ok_detail = self._send_application_ok_bit(
            plc_client,
            checks_ok_before_app_bit,
        )
        result["app_ok_sent"] = app_ok_sent
        result["details"]["application_ok_bit"] = app_ok_detail
        result["messages"].append(app_ok_detail.get("message", "-"))

        result["overall_ok"] = checks_ok_before_app_bit and app_ok_sent

        return result


class FullHardwareCheckWorker(QObject):
    finished = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(self, media_path, light_feedback=None):
        super().__init__()
        self.media_path = media_path
        self.light_feedback = light_feedback or {}

    @pyqtSlot()
    def run(self):
        try:
            checker = FullHardwareChecker(
                media_path=self.media_path,
                light_feedback=self.light_feedback,
            )
            result = checker.run_all_checks()
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))


def start_full_hardware_check_from_test_page(test_page, media_path):
    existing_thread = getattr(test_page, "_hardware_check_thread", None)

    if existing_thread is not None and existing_thread.isRunning():
        QMessageBox.information(
            test_page,
            "Hardware Check",
            "Full hardware check is already running.",
        )
        return

    reset_hardware_state()

    light_feedback = {}
    if hasattr(test_page, "get_light_feedback"):
        light_feedback = test_page.get_light_feedback()

    _set_status(test_page.m99_dot, test_page.m99_txt, "warn", "Checking PLC...")
    _set_status(test_page.cam_dot, test_page.cam_txt, "warn", "Checking cameras...")
    _set_status(test_page.laser_dot, test_page.laser_txt, "warn", "Checking laser...")
    _set_status(test_page.lights_dot, test_page.lights_txt, "warn", "Checking operator light feedback...")

    _set_progress(test_page, "running")

    thread = QThread(test_page)
    worker = FullHardwareCheckWorker(
        media_path=media_path,
        light_feedback=light_feedback,
    )

    worker.moveToThread(thread)
    thread.started.connect(worker.run, Qt.QueuedConnection)

    def on_finished(result):
        test_page.last_hardware_check_result = result

        _apply_result_to_test_page(test_page, result)

        _HARDWARE_STATE["ready"] = bool(result.get("overall_ok"))
        _HARDWARE_STATE["last_result"] = result
        _HARDWARE_STATE["plc_client"] = result.get("plc_client")
        _HARDWARE_STATE["multi_cam"] = result.get("multi_cam")

        # Save Test Mode result to MongoDB.
        # This must happen after result is available, but it should not block hardware state.
        try:
            if hasattr(test_page, "save_hardware_check_result_to_db"):
                test_page.save_hardware_check_result_to_db(result)
        except Exception as e:
            print(f"[TEST MODE][DB][ERROR] Failed to save result from hardware check callback: {e}")

        messages = "\n".join(result.get("messages", []))

        if result.get("overall_ok"):
            QMessageBox.information(
                test_page,
                "System Ready",
                "All checks passed.\n\nApplication OK bit sent to PLC.\nLive Inspection is now allowed.",
            )
        else:
            QMessageBox.warning(
                test_page,
                "Hardware Check Failed",
                "Live Inspection is blocked because one or more checks failed.\n\n"
                f"{messages}",
            )

        thread.quit()

    def on_error(message):
        reset_hardware_state()

        _set_status(test_page.m99_dot, test_page.m99_txt, "err", "PLC check error")
        _set_status(test_page.cam_dot, test_page.cam_txt, "err", "Camera check error")
        _set_status(test_page.laser_dot, test_page.laser_txt, "err", "Laser check error")
        _set_status(test_page.lights_dot, test_page.lights_txt, "err", "Light feedback check error")

        _set_progress(test_page, "fail")

        QMessageBox.critical(
            test_page,
            "Hardware Check Error",
            f"Full hardware check failed:\n\n{message}",
        )

        thread.quit()

    worker.finished.connect(on_finished, Qt.QueuedConnection)
    worker.error.connect(on_error, Qt.QueuedConnection)

    thread.finished.connect(worker.deleteLater)
    thread.finished.connect(thread.deleteLater)

    def cleanup():
        test_page._hardware_check_thread = None
        test_page._hardware_check_worker = None

    thread.finished.connect(cleanup)

    test_page._hardware_check_thread = thread
    test_page._hardware_check_worker = worker

    thread.start()


def _tick(ok):
    return "✅ CONNECTED" if ok else "❌ NOT CONNECTED"


def _apply_result_to_test_page(test_page, result):
    overall_ok = bool(result.get("overall_ok"))

    details = result.get("details", {})
    lights = details.get("lights", {})
    plc = details.get("plc", {})
    camera = details.get("camera", {})
    laser = details.get("laser", {})
    app_bit = details.get("application_ok_bit", {})

    # LIGHTS
    light_lines = []
    for i in range(1, 6):
        key = f"light{i}"
        ok = bool(lights.get("lights", {}).get(key, False))
        light_lines.append(f"Light {i}: {'✅ OK' if ok else '❌ NOT OK'}")

    light_lines.append("")
    light_lines.append(f"Overall Lights: {'✅ OK' if result.get('lights_ok') else '❌ NOT OK'}")

    _set_status(
        test_page.lights_dot,
        test_page.lights_txt,
        "ok" if result.get("lights_ok") else "err",
        "\n".join(light_lines),
    )

    # LASER - concise display
    laser_ok = bool(result.get("laser_ok"))
    laser_text = (
        f"Laser Status: {_tick(laser_ok)}\n"
        f"Message: {laser.get('message', '-')}"
    )

    _set_status(
        test_page.laser_dot,
        test_page.laser_txt,
        "ok" if laser_ok else "err",
        laser_text,
    )

    # CAMERA - concise display
    camera_lines = []
    for cam in camera.get("camera_status", []):
        side = cam.get("side", "-")
        serial = cam.get("serial", "-")
        ok = bool(cam.get("connected", False))
        camera_lines.append(f"{side} | Serial: {serial} | {'✅ CONNECTED' if ok else '❌ NOT CONNECTED'}")

    if not camera_lines:
        camera_lines.append("No camera mapping found.")

    _set_status(
        test_page.cam_dot,
        test_page.cam_txt,
        "ok" if result.get("camera_ok") else "err",
        "\n".join(camera_lines),
    )

    # PLC + APP OK BIT
    plc_text = (
        f"PLC Type: {plc.get('plc_type', '-')}\n"
        f"PLC IP: {plc.get('ip', '-')}\n"
        f"Connected: {'✅ YES' if result.get('plc_ok') else '❌ NO'}\n"
        f"Last Error: {plc.get('last_error', '-')}\n\n"
        f"Application OK Bit: {app_bit.get('address', '-')}\n"
        f"Bit Sent: {'✅ YES' if result.get('app_ok_sent') else '❌ NO'}\n"
        f"Message: {app_bit.get('message', '-')}"
    )

    _set_status(
        test_page.m99_dot,
        test_page.m99_txt,
        "ok" if result.get("plc_ok") and result.get("app_ok_sent") else "err",
        plc_text,
    )

    _set_progress(test_page, "ok" if overall_ok else "fail")