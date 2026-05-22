# src/COMMON/plc_gui_commands.py

import time
import threading
from pathlib import Path

from PyQt5.QtCore import QObject, pyqtSignal

try:
    from snap7 import Client
except Exception:
    Client = None


def _load_env_file(env_path):
    data = {}
    try:
        env_path = Path(env_path)
        if env_path.exists():
            with open(env_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    key, value = line.split("=", 1)
                    data[key.strip()] = value.strip().strip('"').strip("'")
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


class PlcGuiCommandService(QObject):
    started = pyqtSignal(str, str)
    success = pyqtSignal(str, str)
    error = pyqtSignal(str, str, str)
    busy_changed = pyqtSignal(bool)

    def __init__(self, env_path, parent=None):
        super().__init__(parent)
        self.env_path = env_path
        self.env = _load_env_file(env_path)
        self._lock = threading.Lock()
        self._busy = False

    def reload_env(self):
        self.env = _load_env_file(self.env_path)

    def _read_bit(self, client, db_number, byte_index, bit_index):
        data = client.db_read(db_number, byte_index, 1)
        return bool(data[0] & (1 << bit_index))

    def _write_bit(self, client, db_number, byte_index, bit_index, value):
        data = client.db_read(db_number, byte_index, 1)
        byte_val = data[0]

        if value:
            byte_val = byte_val | (1 << bit_index)
        else:
            byte_val = byte_val & ~(1 << bit_index)

        client.db_write(db_number, byte_index, bytes([byte_val]))

    def _pulse_command(self, command_name, db, byte, bit):
        address = f"DB{db}.DBX{byte}.{bit}"
        client = None

        try:
            if Client is None:
                raise RuntimeError("python-snap7 is not installed or not available.")

            self.reload_env()

            deployment = str(self.env.get("DEPLOYMENT", "False")).strip()
            if deployment != "True":
                raise RuntimeError("DEPLOYMENT is not True. PLC command skipped.")

            plc_ip = self.env.get("PLC_IP", "").strip()
            if not plc_ip:
                raise RuntimeError("PLC_IP missing in .env.")

            rack = _env_int(self.env, "PLC_RACK", 0)
            slot = _env_int(self.env, "PLC_SLOT", 1)
            pulse_sec = _env_float(self.env, "PLC_GUI_COMMAND_PULSE_SEC", 0.5)

            self.started.emit(command_name, address)

            client = Client()
            client.connect(plc_ip, rack, slot)

            if hasattr(client, "get_connected") and not client.get_connected():
                raise RuntimeError(f"PLC connection failed: {plc_ip}")

            self._write_bit(client, db, byte, bit, True)
            time.sleep(0.05)

            read_true = self._read_bit(client, db, byte, bit)

            print(
                f"[PLC][GUI_CMD] {command_name} TRUE -> {address} "
                f"read_back={read_true}"
            )

            if not read_true:
                raise RuntimeError(f"TRUE write failed at {address}")

            time.sleep(max(0.1, pulse_sec))

            self._write_bit(client, db, byte, bit, False)
            time.sleep(0.05)

            read_false = self._read_bit(client, db, byte, bit)

            print(
                f"[PLC][GUI_CMD] {command_name} FALSE -> {address} "
                f"read_back={read_false}"
            )

            if read_false:
                raise RuntimeError(f"FALSE reset failed at {address}")

            self.success.emit(command_name, address)

        except Exception as e:
            self.error.emit(command_name, address, str(e))

        finally:
            try:
                if client is not None:
                    client.disconnect()
            except Exception:
                pass

            with self._lock:
                self._busy = False

            self.busy_changed.emit(False)

    def pulse_command(self, command_name, db, byte, bit):
        with self._lock:
            if self._busy:
                self.error.emit(command_name, f"DB{db}.DBX{byte}.{bit}", "Another PLC command is already running.")
                return

            self._busy = True

        self.busy_changed.emit(True)

        t = threading.Thread(
            target=self._pulse_command,
            args=(command_name, db, byte, bit),
            daemon=True,
        )
        t.start()

    def pulse_auto_start(self):
        self.reload_env()

        db = _env_int(self.env, "PLC_AUTO_START_GUI_DB", 74)
        byte = _env_int(self.env, "PLC_AUTO_START_GUI_BYTE", 87)
        bit = _env_int(self.env, "PLC_AUTO_START_GUI_BIT", 1)

        self.pulse_command(
            command_name="AUTO START FROM GUI",
            db=db,
            byte=byte,
            bit=bit,
        )

    def pulse_all_servo_reset(self):
        self.reload_env()

        db = _env_int(self.env, "PLC_ALL_SERVO_RESET_GUI_DB", 74)
        byte = _env_int(self.env, "PLC_ALL_SERVO_RESET_GUI_BYTE", 87)
        bit = _env_int(self.env, "PLC_ALL_SERVO_RESET_GUI_BIT", 2)

        self.pulse_command(
            command_name="ALL SERVO RESET FROM GUI",
            db=db,
            byte=byte,
            bit=bit,
        )