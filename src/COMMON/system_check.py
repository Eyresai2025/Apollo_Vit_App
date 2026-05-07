# src/COMMON/system_check.py

import os
import sys
import platform
import subprocess
import shutil
from datetime import datetime
from pathlib import Path


class SystemChecker:
    """
    Startup system checker.

    IMPORTANT:
    - Does NOT load AI models.
    - Does NOT connect cameras.
    - Does NOT connect PLC.
    - Does NOT connect laser.
    - Only checks availability and displays startup information.
    """

    def __init__(self, media_path, env_path=None):
        self.media_path = Path(media_path)
        self.env_path = Path(env_path) if env_path else self.media_path.parent / ".env"
        self.env = self._load_env_file(self.env_path)

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

    # ------------------------------------------------------------
    # VERSION
    # ------------------------------------------------------------
    def get_version_info(self):
        version_file = self.media_path / "version.txt"

        info = {
            "app_version": self.env.get("APP_VERSION", "v1.0"),
            "build_number": self.env.get("BUILD_NUMBER", datetime.now().strftime("%Y%m%d")),
            "build_date": self.env.get("BUILD_DATE", datetime.now().strftime("%Y-%m-%d")),
            "python_version": sys.version.split()[0],
        }

        if version_file.exists():
            try:
                with open(version_file, "r", encoding="utf-8") as f:
                    for line in f:
                        if "=" in line:
                            key, value = line.strip().split("=", 1)
                            info[key.strip()] = value.strip()
            except Exception:
                pass

        return info

    # ------------------------------------------------------------
    # SYSTEM
    # ------------------------------------------------------------
    def check_system_requirements(self):
        result = {
            "OS": f"{platform.system()} {platform.release()}",
            "Architecture": platform.machine(),
            "CPU Cores": os.cpu_count(),
            "RAM (GB)": self._get_ram_gb(),
            "Free Disk (GB)": self._get_free_disk_gb(),
            "GPU": "Not Available",
            "CUDA Version": "Not Available",
            "GPU Memory (GB)": "Not Available",
        }

        try:
            import torch

            if torch.cuda.is_available():
                result["GPU"] = torch.cuda.get_device_name(0)
                result["CUDA Version"] = str(torch.version.cuda)
                result["GPU Memory (GB)"] = f"{torch.cuda.get_device_properties(0).total_memory / (1024 ** 3):.1f}"
        except Exception as e:
            result["GPU"] = f"Check failed: {e}"

        return result

    def _get_ram_gb(self):
        try:
            import psutil
            return f"{psutil.virtual_memory().total / (1024 ** 3):.1f}"
        except Exception:
            return "Unknown"

    def _get_free_disk_gb(self):
        try:
            usage = shutil.disk_usage(str(self.media_path.drive or self.media_path.anchor or "."))
            return f"{usage.free / (1024 ** 3):.1f}"
        except Exception:
            return "Unknown"


    # ------------------------------------------------------------
    # LUCID ARENA SDK CHECK - NOT CAMERA CONNECTION
    # ------------------------------------------------------------
    def check_lucid_arena_sdk(self):
        try:
            from arena_api.system import system  # noqa

            return {
                "ok": True,
                "message": "Lucid Arena SDK Python package found.",
            }
        except Exception as e:
            return {
                "ok": False,
                "message": f"Lucid Arena SDK not available: {e}",
            }

    # ------------------------------------------------------------
    # TELEDYNE / SAPERA SOFTWARE CHECK - NOT LASER CONNECTION
    # ------------------------------------------------------------
    def check_teledyne_sapera_software(self):
        env_camexpert = self.env.get("SAPERA_CAMEXPERT_PATH", "")

        possible_paths = []

        if env_camexpert:
            possible_paths.append(env_camexpert)

        possible_paths.extend([
            r"C:\Program Files\Teledyne DALSA\Sapera\CamExpert\CamExpert.exe",
            r"C:\Program Files\Teledyne DALSA\Sapera LT\CamExpert\CamExpert.exe",
            r"C:\Program Files\Teledyne DALSA\Sapera\Bin",
            r"C:\Program Files\Teledyne DALSA\Sapera LT\Bin",
        ])

        found = [p for p in possible_paths if p and os.path.exists(p)]

        if found:
            return {
                "ok": True,
                "message": "Teledyne/Sapera software found.",
                "paths": found,
            }

        return {
            "ok": False,
            "message": "Teledyne/Sapera software path not found.",
            "paths": [],
        }

    # ------------------------------------------------------------
    # PLC NETWORK PING ONLY - NOT PLC CONNECT
    # ------------------------------------------------------------
    def check_plc_network(self):
        deployment = self.env.get("DEPLOYMENT", "False")
        plc_ip = self.env.get("PLC_IP", "")

        result = {
            "deployment": deployment,
            "plc_ip": plc_ip,
            "plc_ping_ok": False,
            "message": "Not checked",
        }

        if str(deployment) != "True":
            result["message"] = "DEPLOYMENT=False. PLC ping skipped."
            return result

        if not plc_ip:
            result["message"] = "PLC_IP missing in .env."
            return result

        ok = self._ping_host(plc_ip)
        result["plc_ping_ok"] = ok
        result["message"] = "PLC IP reachable." if ok else "PLC IP not reachable."

        return result

    def _ping_host(self, host, timeout=1):
        try:
            param = "-n" if platform.system().lower() == "windows" else "-c"
            cmd = ["ping", param, "1", host]

            completed = subprocess.run(
                cmd,
                timeout=timeout,
                capture_output=True,
                text=True,
            )

            return completed.returncode == 0

        except Exception:
            return False

    # ------------------------------------------------------------
    # MAIN
    # ------------------------------------------------------------
    def run_startup_checks(self):
        return {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "version": self.get_version_info(),
            "system": self.check_system_requirements(),
            "lucid_arena": self.check_lucid_arena_sdk(),
            "sapera": self.check_teledyne_sapera_software(),
            "plc_network": self.check_plc_network(),
        }


def format_startup_popup_message(result):
    version = result.get("version", {})
    system_info = result.get("system", {})
    lucid = result.get("lucid_arena", {})
    sapera = result.get("sapera", {})
    plc = result.get("plc_network", {})

    message = (
        "Application Startup Check Completed\n\n"
        f"Version: {version.get('app_version', 'Unknown')}\n"
        f"Build Number: {version.get('build_number', 'Unknown')}\n"
        f"Build Date: {version.get('build_date', 'Unknown')}\n"
        f"Python: {version.get('python_version', 'Unknown')}\n\n"

        f"OS: {system_info.get('OS', 'Unknown')}\n"
        f"CPU Cores: {system_info.get('CPU Cores', 'Unknown')}\n"
        f"RAM: {system_info.get('RAM (GB)', 'Unknown')} GB\n"
        f"Free Disk: {system_info.get('Free Disk (GB)', 'Unknown')} GB\n"
        f"GPU: {system_info.get('GPU', 'Unknown')}\n"
        f"CUDA: {system_info.get('CUDA Version', 'Unknown')}\n\n"

        f"Lucid Arena SDK: {'OK' if lucid.get('ok') else 'CHECK REQUIRED'}\n"
        f"Teledyne/Sapera Software: {'OK' if sapera.get('ok') else 'CHECK REQUIRED'}\n"
        f"PLC Network: {plc.get('message', 'Not checked')}\n\n"

        "Important:\n"
        "No AI models were loaded.\n"
        "No camera was connected.\n"
        "No PLC connection was opened.\n"
        "No laser connection was opened.\n\n"

        "Next Step:\n"
        "Open Test Mode and run Full Hardware Check."
    )

    return message


def show_startup_system_popup(parent, media_path, env_path=None):
    """
    Call this directly from GUI.
    """
    from PyQt5.QtWidgets import QMessageBox

    try:
        checker = SystemChecker(media_path=media_path, env_path=env_path)
        result = checker.run_startup_checks()
        message = format_startup_popup_message(result)

        QMessageBox.information(
            parent,
            "Startup System Check",
            message,
        )

        return result

    except Exception as e:
        QMessageBox.warning(
            parent,
            "Startup Check Error",
            f"Startup system check failed:\n\n{e}",
        )
        return None