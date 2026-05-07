# src/COMMON/axis_status_service.py

import os
import struct
from pathlib import Path
from typing import Any, Dict, List, Optional


class AxisStatusService:
    """
    Read-only Axis Status service.

    Production rules:
    - Does NOT reconnect PLC.
    - Does NOT write to PLC.
    - Does NOT move / jog / home / reset any axis.
    - Uses existing PLC client from full_hardware_check.py.
    - Shows UNKNOWN when DB addresses are not configured or PLC is unavailable.
    """

    def __init__(self, media_path: str, env_path: Optional[str] = None):
        self.media_path = Path(media_path)
        self.env_path = Path(env_path) if env_path else self.media_path.parent / ".env"
        self.env = self._load_env_file(self.env_path)

        self.deployment = self.env.get("DEPLOYMENT", "False")
        self.refresh_ms = self._env_int("AXIS_STATUS_REFRESH_MS", 1000)
        self.plc_sku_list_enabled = self.env.get("PLC_SKU_LIST_ENABLED", "False").strip().lower() in ("1", "true", "yes", "on")
        self.plc_sku_list_db = self._env_int("PLC_SKU_LIST_DB", 100)
        self.plc_sku_list_start_byte = self._env_int("PLC_SKU_LIST_START_BYTE", 20)
        self.plc_sku_list_count = self._env_int("PLC_SKU_LIST_COUNT", 10)
        self.plc_sku_list_item_size = self._env_int("PLC_SKU_LIST_ITEM_SIZE", 2)
        self.plc_sku_list_type = self.env.get("PLC_SKU_LIST_TYPE", "WORD").strip().upper()
        # PLC = read SKU from PLC, GUI = use dropdown/manual SKU
        self.deployment

        self.plc_sku_db = self._env_int("PLC_SKU_DB", 100)
        self.plc_sku_byte = self._env_int("PLC_SKU_BYTE", 10)
        self.plc_sku_size = self._env_int("PLC_SKU_SIZE", 2)
        self.plc_sku_type = self.env.get("PLC_SKU_TYPE", "WORD").strip().upper()

    # ------------------------------------------------------------
    # ENV
    # ------------------------------------------------------------
    def _load_env_file(self, env_path: Path) -> Dict[str, str]:
        data = {}

        try:
            if env_path.exists():
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

    def _env_str(self, key: str, default: str = "") -> str:
        val = self.env.get(key, "")
        if val is None or str(val).strip() == "":
            return default
        return str(val).strip()

    def _env_int(self, key: str, default: int) -> int:
        try:
            val = self.env.get(key, "")
            if val is None or str(val).strip() == "":
                return int(default)
            return int(float(str(val).strip()))
        except Exception:
            return int(default)

    def _env_float(self, key: str, default: float) -> float:
        try:
            val = self.env.get(key, "")
            if val is None or str(val).strip() == "":
                return float(default)
            return float(str(val).strip())
        except Exception:
            return float(default)

    # ------------------------------------------------------------
    # HARDWARE STATE
    # ------------------------------------------------------------
    def _get_hardware_state(self) -> Dict[str, Any]:
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

    def _get_plc_client(self):
        state = self._get_hardware_state()
        return state.get("plc_client")

    # ------------------------------------------------------------
    # PLC READ HELPERS
    # ------------------------------------------------------------
    def _read_bytes(self, client, db: int, byte: int, size: int):
        if client is None:
            return None
        return client.db_read(db, byte, size)

    def _read_bool(self, client, db: int, byte: int, bit: int):
        try:
            data = self._read_bytes(client, db, byte, 1)
            if data is None or len(data) < 1:
                return None
            return bool(data[0] & (1 << bit))
        except Exception:
            return None

    def _read_number(self, client, db: int, byte: int, dtype: str):
        """
        Siemens data is big-endian.
        Supported dtype:
        REAL, DINT, INT, WORD, BYTE
        """
        dtype = str(dtype or "REAL").strip().upper()

        try:
            if dtype == "REAL":
                data = self._read_bytes(client, db, byte, 4)
                if data is None or len(data) < 4:
                    return None
                return round(float(struct.unpack(">f", bytes(data[:4]))[0]), 3)

            if dtype == "DINT":
                data = self._read_bytes(client, db, byte, 4)
                if data is None or len(data) < 4:
                    return None
                return int.from_bytes(bytes(data[:4]), byteorder="big", signed=True)

            if dtype == "INT":
                data = self._read_bytes(client, db, byte, 2)
                if data is None or len(data) < 2:
                    return None
                return int.from_bytes(bytes(data[:2]), byteorder="big", signed=True)

            if dtype == "WORD":
                data = self._read_bytes(client, db, byte, 2)
                if data is None or len(data) < 2:
                    return None
                return int.from_bytes(bytes(data[:2]), byteorder="big", signed=False)

            if dtype == "BYTE":
                data = self._read_bytes(client, db, byte, 1)
                if data is None or len(data) < 1:
                    return None
                return int(data[0])

            return None

        except Exception:
            return None

    def _read_string(self, client, db: int, byte: int, size: int):
        """
        Supports simple ASCII string area.
        If PLC uses Siemens STRING format with max/current length bytes,
        this still tries to extract readable content safely.
        """
        try:
            data = self._read_bytes(client, db, byte, size)
            if data is None:
                return None

            raw = bytes(data)

            # Try Siemens STRING format: byte0=max_len, byte1=current_len
            if len(raw) >= 2 and raw[1] <= len(raw) - 2 and raw[1] > 0:
                possible = raw[2:2 + raw[1]].decode("ascii", errors="ignore").strip("\x00 ").strip()
                if possible:
                    return possible

            # Fallback plain ASCII
            return raw.decode("ascii", errors="ignore").strip("\x00 ").strip() or None

        except Exception:
            return None

    # ------------------------------------------------------------
    # SKU
    # ------------------------------------------------------------
    def get_available_skus(self) -> List[str]:
        """
        Axis Status SKU list rule:

        DEPLOYMENT=True:
            Try to read all available SKUs from PLC SKU list DB.
            If PLC list is not available, fallback to .env SKU_ID_* mappings.

        DEPLOYMENT=False:
            Load SKU list from .env SKU_ID_* mappings.
        """
        if str(self.deployment) == "True":
            plc_skus = self._read_sku_list_from_plc()
            if plc_skus:
                return plc_skus

        return self._get_skus_from_env_mapping()
    def _get_skus_from_env_mapping(self) -> List[str]:
        sku_names = set()

        for key, value in self.env.items():
            if key.startswith("SKU_ID_"):
                value = str(value).strip()
                if value:
                    sku_names.add(value)

        return sorted(sku_names)


    def _read_sku_list_from_plc(self) -> List[str]:
        """
        Reads all available SKU IDs/names from PLC DB list.

        Example WORD list:
            DB100.DBW20 = 1 -> SKU_001
            DB100.DBW22 = 2 -> SKU_002
            DB100.DBW24 = 3 -> SKU_003
        """
        if not self.plc_sku_list_enabled:
            return []

        client = self._get_plc_client()

        if client is None:
            return []

        skus = []

        try:
            for i in range(self.plc_sku_list_count):
                byte_offset = self.plc_sku_list_start_byte + (i * self.plc_sku_list_item_size)

                if self.plc_sku_list_type == "STRING":
                    raw_value = self._read_string(
                        client,
                        self.plc_sku_list_db,
                        byte_offset,
                        self.plc_sku_list_item_size,
                    )
                else:
                    raw_value = self._read_number(
                        client,
                        self.plc_sku_list_db,
                        byte_offset,
                        self.plc_sku_list_type,
                    )

                if raw_value in (None, "", 0):
                    continue

                if self.plc_sku_list_type == "STRING":
                    sku_name = str(raw_value).strip()
                else:
                    sku_name = self._map_sku_id_to_name(raw_value)

                if sku_name and sku_name != "UNKNOWN":
                    skus.append(sku_name)

        except Exception as e:
            print(f"[AXIS][SKU LIST][ERROR] Failed to read PLC SKU list: {e}")
            return []

        # remove duplicates, keep order
        seen = set()
        final_skus = []
        for sku in skus:
            if sku not in seen:
                seen.add(sku)
                final_skus.append(sku)

        return final_skus

    def _map_sku_id_to_name(self, sku_id: Any) -> str:
        key = f"SKU_ID_{sku_id}"
        mapped = self.env.get(key, "").strip()

        if mapped:
            return mapped

        # fallback
        return f"SKU_{sku_id}"

    def _read_sku_from_plc(self, client):
        if client is None:
            return {
                "sku_name": "UNKNOWN",
                "source": "PLC",
                "raw_value": None,
                "message": "PLC client not available",
            }

        try:
            if self.plc_sku_type == "STRING":
                value = self._read_string(
                    client,
                    self.plc_sku_db,
                    self.plc_sku_byte,
                    self.plc_sku_size,
                )
                return {
                    "sku_name": value or "UNKNOWN",
                    "source": "PLC",
                    "raw_value": value,
                    "message": "Read PLC SKU string" if value else "PLC SKU string empty/invalid",
                }

            raw_id = self._read_number(
                client,
                self.plc_sku_db,
                self.plc_sku_byte,
                self.plc_sku_type,
            )

            if raw_id is None:
                return {
                    "sku_name": "UNKNOWN",
                    "source": "PLC",
                    "raw_value": None,
                    "message": "PLC SKU ID read failed",
                }

            sku_name = self._map_sku_id_to_name(raw_id)

            return {
                "sku_name": sku_name,
                "source": "PLC",
                "raw_value": raw_id,
                "message": f"Read PLC SKU ID {raw_id}",
            }

        except Exception as e:
            return {
                "sku_name": "UNKNOWN",
                "source": "PLC",
                "raw_value": None,
                "message": f"PLC SKU read error: {e}",
            }

    def _resolve_active_sku(self, selected_sku: Optional[str], client):
        """
        Axis Status SKU rule:

        DEPLOYMENT=True:
            - PLC gives current active SKU.
            - Dropdown still allows selecting any SKU from PLC SKU list.
            - Selected dropdown SKU is used for recipe comparison.
            - PLC SKU is shown separately for reference.

        DEPLOYMENT=False:
            - Dropdown SKUs come from .env SKU_ID_* mappings.
        """

        plc_info = None

        if str(self.deployment) == "True":
            plc_info = self._read_sku_from_plc(client)
            plc_sku = plc_info.get("sku_name", "UNKNOWN")

            if selected_sku and selected_sku != "UNKNOWN":
                return {
                    "sku_name": selected_sku,
                    "source": "GUI_DROPDOWN",
                    "raw_value": selected_sku,
                    "message": f"Using dropdown SKU {selected_sku} for axis recipe comparison. PLC active SKU is {plc_sku}.",
                    "plc_sku_name": plc_sku,
                    "plc_raw_value": plc_info.get("raw_value"),
                }

            return {
                **plc_info,
                "plc_sku_name": plc_sku,
                "plc_raw_value": plc_info.get("raw_value"),
            }

        # DEPLOYMENT=False
        if selected_sku and selected_sku != "UNKNOWN":
            return {
                "sku_name": selected_sku,
                "source": "GUI_DROPDOWN",
                "raw_value": selected_sku,
                "message": f"Using dropdown SKU {selected_sku} from .env.",
                "plc_sku_name": "UNKNOWN",
                "plc_raw_value": None,
            }

        skus = self.get_available_skus()

        if skus:
            return {
                "sku_name": skus[0],
                "source": "ENV_DEFAULT",
                "raw_value": skus[0],
                "message": f"Defaulted to first .env SKU {skus[0]}.",
                "plc_sku_name": "UNKNOWN",
                "plc_raw_value": None,
            }

        return {
            "sku_name": "UNKNOWN",
            "source": "ENV",
            "raw_value": None,
            "message": "No SKU found.",
            "plc_sku_name": "UNKNOWN",
            "plc_raw_value": None,
        }

    # ------------------------------------------------------------
    # AXIS CONFIG
    # ------------------------------------------------------------
    def _axis_cfg(self, index: int) -> Dict[str, Any]:
        p = f"AXIS_{index}_"

        return {
            "index": index,
            "name": self._env_str(p + "NAME", f"Axis {index}"),

            "pos_db": self._env_int(p + "POS_DB", 120),
            "pos_byte": self._env_int(p + "POS_BYTE", (index - 1) * 20),
            "pos_type": self._env_str(p + "POS_TYPE", "REAL").upper(),

            "enabled_db": self._env_int(p + "ENABLED_DB", 120),
            "enabled_byte": self._env_int(p + "ENABLED_BYTE", 10 + ((index - 1) * 20)),
            "enabled_bit": self._env_int(p + "ENABLED_BIT", 0),

            "homed_db": self._env_int(p + "HOMED_DB", 120),
            "homed_byte": self._env_int(p + "HOMED_BYTE", 10 + ((index - 1) * 20)),
            "homed_bit": self._env_int(p + "HOMED_BIT", 1),

            "fault_db": self._env_int(p + "FAULT_DB", 120),
            "fault_byte": self._env_int(p + "FAULT_BYTE", 10 + ((index - 1) * 20)),
            "fault_bit": self._env_int(p + "FAULT_BIT", 2),

            "alarm_db": self._env_int(p + "ALARM_DB", 120),
            "alarm_byte": self._env_int(p + "ALARM_BYTE", 12 + ((index - 1) * 20)),
            "alarm_size": self._env_int(p + "ALARM_SIZE", 2),
            "alarm_type": self._env_str(p + "ALARM_TYPE", "WORD").upper(),

            "recipe_pos": self._env_float(p + "RECIPE_POS", 0.0),
            "tolerance": self._env_float(p + "TOLERANCE", 1.0),
        }

    def _read_axis(self, client, cfg: Dict[str, Any]) -> Dict[str, Any]:
        position = None
        enabled = None
        homed = None
        fault = None
        alarm = None

        read_message = "-"

        if str(self.deployment) == "True":
            if client is None:
                read_message = "PLC client not available"
            else:
                position = self._read_number(
                    client,
                    cfg["pos_db"],
                    cfg["pos_byte"],
                    cfg["pos_type"],
                )
                enabled = self._read_bool(
                    client,
                    cfg["enabled_db"],
                    cfg["enabled_byte"],
                    cfg["enabled_bit"],
                )
                homed = self._read_bool(
                    client,
                    cfg["homed_db"],
                    cfg["homed_byte"],
                    cfg["homed_bit"],
                )
                fault = self._read_bool(
                    client,
                    cfg["fault_db"],
                    cfg["fault_byte"],
                    cfg["fault_bit"],
                )
                alarm = self._read_number(
                    client,
                    cfg["alarm_db"],
                    cfg["alarm_byte"],
                    cfg["alarm_type"],
                )
                read_message = "PLC read attempted"
        else:
            # Demo values only for UI validation
            position = cfg["recipe_pos"]
            enabled = True
            homed = True
            fault = False
            alarm = 0
            read_message = "DEPLOYMENT=False demo values"

        status = self._calculate_axis_status(
            position=position,
            recipe_pos=cfg["recipe_pos"],
            tolerance=cfg["tolerance"],
            enabled=enabled,
            homed=homed,
            fault=fault,
        )

        return {
            "index": cfg["index"],
            "name": cfg["name"],

            "current_position": position,
            "recipe_position": cfg["recipe_pos"],
            "tolerance": cfg["tolerance"],

            "enabled": enabled,
            "homed": homed,
            "fault": fault,
            "alarm_code": alarm,

            "status": status,
            "message": read_message,

            "address_info": {
                "position": f'DB{cfg["pos_db"]}.DBB{cfg["pos_byte"]} ({cfg["pos_type"]})',
                "enabled": f'DB{cfg["enabled_db"]}.DBX{cfg["enabled_byte"]}.{cfg["enabled_bit"]}',
                "homed": f'DB{cfg["homed_db"]}.DBX{cfg["homed_byte"]}.{cfg["homed_bit"]}',
                "fault": f'DB{cfg["fault_db"]}.DBX{cfg["fault_byte"]}.{cfg["fault_bit"]}',
                "alarm": f'DB{cfg["alarm_db"]}.DBB{cfg["alarm_byte"]} ({cfg["alarm_type"]})',
            },
        }

    def _calculate_axis_status(
        self,
        position,
        recipe_pos,
        tolerance,
        enabled,
        homed,
        fault,
    ) -> str:
        if position is None or enabled is None or homed is None or fault is None:
            return "UNKNOWN"

        if bool(fault):
            return "FAULT"

        if not bool(enabled):
            return "DISABLED"

        if not bool(homed):
            return "NOT HOMED"

        try:
            error = abs(float(position) - float(recipe_pos))
            if error > float(tolerance):
                return "OUT OF RANGE"
        except Exception:
            return "UNKNOWN"

        return "OK"

    # ------------------------------------------------------------
    # MAIN
    # ------------------------------------------------------------
    def get_axis_status(self, selected_sku: Optional[str] = None) -> Dict[str, Any]:
        client = self._get_plc_client()
        sku_info = self._resolve_active_sku(selected_sku, client)

        axes = []
        for i in range(1, 13):
            cfg = self._axis_cfg(i)
            axes.append(self._read_axis(client, cfg))

        overall_ok = bool(axes) and all(axis["status"] == "OK" for axis in axes)

        return {
            "deployment": self.deployment,
            "sku_info": sku_info,
            "active_sku": sku_info.get("sku_name", "UNKNOWN"),
            "sku_source": sku_info.get("source", "-"),
            "sku_message": sku_info.get("message", "-"),
            "recipe_status": "LOADED FROM AXIS CONFIG",
            "overall_ok": overall_ok,
            "axes": axes,
        }