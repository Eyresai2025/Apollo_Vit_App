# src/COMMON/axis_status_service.py

import struct
from pathlib import Path
from typing import Any, Dict, List, Optional

from bson import ObjectId  # type: ignore
from src.COMMON.recipe_tag_map import RECIPE_TARGETS
from src.COMMON.db import get_collection


class AxisStatusService:
    """
    Read-only Axis Status service.

    Current production-safe flow WITHOUT PLC active SKU tag:
    - DB74 = live actual machine values/status.
    - DB75.DBD0 to DB75.DBD284 = current/running servo recipe values.
    - MongoDB Active Recipe = last recipe this application loaded to PLC.
    - MongoDB SKU Recipes = full saved recipe values.

    Important:
    - DB75.DBW288 is NOT active recipe read tag.
    - DB75.DBW288 is recipe number ENTRY/WRITE tag.
    - This service does NOT write anything to PLC.
    """

    def __init__(self, media_path: str, env_path: Optional[str] = None):
        self.media_path = Path(media_path)
        self.env_path = Path(env_path) if env_path else self.media_path.parent / ".env"
        self.env = self._load_env_file(self.env_path)

        self.deployment = self.env.get("DEPLOYMENT", "False")
        self.refresh_ms = self._env_int("AXIS_STATUS_REFRESH_MS", 1000)

        self.recipe_col = get_collection("SKU Recipes")
        self.active_recipe_col = get_collection("Active Recipe")
        # PLC active running recipe number.
        # PLC confirmed this tag is working.
        self.active_recipe_db = self._env_int("PLC_ACTIVE_RECIPE_DB", 74)
        self.active_recipe_byte = self._env_int("PLC_ACTIVE_RECIPE_BYTE", 78)
        self.active_recipe_type = self._env_str("PLC_ACTIVE_RECIPE_TYPE", "INT").upper()
        # DB75 running servo recipe values config.
        # Do NOT read DB75.DBW288 here.
        self.running_recipe_db = self._env_int("ACTIVE_RECIPE_DB", 75)
        self.running_recipe_value_type = self._env_str("ACTIVE_RECIPE_VALUE_TYPE", "REAL").upper()
        self.running_recipe_axis_block_bytes = self._env_int("ACTIVE_RECIPE_AXIS_BLOCK_BYTES", 24)
        self.running_recipe_work1_offset = self._env_int("ACTIVE_RECIPE_WORK1_OFFSET", 4)
        self.running_recipe_work2_offset = self._env_int("ACTIVE_RECIPE_WORK2_OFFSET", 8)
        self.running_recipe_tolerance = self._env_float("ACTIVE_RECIPE_TOLERANCE", 1.0)

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
        return str(val).strip().strip('"').strip("'")

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

    def _env_float_optional(self, key: str):
        try:
            val = self.env.get(key, "")
            if val is None or str(val).strip() == "":
                return None
            return float(str(val).strip())
        except Exception:
            return None

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

        try:
            return client.db_read(int(db), int(byte), int(size))
        except Exception:
            return None

    def _read_bool(self, client, db: int, byte: int, bit: int):
        try:
            data = self._read_bytes(client, db, byte, 1)
            if data is None or len(data) < 1:
                return None

            return bool(data[0] & (1 << int(bit)))
        except Exception:
            return None

    def _read_number(self, client, db: int, byte: int, dtype: str):
        """
        Siemens values are big-endian.
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

    # ------------------------------------------------------------
    # AXIS CONFIG FROM .env
    # ------------------------------------------------------------
    def _axis_cfg(self, axis_id: int) -> Dict[str, Any]:
        p = f"AXIS_{axis_id}_"

        return {
            "axis_id": axis_id,
            "axis_key": f"axis_{axis_id:02d}",
            "name": self._env_str(p + "NAME", f"Axis {axis_id}"),

            # DB74 live position
            "pos_db": self._env_int(p + "POS_DB", 0),
            "pos_byte": self._env_int(p + "POS_BYTE", 0),
            "pos_type": self._env_str(p + "POS_TYPE", "REAL").upper(),

            # DB74 enabled/fault/home bits
            "enabled_configured": (p + "ENABLED_DB") in self.env,
            "enabled_db": self._env_int(p + "ENABLED_DB", 0),
            "enabled_byte": self._env_int(p + "ENABLED_BYTE", 0),
            "enabled_bit": self._env_int(p + "ENABLED_BIT", 0),

            "homed_configured": (p + "HOMED_DB") in self.env,
            "homed_db": self._env_int(p + "HOMED_DB", 0),
            "homed_byte": self._env_int(p + "HOMED_BYTE", 0),
            "homed_bit": self._env_int(p + "HOMED_BIT", 0),

            "fault_configured": (p + "FAULT_DB") in self.env,
            "fault_db": self._env_int(p + "FAULT_DB", 0),
            "fault_byte": self._env_int(p + "FAULT_BYTE", 0),
            "fault_bit": self._env_int(p + "FAULT_BIT", 0),

            "tolerance": self._env_float(p + "TOLERANCE", self.running_recipe_tolerance),
        }

    def _position_sort_rank(self, position: str) -> int:
        """
        Sort Axis Status rows by position group:
        HOME first, then WORK 1, WORK 2, WORK 3, WORK 4, SAFE.
        """
        p = str(position or "").upper().strip()
        p = p.replace("_", " ").replace("-", " ")
        p = " ".join(p.split())

        order = {
            "HOME": 0,
            "WORK 1": 1,
            "WORK1": 1,
            "WORK 2": 2,
            "WORK2": 2,
            "WORK 3": 3,
            "WORK3": 3,
            "WORK 4": 4,
            "WORK4": 4,
            "SAFE": 5,
        }

        return order.get(p, 99)
    def _read_axis_live_state(self, client, axis_id: int) -> Dict[str, Any]:
        cfg = self._axis_cfg(axis_id)

        position = None
        enabled = None
        homed = None
        fault = None

        if str(self.deployment) == "True":
            if client is not None:
                position = self._read_number(
                    client,
                    cfg["pos_db"],
                    cfg["pos_byte"],
                    cfg["pos_type"],
                )

                if cfg.get("enabled_configured"):
                    enabled = self._read_bool(
                        client,
                        cfg["enabled_db"],
                        cfg["enabled_byte"],
                        cfg["enabled_bit"],
                    )

                if cfg.get("homed_configured"):
                    homed = self._read_bool(
                        client,
                        cfg["homed_db"],
                        cfg["homed_byte"],
                        cfg["homed_bit"],
                    )

                if cfg.get("fault_configured"):
                    fault = self._read_bool(
                        client,
                        cfg["fault_db"],
                        cfg["fault_byte"],
                        cfg["fault_bit"],
                    )
        else:
            # Demo values
            position = self._env_float_optional(f"AXIS_{axis_id}_RECIPE_POS")
            enabled = True
            homed = True
            fault = False

        return {
            "axis_id": axis_id,
            "axis_key": cfg["axis_key"],
            "axis_name": cfg["name"],
            "live_position": position,
            "enabled": enabled,
            "homed": homed,
            "fault": fault,
            "tolerance": cfg["tolerance"],
            "db74_address": f'DB{cfg["pos_db"]}.DBD{cfg["pos_byte"]}',
        }

    # ------------------------------------------------------------
    # RECIPE TARGET CONFIG FROM .env
    # ------------------------------------------------------------
    def _recipe_target_configs(self) -> List[Dict[str, Any]]:
        """
        Load recipe target config from central recipe_tag_map.py.

        This supports:
            HOME
            WORK 1
            WORK 2
            WORK 3
            WORK 4
            SAFE

        Instead of only old 17 target rows.
        """
        targets: List[Dict[str, Any]] = []

        for idx, item in enumerate(RECIPE_TARGETS, start=1):
            axis_id = int(item.get("axis_id", 0))

            if axis_id <= 0:
                continue

            targets.append({
                "target_index": idx,
                "target_key": item["key"],
                "legacy_key": item.get("legacy_key"),

                "target_name": f"{item.get('sd', '')} {item.get('description', '')}".strip(),
                "group": str(item.get("group", "MACHINE")).upper(),
                "position": item.get("position", "-"),

                "axis_id": axis_id,
                "axis_key": f"axis_{axis_id:02d}",
                "type": item.get("db75_type", "REAL").upper(),

                "db75_read_db": int(item.get("db75_db", self.running_recipe_db)),
                "db75_read_byte": int(item.get("db75_byte", -1)),
                "db75_type": item.get("db75_type", "REAL").upper(),

                "db53_write_db": int(item.get("db53_db", 53)),
                "db53_write_byte": int(item.get("db53_byte", -1)),
                "db53_type": item.get("db53_type", "REAL").upper(),
            })

        return targets

    # ------------------------------------------------------------
    # DB75 RUNNING SERVO VALUES READ
    # ------------------------------------------------------------
    def _db75_byte_for_target(self, axis_id: int, group: str) -> int:
        """
        DB75 layout:
            each axis block = 24 bytes
            HOME   = base + 0
            WORK 1 = base + 4
            WORK 2 = base + 8
            WORK 3 = base + 12
            WORK 4 = base + 16
            SAFE   = base + 20

        Rule used by current recipe targets:
            MACHINE/CAMERA = WORK 1
            LASER          = WORK 2
        """
        base = (int(axis_id) - 1) * int(self.running_recipe_axis_block_bytes)

        if str(group).upper() == "LASER":
            return base + int(self.running_recipe_work2_offset)

        return base + int(self.running_recipe_work1_offset)

    def _read_db75_running_value(self, client, target_cfg: Dict[str, Any]):
        """
        Read active/running recipe value from explicit DB75 address.

        DB75 byte comes from recipe_tag_map.py.
        No more MACHINE/CAMERA=WORK1 and LASER=WORK2 assumption.
        """
        if str(self.deployment) != "True":
            return None, ""

        if client is None:
            return None, ""

        db = int(target_cfg.get("db75_read_db", self.running_recipe_db))
        byte = int(target_cfg.get("db75_read_byte", -1))
        dtype = str(target_cfg.get("db75_type", "REAL")).upper()

        if byte < 0:
            return None, ""

        value = self._read_number(
            client,
            db,
            byte,
            dtype,
        )

        if dtype == "REAL":
            address = f"DB{db}.DBD{byte}"
        else:
            address = f"DB{db}.DBW{byte}"

        return value, address

    # ------------------------------------------------------------
    # MONGODB ACTIVE / LAST LOADED RECIPE
    # ------------------------------------------------------------
    def _get_last_loaded_recipe_state(self) -> Optional[Dict[str, Any]]:
        """
        Reads application-side last loaded recipe state.

        This comes from Active Recipe collection:
            type = last_loaded_recipe

        This is NOT PLC active SKU.
        It only tells which recipe the application last loaded to PLC.
        """
        try:
            return self.active_recipe_col.find_one({"type": "last_loaded_recipe"})
        except Exception:
            return None

    def _get_recipe_from_last_loaded_state(
        self,
        state: Optional[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """
        Fetch full recipe from SKU Recipes using Active Recipe pointer.
        Preferred lookup:
            recipe_id

        Fallback:
            sku_name + version
            recipe_number + version/latest
        """
        if not state:
            return None

        recipe_id = str(state.get("recipe_id", "")).strip()

        if recipe_id:
            try:
                recipe = self.recipe_col.find_one({"_id": ObjectId(recipe_id)})
                if recipe:
                    return recipe
            except Exception:
                pass

        sku_name = str(state.get("sku_name", "")).strip()
        recipe_version = state.get("recipe_version")

        if sku_name and recipe_version not in (None, ""):
            try:
                recipe = self.recipe_col.find_one(
                    {
                        "type": "sku_recipe",
                        "sku_name": sku_name,
                        "version": int(recipe_version),
                    }
                )
                if recipe:
                    return recipe
            except Exception:
                pass

        recipe_number = state.get("recipe_number") or state.get("plc_recipe_number")

        if recipe_number not in (None, ""):
            try:
                recipe = self.recipe_col.find_one(
                    {
                        "type": "sku_recipe",
                        "$or": [
                            {"recipe_number": int(recipe_number)},
                            {"plc_recipe_number": int(recipe_number)},
                        ],
                    },
                    sort=[("version", -1)],
                )
                if recipe:
                    return recipe
            except Exception:
                pass

        return None

    def _extract_target_value(self, container, key):
        if not container or not key:
            return None

        raw = container.get(key)

        if isinstance(raw, dict):
            raw = raw.get("value")

        try:
            if raw is None or raw == "":
                return None
            return float(raw)
        except Exception:
            return None


    def _get_mongo_target_value(self, recipe: Optional[Dict[str, Any]], target_cfg: Dict[str, Any]):
        """
        Reads MongoDB target value.

        Priority:
        1. New key from recipe_tag_map.py
        2. Legacy key from old 17-target setup
        3. Search in recipe_axis_targets
        4. Search in camera_axis_targets
        5. Search in laser_axis_targets
        """
        if not recipe:
            return None

        keys_to_try = [
            target_cfg.get("target_key"),
            target_cfg.get("legacy_key"),
        ]

        containers = [
            recipe.get("recipe_axis_targets", {}) or {},
            recipe.get("camera_axis_targets", {}) or {},
            recipe.get("laser_axis_targets", {}) or {},
        ]

        for key in keys_to_try:
            if not key:
                continue

            for container in containers:
                value = self._extract_target_value(container, key)
                if value is not None:
                    return value

        return None

    # ------------------------------------------------------------
    # COMPARISON / STATUS
    # ------------------------------------------------------------
    def _delta(self, a, b):
        try:
            if a is None or b is None:
                return None
            return round(float(a) - float(b), 3)
        except Exception:
            return None

    def _abs_gt(self, value, tolerance) -> bool:
        try:
            if value is None:
                return False
            return abs(float(value)) > float(tolerance)
        except Exception:
            return False

    def _calculate_status(
        self,
        live_position,
        running_db75_value,
        mongo_value,
        enabled,
        homed,
        fault,
        tolerance,
    ) -> str:
        """
        Final Status rule for Axis Status page:

        - Current Axis Position / Enabled / Homed / Fault are displayed only.
        - Status is calculated only by comparing:
            Active Recipe Value from PLC DB75
            vs
            MongoDB saved recipe value

        Required behavior:
            same value  -> OK
            different   -> RUNNING/MONGO MISMATCH
        """

        if running_db75_value is None:
            return "DB75 UNKNOWN"

        if mongo_value is None:
            return "MONGO MISSING"

        running_mongo_delta = self._delta(running_db75_value, mongo_value)

        if self._abs_gt(running_mongo_delta, tolerance):
            return "RUNNING/MONGO MISMATCH"

        return "OK"
    
    def _read_plc_active_recipe_number(self, client):
        """
        Reads active running recipe number from PLC.

        Confirmed by PLC test:
            DB74.DBW78 INT
        """
        if str(self.deployment) != "True":
            return None

        if client is None:
            return None

        return self._read_number(
            client,
            self.active_recipe_db,
            self.active_recipe_byte,
            self.active_recipe_type,
        )
    
    def _get_recipe_by_plc_active_number(self, recipe_number):
        """
        Fetch MongoDB recipe using PLC active running recipe number.

        PLC active recipe:
            DB74.DBW78 INT

        MongoDB expected fields:
            recipe_number or plc_recipe_number
        """
        if recipe_number in (None, "", "UNKNOWN"):
            return None

        try:
            return self.recipe_col.find_one(
                {
                    "type": "sku_recipe",
                    "$or": [
                        {"recipe_number": int(recipe_number)},
                        {"plc_recipe_number": int(recipe_number)},
                    ],
                },
                sort=[("version", -1)],
            )
        except Exception:
            return None
    # ------------------------------------------------------------
    # MAIN
    # ------------------------------------------------------------
    def get_axis_status(self, selected_sku: Optional[str] = None) -> Dict[str, Any]:
        """
        Axis Status comparison flow.

        Production rule:
            - PLC active recipe number from DB74.DBW78 is the source of truth.
            - Axis Status must not show last_loaded_recipe as active recipe.
            - last_loaded_recipe is only used for PLC Written / PLC Verified display
            if it matches the same PLC active recipe number.
        """
        client = self._get_plc_client()

        plc_active_recipe_number = self._read_plc_active_recipe_number(client)

        last_loaded_state = self._get_last_loaded_recipe_state()

        # Source of truth: PLC active recipe number
        recipe = self._get_recipe_by_plc_active_number(plc_active_recipe_number)

        plc_written = None
        plc_verified = None

        if recipe:
            loaded_recipe_number = plc_active_recipe_number
            loaded_sku = recipe.get("sku_name", "UNKNOWN")
            loaded_version = recipe.get("version", "-")
            active_sku = loaded_sku
            recipe_version = loaded_version
            recipe_status = "FOUND FROM PLC ACTIVE RECIPE"

            # Show written/verified only if last loaded recipe matches PLC active recipe.
            if last_loaded_state:
                state_recipe_no = (
                    last_loaded_state.get("recipe_number")
                    or last_loaded_state.get("plc_recipe_number")
                )

                try:
                    if int(state_recipe_no) == int(plc_active_recipe_number):
                        plc_written = last_loaded_state.get("plc_written")
                        plc_verified = last_loaded_state.get("plc_verified")
                except Exception:
                    pass

        else:
            # Do NOT fallback header to last_loaded_recipe.
            loaded_recipe_number = plc_active_recipe_number or "UNKNOWN"
            loaded_sku = "UNKNOWN"
            loaded_version = "-"
            active_sku = "UNKNOWN"
            recipe_version = "-"

            if plc_active_recipe_number in (None, "", "UNKNOWN"):
                recipe_status = "PLC ACTIVE RECIPE UNKNOWN"
            else:
                recipe_status = "NOT FOUND FOR PLC ACTIVE RECIPE"

            # Still show last-loaded flags only as reference if same recipe number.
            if last_loaded_state:
                state_recipe_no = (
                    last_loaded_state.get("recipe_number")
                    or last_loaded_state.get("plc_recipe_number")
                )

                try:
                    if int(state_recipe_no) == int(plc_active_recipe_number):
                        plc_written = last_loaded_state.get("plc_written")
                        plc_verified = last_loaded_state.get("plc_verified")
                except Exception:
                    pass

        rows = []

        for target_cfg in self._recipe_target_configs():
            axis_id = int(target_cfg.get("axis_id") or 0)

            if axis_id <= 0:
                continue

            live_state = self._read_axis_live_state(client, axis_id)

            running_value, db75_address = self._read_db75_running_value(
                client=client,
                target_cfg=target_cfg,
            )

            mongo_value = self._get_mongo_target_value(
                recipe=recipe,
                target_cfg=target_cfg,
            )

            tolerance = float(live_state.get("tolerance") or self.running_recipe_tolerance)

            live_running_delta = self._delta(live_state.get("live_position"), running_value)
            running_mongo_delta = self._delta(running_value, mongo_value)

            status = self._calculate_status(
                live_position=live_state.get("live_position"),
                running_db75_value=running_value,
                mongo_value=mongo_value,
                enabled=live_state.get("enabled"),
                homed=live_state.get("homed"),
                fault=live_state.get("fault"),
                tolerance=tolerance,
            )

            rows.append({
                "target_index": target_cfg["target_index"],
                "target_key": target_cfg["target_key"],
                "legacy_key": target_cfg.get("legacy_key"),
                "target_name": target_cfg["target_name"],
                "group": target_cfg["group"],
                "position": target_cfg.get("position", "-"),

                "axis_id": axis_id,
                "axis_key": target_cfg["axis_key"],
                "axis_name": live_state.get("axis_name"),

                "live_db74": live_state.get("live_position"),
                "running_db75": running_value,
                "mongo_target": mongo_value,

                "live_running_delta": live_running_delta,
                "running_mongo_delta": running_mongo_delta,

                "active_db75": running_value,
                "live_active_delta": live_running_delta,
                "active_mongo_delta": running_mongo_delta,

                "tolerance": tolerance,

                "enabled": live_state.get("enabled"),
                "homed": live_state.get("homed"),
                "fault": live_state.get("fault"),

                "db74_address": live_state.get("db74_address"),
                "db75_address": db75_address,
                "db53_address": (
                    f"DB{target_cfg.get('db53_write_db')}.DBD{target_cfg.get('db53_write_byte')}"
                    if target_cfg.get("db53_write_byte", -1) >= 0
                    else ""
                ),

                "status": status,
            })

        rows.sort(
            key=lambda row: (
                self._position_sort_rank(row.get("position", "")),
                int(row.get("target_index", 9999)),
            )
        )

        overall_ok = bool(rows) and all(row.get("status") in ("OK", "DISABLED") for row in rows)

        if recipe:
            sku_message = (
                f"PLC active recipe={plc_active_recipe_number}, "
                f"SKU={active_sku}, "
                f"version={recipe_version}. "
                f"MongoDB recipe status={recipe_status}."
            )
        else:
            sku_message = (
                f"PLC active recipe={plc_active_recipe_number}. "
                f"MongoDB recipe status={recipe_status}. "
                "Check whether this active recipe number exists in SKU Recipes."
            )

        return {
            "deployment": self.deployment,

            "loaded_recipe_number": loaded_recipe_number,
            "loaded_sku": loaded_sku,
            "loaded_recipe_version": loaded_version,
            "plc_written": plc_written,
            "plc_verified": plc_verified,

            "active_recipe_number": loaded_recipe_number,
            "active_sku": active_sku,
            "recipe_version": recipe_version,

            "plc_active_recipe_number": plc_active_recipe_number,

            "recipe_status": recipe_status,
            "recipe_found": recipe is not None,
            "overall_ok": overall_ok,
            "sku_message": sku_message,

            "sku_info": {
                "plc_raw_value": plc_active_recipe_number,
                "message": sku_message,
                "source": "PLC DB74.DBW78 active recipe",
            },

            "targets": rows,
            "axes": rows,
        }