from __future__ import annotations

import json
import re
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, Optional, List
import struct
from src.COMMON.common import load_env
from src.COMMON.db import get_collection
from src.COMMON.recipe_tag_map import RECIPE_TARGETS
import time
try:
    import snap7  # type: ignore
    from snap7.util import (  # type: ignore
        get_real,
        get_int,
        get_dint,
        get_word,
        set_real,
    )
except Exception:
    snap7 = None
    get_real = get_int = get_dint = get_word = set_real = None


RECIPE_COLLECTION = "SKU Recipes"


def _now_iso() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _safe_name(text: str) -> str:
    text = str(text or "").strip()
    if not text:
        return "unknown_sku"
    text = re.sub(r'[<>:"/\\|?*]+', "_", text)
    text = re.sub(r"\s+", "_", text)
    text = text.strip("._")
    return text or "unknown_sku"


def _to_bool(value: Any) -> bool:
    return str(value).strip().lower() in ("1", "true", "yes", "y", "on")


def _parse_int_list(value: str, default: Optional[List[int]] = None) -> List[int]:
    default = default or []
    value = str(value or "").strip()
    if not value:
        return list(default)

    out = []
    for part in value.split(","):
        part = part.strip()
        if part.isdigit():
            out.append(int(part))

    return out or list(default)


def _env_int(env: Dict[str, Any], key: str, default: int = 0) -> int:
    try:
        value = str(env.get(key, "")).strip().strip('"').strip("'")
        if value == "":
            return int(default)
        return int(float(value))
    except Exception:
        return int(default)


def _env_float(env: Dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        value = str(env.get(key, "")).strip().strip('"').strip("'")
        if value == "":
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def _env_str(env: Dict[str, Any], key: str, default: str = "") -> str:
    value = env.get(key, default)
    if value is None:
        return str(default)
    return str(value).strip().strip('"').strip("'")


class RecipeService:
    """
    Central backend for:
    - New SKU axis teaching
    - SKU recipe save/versioning
    - Current axis live position read
    - Production recipe target configuration
    - Optional PLC recipe write

    Important production concepts:
    - AXIS_1..AXIS_12 = physical servo axes.
    - RECIPE_TARGET_1..N = recipe target rows.
      One physical axis can appear more than once with different purpose.
    """

    def __init__(
        self,
        media_path: str,
        env_path: Optional[str] = None,
        plc_client=None,
    ):
        self.media_path = Path(media_path)
        self.project_root = self.media_path.parent
        self.env_path = env_path or str(self.project_root / ".env")
        self.env = load_env(self.env_path)

        self.deployment = _to_bool(self.env.get("DEPLOYMENT", "False"))
        self.plc_client = plc_client

        self.recipe_col = get_collection(RECIPE_COLLECTION)
        self.new_sku_col = get_collection("New SKU")

        self.backup_dir = Path(
            self.env.get(
                "RECIPE_BACKUP_DIR",
                str(self.media_path / "recipe_backups"),
            )
        )

        if not self.backup_dir.is_absolute():
            self.backup_dir = self.project_root / self.backup_dir

        self.backup_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------
    # PLC CLIENT
    # ------------------------------------------------------------
    def set_plc_client(self, plc_client):
        self.plc_client = plc_client

    # ------------------------------------------------------------
    # AXIS MASTER CONFIG
    # ------------------------------------------------------------
    def get_axis_count(self) -> int:
        axis_ids = []

        for key in self.env.keys():
            m = re.match(r"AXIS_(\d+)_NAME", str(key))
            if m:
                axis_ids.append(int(m.group(1)))

        return max(axis_ids) if axis_ids else 12

    def get_axis_config(self, axis_id: int) -> Dict[str, Any]:
        """
        Physical servo axis configuration from .env.

        Example:
            AXIS_5_NAME=SIDE WALL ONE FWD REV
            AXIS_5_IP=192.168.10.15
            AXIS_5_POS_DB=74
            AXIS_5_POS_BYTE=28
            AXIS_5_POS_TYPE=REAL
        """
        return {
            "axis_id": axis_id,
            "axis_key": f"axis_{axis_id:02d}",
            "name": _env_str(self.env, f"AXIS_{axis_id}_NAME", f"Axis {axis_id}"),
            "ip": _env_str(self.env, f"AXIS_{axis_id}_IP", ""),
            "pos_db": _env_int(self.env, f"AXIS_{axis_id}_POS_DB", 0),
            "pos_byte": _env_int(self.env, f"AXIS_{axis_id}_POS_BYTE", 0),
            "pos_type": _env_str(self.env, f"AXIS_{axis_id}_POS_TYPE", "REAL").upper(),
        }

    def get_all_axis_configs(self) -> Dict[int, Dict[str, Any]]:
        return {
            axis_id: self.get_axis_config(axis_id)
            for axis_id in range(1, self.get_axis_count() + 1)
        }

    # ------------------------------------------------------------
    # LEGACY GROUPING
    # Kept only for old NewSKUPage compatibility.
    # Production target rows should use get_recipe_target_configs().
    # ------------------------------------------------------------
    def get_camera_axis_ids(self) -> List[int]:
        return _parse_int_list(
            self.env.get("CAMERA_AXIS_IDS", ""),
            [1, 2, 3, 4, 5, 6],
        )

    def get_laser_axis_ids(self) -> List[int]:
        return _parse_int_list(
            self.env.get("LASER_AXIS_IDS", ""),
            [7, 8, 9, 10, 11, 12],
        )
    
    def _position_sort_rank(self, position: str) -> int:
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
    # ------------------------------------------------------------
    # PRODUCTION RECIPE TARGET CONFIG
    # ------------------------------------------------------------
    def get_recipe_target_configs(self) -> List[Dict[str, Any]]:
        """
        Production recipe target rows from shared recipe_tag_map.py.

        This is used by:
            - New SKU Axis Teaching
            - Save Recipe
            - DB53 PLC write
            - Recipe Management later

        We do NOT use old .env RECIPE_TARGET_COUNT=17 here anymore.
        """

        targets: List[Dict[str, Any]] = []
        axis_configs = self.get_all_axis_configs()

        for idx, item in enumerate(RECIPE_TARGETS, start=1):
            axis_id = int(item.get("axis_id", 0) or 0)
            if axis_id <= 0:
                continue

            axis_cfg = axis_configs.get(axis_id, {})

            target_name = (
                f"{item.get('sd', '')} "
                f"{item.get('description', '')} "
                f"{item.get('position', '')}"
            ).strip()

            targets.append({
                "target_index": idx,
                "target_key": item.get("key", ""),
                "legacy_key": item.get("legacy_key"),

                "group": str(item.get("group", "MACHINE")).upper(),
                "position": item.get("position", ""),

                "axis_id": axis_id,
                "axis_key": f"axis_{axis_id:02d}",
                "axis_name": axis_cfg.get("name", f"Axis {axis_id}"),
                "axis_ip": axis_cfg.get("ip", ""),

                "target_name": target_name,

                "write_db": int(item.get("db53_db", 53)),
                "write_byte": int(item.get("db53_byte", -1)),
                "type": str(item.get("db53_type", "REAL")).upper(),

                "db75_db": int(item.get("db75_db", 75)),
                "db75_byte": int(item.get("db75_byte", -1)),
                "db75_type": str(item.get("db75_type", "REAL")).upper(),
            })

        targets.sort(
            key=lambda cfg: (
                self._position_sort_rank(cfg.get("position", "")),
                int(cfg.get("axis_id", 9999)),
                int(cfg.get("target_index", 9999)),
            )
        )

        return targets

    def get_recipe_target_config_map(self) -> Dict[str, Dict[str, Any]]:
        return {
            cfg["target_key"]: cfg
            for cfg in self.get_recipe_target_configs()
            if cfg.get("target_key")
        }

    # ------------------------------------------------------------
    # LIVE AXIS READ
    # ------------------------------------------------------------
    def read_current_axis_positions(self, plc_client=None) -> Dict[str, Dict[str, Any]]:
        """
        Reads current physical servo axis positions.

        DEPLOYMENT=False:
            returns AXIS_i_RECIPE_POS from .env if present.

        DEPLOYMENT=True:
            reads AXIS_i_POS_DB / AXIS_i_POS_BYTE / AXIS_i_POS_TYPE from PLC.

        Uses shared PLC client when available.
        If no client is available, creates one temporary client for the whole refresh.
        """
        result: Dict[str, Dict[str, Any]] = {}

        client = plc_client or self.plc_client
        own_client = False

        if self.deployment:
            if snap7 is None:
                raise RuntimeError("snap7 not installed")

            if client is None:
                client = snap7.client.Client()
                own_client = True
                client.connect(
                    self.env.get("PLC_IP", "192.168.10.1"),
                    int(self.env.get("PLC_RACK", "0")),
                    int(self.env.get("PLC_SLOT", "1")),
                )

        try:
            for axis_id in range(1, self.get_axis_count() + 1):
                cfg = self.get_axis_config(axis_id)
                axis_key = cfg["axis_key"]

                try:
                    value = self._read_one_axis_position(axis_id, plc_client=client)
                    status = "OK"
                except Exception as e:
                    value = None
                    status = f"ERROR: {e}"

                result[axis_key] = {
                    "axis_id": axis_id,
                    "axis_key": axis_key,
                    "name": cfg["name"],
                    "ip": cfg["ip"],
                    "value": value,
                    "status": status,
                    "source": "PLC" if self.deployment else "ENV_DEMO",
                    "pos_db": cfg["pos_db"],
                    "pos_byte": cfg["pos_byte"],
                    "pos_type": cfg["pos_type"],
                }

        finally:
            if own_client and client is not None:
                try:
                    client.disconnect()
                except Exception:
                    pass

        return result

    def _read_one_axis_position(self, axis_id: int, plc_client=None):
        if not self.deployment:
            return float(self.env.get(f"AXIS_{axis_id}_RECIPE_POS", "0.0"))

        cfg = self.get_axis_config(axis_id)

        db_no = int(cfg["pos_db"])
        byte = int(cfg["pos_byte"])
        data_type = str(cfg["pos_type"]).upper()

        if db_no <= 0:
            raise RuntimeError(f"AXIS_{axis_id}_POS_DB not configured")

        return self._read_plc_value(
            db_no=db_no,
            byte=byte,
            data_type=data_type,
            plc_client=plc_client,
        )

    def _read_plc_value(self, db_no: int, byte: int, data_type: str, plc_client=None):
        """
        Generic PLC DB read.

        Supports:
            REAL  -> 4 bytes
            INT   -> 2 bytes signed
            DINT  -> 4 bytes signed
            WORD  -> 2 bytes unsigned
            BYTE  -> 1 byte unsigned
        """

        data_type = str(data_type or "REAL").strip().upper()

        client = plc_client or self.plc_client

        if client is None:
            raise RuntimeError("PLC client is not available.")

        if data_type == "REAL":
            raw = client.db_read(int(db_no), int(byte), 4)
            return round(float(struct.unpack(">f", bytes(raw))[0]), 3)

        if data_type == "INT":
            raw = client.db_read(int(db_no), int(byte), 2)
            return int(struct.unpack(">h", bytes(raw))[0])

        if data_type == "DINT":
            raw = client.db_read(int(db_no), int(byte), 4)
            return int(struct.unpack(">i", bytes(raw))[0])

        if data_type == "WORD":
            raw = client.db_read(int(db_no), int(byte), 2)
            return int(struct.unpack(">H", bytes(raw))[0])

        if data_type == "BYTE":
            raw = client.db_read(int(db_no), int(byte), 1)
            return int(raw[0])

        raise RuntimeError(f"Unsupported PLC read type: {data_type}")

    # ------------------------------------------------------------
    # RECIPE DOC
    # ------------------------------------------------------------
    def build_recipe_doc(
        self,
        sku_meta: Dict[str, Any],
        camera_axis_targets: Optional[Dict[str, Any]] = None,
        laser_axis_targets: Optional[Dict[str, Any]] = None,
        camera_config_links: Optional[Dict[str, Any]] = None,
        laser_config_links: Optional[Dict[str, Any]] = None,
        vit_model_path: str = "",
        training_summary: Optional[Dict[str, Any]] = None,
        validation_result: Optional[Dict[str, Any]] = None,
        author: str = "operator",
        recipe_axis_targets: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        sku_name = str(
            sku_meta.get("sku_name")
            or sku_meta.get("tyre_name")
            or ""
        ).strip()

        if not sku_name:
            raise ValueError("SKU name is required before saving recipe.")

        next_version = self.get_next_version(sku_name)

        validation_result = validation_result or {}
        training_summary = training_summary or {}

        camera_axis_targets = camera_axis_targets or {}
        laser_axis_targets = laser_axis_targets or {}
        recipe_axis_targets = recipe_axis_targets or {}

        return {
            "type": "sku_recipe",
            "sku_name": sku_name,
            "sku_folder": _safe_name(sku_name),
            "version": next_version,
            "status": "DRAFT" if not validation_result.get("accepted") else "ACCEPTED",

            "tyre_name": sku_meta.get("tyre_name", ""),
            "tyre_size": sku_meta.get("tyre_size", ""),
            "tyre_outer_diameter": sku_meta.get("tyre_outer_diameter"),
            "tyre_rpm": sku_meta.get("tyre_rpm"),
            "barcode": sku_meta.get("barcode", ""),
            "barcode_pattern": sku_meta.get("barcode_pattern", ""),
            "inspection_zones": int(sku_meta.get("inspection_zones", 5)),
            "image_count_per_zone": int(sku_meta.get("image_count_per_zone", 20)),

            # Legacy fields kept for current pages/backward compatibility.
            "camera_axis_targets": camera_axis_targets,
            "laser_axis_targets": laser_axis_targets,

            # New production-grade field.
            # New SKU page will fill this after next update.
            "recipe_axis_targets": recipe_axis_targets,

            # Store target config snapshot for traceability.
            "recipe_target_config_snapshot": self.get_recipe_target_configs(),

            "camera_config_links": camera_config_links or {},
            "laser_config_links": laser_config_links or {},

            "vit_model_path": vit_model_path or "",
            "training_date": training_summary.get("training_date", _now_iso()),
            "training_summary": training_summary,

            "validation_score": validation_result.get("f1_macro"),
            "validation_result": validation_result,

            "created_at": _now_iso(),
            "updated_at": _now_iso(),
            "author": author,
        }
    def find_recipe_by_number(self, recipe_number):
        """
        Find existing SKU recipe by recipe_number / plc_recipe_number.

        Used to prevent duplicate recipe numbers.
        """
        try:
            recipe_number = int(recipe_number)
        except Exception:
            return None

        return self.recipe_col.find_one(
            {
                "type": "sku_recipe",
                "$or": [
                    {"recipe_number": recipe_number},
                    {"plc_recipe_number": recipe_number},
                ],
            },
            sort=[("version", -1)],
        )
    def get_next_version(self, sku_name: str) -> int:
        last = self.recipe_col.find_one(
            {
                "type": "sku_recipe",
                "sku_name": sku_name,
            },
            sort=[("version", -1)],
        )

        if not last:
            return 1

        return int(last.get("version", 0)) + 1

    def save_recipe(
        self,
        recipe_doc: Dict[str, Any],
        plc_client=None,
        write_to_plc: Optional[bool] = None,
    ) -> Dict[str, Any]:
        sku_name = recipe_doc["sku_name"]

        recipe_doc = dict(recipe_doc)
        recipe_doc["updated_at"] = _now_iso()
        recipe_number = (
            recipe_doc.get("recipe_number")
            or recipe_doc.get("plc_recipe_number")
            or recipe_doc.get("sku_meta", {}).get("recipe_number")
        )

        existing_recipe = self.find_recipe_by_number(recipe_number)

        if existing_recipe:
            raise ValueError(
                f"Recipe number {recipe_number} already exists for "
                f"SKU {existing_recipe.get('sku_name', 'UNKNOWN')} "
                f"version {existing_recipe.get('version', '-')}. "
                "Duplicate recipe was not saved."
            )
        inserted = self.recipe_col.insert_one(recipe_doc)

        # Important: keep inserted MongoDB _id inside recipe_doc
        # so Active Recipe / last_loaded_recipe can point to this exact recipe.
        recipe_doc["_id"] = inserted.inserted_id

        backup_path = self._save_local_backup(recipe_doc)

        plc_result = {
            "enabled": False,
            "written": False,
            "message": "PLC recipe write disabled.",
        }

        if write_to_plc is None:
            write_to_plc = _to_bool(self.env.get("RECIPE_WRITE_TO_PLC", "False"))

        if write_to_plc:
            plc_result = self.write_recipe_to_plc(
                recipe_doc,
                plc_client=plc_client,
            )

        return {
            "ok": True,
            "inserted_id": str(inserted.inserted_id),
            "sku_name": sku_name,
            "version": recipe_doc.get("version"),
            "backup_path": str(backup_path),
            "plc_result": plc_result,
        }

    def _save_local_backup(self, recipe_doc: Dict[str, Any]) -> Path:
        sku_folder = _safe_name(recipe_doc.get("sku_name", "unknown_sku"))
        version = int(recipe_doc.get("version", 1))

        sku_dir = self.backup_dir / sku_folder
        sku_dir.mkdir(parents=True, exist_ok=True)

        backup_path = sku_dir / f"{sku_folder}_recipe_v{version:03d}.json"

        clean_doc = dict(recipe_doc)
        clean_doc.pop("_id", None)

        with open(backup_path, "w", encoding="utf-8") as f:
            json.dump(clean_doc, f, indent=2, ensure_ascii=False)

        return backup_path
    
    # ------------------------------------------------------------
    # PLC RECIPE READ / VERIFY
    # ------------------------------------------------------------
    def verify_recipe_write(
        self,
        recipe_doc: Dict[str, Any],
        plc_client=None,
        tolerance: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Read DB53 values back after writing and compare with MongoDB recipe values.

        This verifies:
            recipe_axis_targets[target_key]["value"]
                ==
            PLC DB value at recipe_axis_targets[target_key]["write_db/write_byte"]

        For production:
            DB53 = recipe write/read DB.
            DB74 = live machine values.
            DB75 = active/current recipe read later.
        """

        if not self.deployment:
            return {
                "enabled": True,
                "ok": False,
                "verified": False,
                "message": "DEPLOYMENT=False, PLC read-back verification skipped.",
                "verified_count": 0,
                "mismatch_count": 0,
                "mismatches": [],
                "items": [],
            }

        if snap7 is None:
            return {
                "enabled": True,
                "ok": False,
                "verified": False,
                "message": "snap7 not installed.",
                "verified_count": 0,
                "mismatch_count": 0,
                "mismatches": [],
                "items": [],
            }

        recipe_axis_targets = recipe_doc.get("recipe_axis_targets", {}) or {}

        if not recipe_axis_targets:
            return {
                "enabled": True,
                "ok": False,
                "verified": False,
                "message": "No recipe_axis_targets found for verification.",
                "verified_count": 0,
                "mismatch_count": 0,
                "mismatches": [],
                "items": [],
            }

        if tolerance is None:
            tolerance = _env_float(self.env, "RECIPE_VERIFY_TOLERANCE", 0.01)

        target_cfg_map = self.get_recipe_target_config_map()

        own_client = False
        client = plc_client or self.plc_client

        if client is None:
            client = snap7.client.Client()
            own_client = True
            client.connect(
                self.env.get("PLC_IP", "192.168.10.1"),
                int(self.env.get("PLC_RACK", "0")),
                int(self.env.get("PLC_SLOT", "1")),
            )

        items = []
        mismatches = []

        try:
            if hasattr(client, "get_connected") and not client.get_connected():
                raise RuntimeError("PLC client is disconnected")

            for target_key, target in recipe_axis_targets.items():
                cfg = target_cfg_map.get(target_key, {})

                expected = target.get("value", None)

                if expected is None or expected == "":
                    continue

                db_no = int(
                    target.get(
                        "write_db",
                        cfg.get("write_db", self.env.get("RECIPE_PLC_DB", 53)),
                    )
                )

                byte = int(
                    target.get(
                        "write_byte",
                        cfg.get("write_byte", -1),
                    )
                )

                data_type = str(
                    target.get(
                        "type",
                        cfg.get("type", self.env.get("RECIPE_AXIS_VALUE_TYPE", "REAL")),
                    )
                ).upper()

                if db_no <= 0 or byte < 0:
                    mismatches.append({
                        "target_key": target_key,
                        "expected": expected,
                        "actual": None,
                        "db": db_no,
                        "byte": byte,
                        "reason": "invalid PLC address",
                    })
                    continue

                actual = self._read_plc_value(
                    db_no=db_no,
                    byte=byte,
                    data_type=data_type,
                    plc_client=client,
                )

                expected_f = float(expected)
                actual_f = float(actual)
                delta = actual_f - expected_f
                ok = abs(delta) <= float(tolerance)

                item = {
                    "target_key": target_key,
                    "target_name": target.get("target_name", cfg.get("target_name", "")),
                    "expected": expected_f,
                    "actual": actual_f,
                    "delta": delta,
                    "ok": ok,
                    "db": db_no,
                    "byte": byte,
                    "type": data_type,
                }

                items.append(item)

                if not ok:
                    mismatches.append(item)

            verified_count = len(items)
            mismatch_count = len(mismatches)
            ok_all = verified_count > 0 and mismatch_count == 0

            return {
                "enabled": True,
                "ok": ok_all,
                "verified": True,
                "message": (
                    f"PLC read-back verification complete. "
                    f"Verified={verified_count}, mismatches={mismatch_count}, "
                    f"tolerance={tolerance}."
                ),
                "verified_count": verified_count,
                "mismatch_count": mismatch_count,
                "tolerance": tolerance,
                "items": items,
                "mismatches": mismatches,
            }

        except Exception as e:
            return {
                "enabled": True,
                "ok": False,
                "verified": False,
                "message": str(e),
                "verified_count": len(items),
                "mismatch_count": len(mismatches),
                "mismatches": mismatches,
                "items": items,
            }

        finally:
            if own_client and client is not None:
                try:
                    client.disconnect()
                except Exception:
                    pass

    def _mark_recipe_as_last_loaded(self, recipe_doc: Dict[str, Any], plc_result: Dict[str, Any]) -> bool:
        """
        Stores the last recipe that this application loaded to PLC.

        This is NOT PLC active SKU.
        It is application-side history/state.

        Axis Status can later use this until PLC gives real active SKU tag.
        """

        try:
            active_col = get_collection("Active Recipe")

            active_col.update_one(
                {"type": "last_loaded_recipe"},
                {
                    "$set": {
                        "type": "last_loaded_recipe",
                        "sku_name": recipe_doc.get("sku_name", ""),
                        "recipe_id": str(recipe_doc.get("_id", "")),
                        "recipe_version": recipe_doc.get("version"),
                        "recipe_number": recipe_doc.get("recipe_number"),
                        "plc_recipe_number": recipe_doc.get("plc_recipe_number"),
                        "status": recipe_doc.get("status", ""),
                        "vit_model_path": recipe_doc.get("vit_model_path", ""),
                        "plc_written": plc_result.get("written", False),
                        "plc_verified": plc_result.get("verified", False),
                        "recipe_number_result": plc_result.get("recipe_number_result", {}),
                        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "source": "APPLICATION_LOADED_TO_PLC",
                    }
                },
                upsert=True,
            )

            return True

        except Exception:
            return False
        
    def _write_recipe_name_to_plc(self, client, recipe_doc: Dict[str, Any]) -> Dict[str, Any]:
        """
        Writes recipe/tyre name to PLC.

        PLC tag:
            RECIPE Name = STRING at DB53.DBX0.0

        Interpreted as Siemens STRING starting at:
            DB53, byte 0

        Siemens STRING[n] format:
            byte 0 = max length
            byte 1 = actual length
            byte 2 onward = ASCII characters

        Write-only tag: no read-back verification is done here.
        """

        enabled = _to_bool(self.env.get("RECIPE_NAME_WRITE_ENABLED", "False"))

        if not enabled:
            return {
                "enabled": False,
                "written": False,
                "verified": False,
                "recipe_name": "",
                "message": "Recipe name PLC write disabled.",
            }

        recipe_name = (
            recipe_doc.get("tyre_name")
            or recipe_doc.get("recipe_name")
            or recipe_doc.get("sku_name")
            or recipe_doc.get("sku_meta", {}).get("tyre_name")
            or recipe_doc.get("sku_meta", {}).get("sku_name")
            or ""
        )

        recipe_name = str(recipe_name).strip()

        if not recipe_name:
            return {
                "enabled": True,
                "written": False,
                "verified": False,
                "recipe_name": "",
                "message": "Recipe name is empty; DB53 string not written.",
            }

        db_no = int(self.env.get("RECIPE_NAME_WRITE_DB", "53"))
        byte = int(self.env.get("RECIPE_NAME_WRITE_BYTE", "0"))
        max_len = int(self.env.get("RECIPE_NAME_WRITE_MAX_LEN", "50"))

        try:
            # Siemens STRING[n]:
            # [max_len][actual_len][characters...]
            encoded = recipe_name.encode("ascii", errors="ignore")[:max_len]

            data = bytearray(max_len + 2)
            data[0] = max_len
            data[1] = len(encoded)
            data[2:2 + len(encoded)] = encoded

            client.db_write(db_no, byte, data)

            return {
                "enabled": True,
                "written": True,
                "verified": True,  # write-only tag, treated as OK if db_write succeeds
                "recipe_name": recipe_name,
                "db": db_no,
                "byte": byte,
                "max_len": max_len,
                "message": (
                    f"Recipe name '{recipe_name}' written to DB{db_no}.DBX{byte}.0 "
                    f"as STRING[{max_len}]."
                ),
            }

        except Exception as e:
            return {
                "enabled": True,
                "written": False,
                "verified": False,
                "recipe_name": recipe_name,
                "db": db_no,
                "byte": byte,
                "max_len": max_len,
                "message": f"Recipe name PLC write failed: {e}",
            }   
        
    def _read_plc_bit(self, client, db_no: int, byte: int, bit: int):
        """
        Read one PLC BOOL bit from DBX address.
        Example: DB53.DBX546.2
        """
        raw = client.db_read(int(db_no), int(byte), 1)
        return bool(raw[0] & (1 << int(bit)))


    def _write_plc_bit(self, client, db_no: int, byte: int, bit: int, value: bool):
        """
        Safe PLC BOOL write.

        Reads full byte, modifies only selected bit, writes full byte back.
        This avoids disturbing other bits in the same byte.
        """
        raw = client.db_read(int(db_no), int(byte), 1)
        byte_val = int(raw[0])

        if value:
            byte_val = byte_val | (1 << int(bit))
        else:
            byte_val = byte_val & ~(1 << int(bit))

        client.db_write(int(db_no), int(byte), bytes([byte_val]))


    def _pulse_recipe_save_bit(self, client) -> Dict[str, Any]:
        """
        Pulses PLC recipe save bit.

        PLC confirmed:
            RECIPE save bit = DB53.DBX546.2 BOOL

        Purpose:
            After recipe values and recipe number are written,
            PLC needs this bit TRUE to save/copy recipe internally.
        """
        enabled = _to_bool(self.env.get("RECIPE_SAVE_BIT_ENABLED", "False"))

        if not enabled:
            return {
                "enabled": False,
                "written": True,
                "verified": True,
                "message": "Recipe save bit disabled.",
            }

        db_no = int(self.env.get("RECIPE_SAVE_BIT_DB", "53"))
        byte = int(self.env.get("RECIPE_SAVE_BIT_BYTE", "546"))
        bit = int(self.env.get("RECIPE_SAVE_BIT_BIT", "2"))
        pulse_sec = float(self.env.get("RECIPE_SAVE_BIT_PULSE_SEC", "0.5"))

        try:
            # Start LOW
            self._write_plc_bit(client, db_no, byte, bit, False)
            time.sleep(0.1)

            # Pulse HIGH
            self._write_plc_bit(client, db_no, byte, bit, True)
            time.sleep(pulse_sec)

            read_true = self._read_plc_bit(client, db_no, byte, bit)

            # Reset LOW
            self._write_plc_bit(client, db_no, byte, bit, False)
            time.sleep(0.1)

            read_false = self._read_plc_bit(client, db_no, byte, bit)

            return {
                "enabled": True,
                "written": True,
                "verified": read_false is False,
                "db": db_no,
                "byte": byte,
                "bit": bit,
                "pulse_sec": pulse_sec,
                "read_true": read_true,
                "read_false": read_false,
                "message": (
                    f"Recipe save bit pulsed DB{db_no}.DBX{byte}.{bit} "
                    f"TRUE for {pulse_sec}s then reset FALSE. "
                    f"Final readback={read_false}."
                ),
            }

        except Exception as e:
            return {
                "enabled": True,
                "written": False,
                "verified": False,
                "db": db_no,
                "byte": byte,
                "bit": bit,
                "pulse_sec": pulse_sec,
                "message": f"Recipe save bit pulse failed at DB{db_no}.DBX{byte}.{bit}: {e}",
            }        
    def _write_recipe_number_to_plc(self, client, recipe_doc: Dict[str, Any]) -> Dict[str, Any]:
        """
        Writes PLC recipe number entry tag.

        PLC confirmed:
            RECIPE NUMBER = INT at DB75.DBW288

        Meaning:
            This is a recipe number ENTRY/WRITE tag.
            It is NOT active SKU / active recipe read tag.
        """

        recipe_number = (
            recipe_doc.get("recipe_number")
            or recipe_doc.get("plc_recipe_number")
            or recipe_doc.get("sku_meta", {}).get("recipe_number")
        )

        try:
            recipe_number = int(recipe_number)
        except Exception:
            return {
                "enabled": True,
                "written": False,
                "verified": False,
                "recipe_number": None,
                "actual": None,
                "message": "Recipe number missing or invalid; DB75.DBW288 not written.",
            }

        db_no = int(self.env.get("RECIPE_NUMBER_WRITE_DB", "75"))
        byte = int(self.env.get("RECIPE_NUMBER_WRITE_BYTE", "288"))
        dtype = str(self.env.get("RECIPE_NUMBER_WRITE_TYPE", "INT")).upper()

        try:
            self._write_plc_value(
                client=client,
                db_no=db_no,
                byte=byte,
                data_type=dtype,
                value=recipe_number,
            )

            actual = self._read_plc_value(
                db_no=db_no,
                byte=byte,
                data_type=dtype,
                plc_client=client,
            )

            verified = actual is not None and int(actual) == int(recipe_number)

            return {
                "enabled": True,
                "written": True,
                "verified": verified,
                "recipe_number": recipe_number,
                "actual": actual,
                "db": db_no,
                "byte": byte,
                "type": dtype,
                "message": (
                    f"Recipe number {recipe_number} written to DB{db_no}.DBW{byte}. "
                    f"Readback={actual}, verified={verified}."
                ),
            }

        except Exception as e:
            return {
                "enabled": True,
                "written": False,
                "verified": False,
                "recipe_number": recipe_number,
                "actual": None,
                "db": db_no,
                "byte": byte,
                "type": dtype,
                "message": f"Recipe number PLC write failed: {e}",
            }
    # ------------------------------------------------------------
    # PLC RECIPE WRITE
    # ------------------------------------------------------------
    def write_recipe_to_plc(
        self,
        recipe_doc: Dict[str, Any],
        plc_client=None,
    ) -> Dict[str, Any]:
        """
        Writes recipe to PLC.

        Writes:
            1. Recipe target values to DB53
            2. Recipe number to DB75.DBW288

        Verifies:
            1. DB53 target values by read-back
            2. DB75.DBW288 recipe number by read-back

        Important:
            DB74 = live actual machine values, read only.
            DB53 = recipe target write/read/verify DB.
            DB75.DBD0-DBD284 = running servo recipe values, read only.
            DB75.DBW288 = recipe number entry/write tag.
        """

        write_enabled = _to_bool(self.env.get("RECIPE_WRITE_TO_PLC", "False"))

        if not write_enabled:
            return {
                "enabled": False,
                "written": False,
                "verified": False,
                "message": "PLC recipe write disabled. Set RECIPE_WRITE_TO_PLC=True only during PLC DB53/DB75 recipe test/production.",
            }

        if not self.deployment:
            return {
                "enabled": True,
                "written": False,
                "verified": False,
                "message": "DEPLOYMENT=False, PLC write skipped.",
            }

        if snap7 is None:
            return {
                "enabled": True,
                "written": False,
                "verified": False,
                "message": "snap7 not installed.",
            }

        own_client = False
        client = plc_client or self.plc_client

        if client is None:
            client = snap7.client.Client()
            own_client = True
            client.connect(
                self.env.get("PLC_IP", "192.168.10.1"),
                int(self.env.get("PLC_RACK", "0")),
                int(self.env.get("PLC_SLOT", "1")),
            )

        try:
            if hasattr(client, "get_connected") and not client.get_connected():
                raise RuntimeError("PLC client is disconnected")

            recipe_axis_targets = recipe_doc.get("recipe_axis_targets", {}) or {}

            if recipe_axis_targets:
                recipe_name_result = self._write_recipe_name_to_plc(
                    client=client,
                    recipe_doc=recipe_doc,
                )

                write_result = self._write_recipe_targets_to_plc(
                    client=client,
                    recipe_axis_targets=recipe_axis_targets,
                )

                recipe_number_result = self._write_recipe_number_to_plc(
                    client=client,
                    recipe_doc=recipe_doc,
                )

                recipe_save_bit_result = self._pulse_recipe_save_bit(
                    client=client,
                )

                verify_result = self.verify_recipe_write(
                    recipe_doc=recipe_doc,
                    plc_client=client,
                )

                db53_written_ok = bool(write_result.get("written", False))
                db53_verify_ok = bool(verify_result.get("ok", False))

                recipe_name_enabled = bool(recipe_name_result.get("enabled", False))
                recipe_name_written_ok = (
                    True if not recipe_name_enabled
                    else bool(recipe_name_result.get("written", False))
                )

                recipe_no_written_ok = bool(recipe_number_result.get("written", False))
                recipe_no_verify_ok = bool(recipe_number_result.get("verified", False))

                save_bit_enabled = bool(recipe_save_bit_result.get("enabled", False))
                save_bit_ok = (
                    True if not save_bit_enabled
                    else bool(recipe_save_bit_result.get("written", False))
                )

                overall_written = (
                    db53_written_ok
                    and recipe_name_written_ok
                    and recipe_no_written_ok
                    and save_bit_ok
                )

                overall_verified = (
                    db53_verify_ok
                    and recipe_no_verify_ok
                    and save_bit_ok
                )

                final_result = {
                    "enabled": True,
                    "written": overall_written,
                    "verified": overall_verified,
                    "message": (
                        f"{recipe_name_result.get('message', '')} "
                        f"{write_result.get('message', '')} "
                        f"{recipe_number_result.get('message', '')} "
                        f"{recipe_save_bit_result.get('message', '')} "
                        f"{verify_result.get('message', '')}"
                    ).strip(),

                    "write_result": write_result,
                    "verify_result": verify_result,
                    "recipe_name_result": recipe_name_result,
                    "recipe_number_result": recipe_number_result,
                    "recipe_save_bit_result": recipe_save_bit_result,

                    "written_items": write_result.get("written_items", []),
                    "skipped_items": write_result.get("skipped_items", []),
                    "mismatches": verify_result.get("mismatches", []),

                    "db53_written": db53_written_ok,
                    "db53_verified": db53_verify_ok,
                    "recipe_name_written": recipe_name_written_ok,
                    "recipe_number_written": recipe_no_written_ok,
                    "recipe_number_verified": recipe_no_verify_ok,
                    "recipe_save_bit_written": save_bit_ok,
                }

                if overall_written:
                    self._mark_recipe_as_last_loaded(recipe_doc, final_result)

                return final_result

            # Legacy fallback. This writes old camera/laser groups.
            # Recipe number will still be written if recipe number is present.
            legacy_result = self._write_legacy_axis_targets_to_plc(
                client=client,
                recipe_doc=recipe_doc,
            )

            recipe_name_result = self._write_recipe_name_to_plc(
                client=client,
                recipe_doc=recipe_doc,
            )

            recipe_number_result = self._write_recipe_number_to_plc(
                client=client,
                recipe_doc=recipe_doc,
            )

            recipe_save_bit_result = self._pulse_recipe_save_bit(
                client=client,
            )
            legacy_written = bool(legacy_result.get("written", False))
            recipe_no_written = bool(recipe_number_result.get("written", False))
            recipe_no_verified = bool(recipe_number_result.get("verified", False))

            recipe_name_enabled = bool(recipe_name_result.get("enabled", False))
            recipe_name_written = (
                True if not recipe_name_enabled
                else bool(recipe_name_result.get("written", False))
            )

            save_bit_enabled = bool(recipe_save_bit_result.get("enabled", False))
            save_bit_ok = (
                True if not save_bit_enabled
                else bool(recipe_save_bit_result.get("written", False))
            )

            final_result = {
                "enabled": True,
                "written": legacy_written and recipe_name_written and recipe_no_written and save_bit_ok,
                "verified": recipe_no_verified and save_bit_ok,
                "message": (
                    f"{recipe_name_result.get('message', '')} "
                    f"{legacy_result.get('message', '')} "
                    f"{recipe_number_result.get('message', '')} "
                    f"{recipe_save_bit_result.get('message', '')} "
                    "Verification skipped for legacy recipe target format."
                ).strip(),
                "write_result": legacy_result,
                "verify_result": {
                    "enabled": True,
                    "ok": False,
                    "verified": False,
                    "message": "Verification skipped for legacy recipe format.",
                },
                "recipe_number_result": recipe_number_result,
                "recipe_number_written": recipe_no_written,
                "recipe_number_verified": recipe_no_verified,
                "recipe_save_bit_result": recipe_save_bit_result,
                "recipe_save_bit_written": save_bit_ok,
            }

            if final_result["written"]:
                self._mark_recipe_as_last_loaded(recipe_doc, final_result)

            return final_result

        except Exception as e:
            return {
                "enabled": True,
                "written": False,
                "verified": False,
                "message": str(e),
            }

        finally:
            if own_client and client is not None:
                try:
                    client.disconnect()
                except Exception:
                    pass

    def _write_recipe_targets_to_plc(
        self,
        client,
        recipe_axis_targets: Dict[str, Any],
    ) -> Dict[str, Any]:
        written_items = []
        skipped_items = []

        target_cfg_map = self.get_recipe_target_config_map()

        for target_key, target in recipe_axis_targets.items():
            cfg = target_cfg_map.get(target_key, {})

            value = target.get("value", None)
            if value is None or value == "":
                skipped_items.append(
                    {
                        "target_key": target_key,
                        "reason": "empty value",
                    }
                )
                continue

            db_no = int(
                target.get(
                    "write_db",
                    cfg.get("write_db", self.env.get("RECIPE_PLC_DB", 53)),
                )
            )

            byte = int(
                target.get(
                    "write_byte",
                    cfg.get("write_byte", -1),
                )
            )

            data_type = str(
                target.get(
                    "type",
                    cfg.get("type", self.env.get("RECIPE_AXIS_VALUE_TYPE", "REAL")),
                )
            ).upper()

            if db_no <= 0 or byte < 0:
                skipped_items.append(
                    {
                        "target_key": target_key,
                        "reason": f"invalid PLC address DB{db_no}, byte {byte}",
                    }
                )
                continue

            self._write_plc_value(
                client=client,
                db_no=db_no,
                byte=byte,
                data_type=data_type,
                value=float(value),
            )

            written_items.append(
                {
                    "target_key": target_key,
                    "value": float(value),
                    "db": db_no,
                    "byte": byte,
                    "type": data_type,
                }
            )

        return {
            "enabled": True,
            "written": len(written_items) > 0,
            "message": (
                f"Recipe target write complete. "
                f"Written={len(written_items)}, skipped={len(skipped_items)}."
            ),
            "written_items": written_items,
            "skipped_items": skipped_items,
        }
    
    def _write_legacy_axis_targets_to_plc(
        self,
        client,
        recipe_doc: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Legacy fallback for old recipe structure.

        Writes:
            camera_axis_targets -> RECIPE_CAMERA_AXIS_START_BYTE
            laser_axis_targets  -> RECIPE_LASER_AXIS_START_BYTE
        """
        db_no = int(self.env.get("RECIPE_PLC_DB", "130"))
        camera_start = int(self.env.get("RECIPE_CAMERA_AXIS_START_BYTE", "0"))
        laser_start = int(self.env.get("RECIPE_LASER_AXIS_START_BYTE", "100"))
        step = int(self.env.get("RECIPE_AXIS_STEP_BYTES", "4"))

        camera_targets = recipe_doc.get("camera_axis_targets", {}) or {}
        laser_targets = recipe_doc.get("laser_axis_targets", {}) or {}

        self._write_axis_group_to_plc(
            client=client,
            db_no=db_no,
            start_byte=camera_start,
            axis_ids=self.get_camera_axis_ids(),
            targets=camera_targets,
            step=step,
        )

        self._write_axis_group_to_plc(
            client=client,
            db_no=db_no,
            start_byte=laser_start,
            axis_ids=self.get_laser_axis_ids(),
            targets=laser_targets,
            step=step,
        )

        return {
            "enabled": True,
            "written": True,
            "message": f"Legacy recipe written to PLC DB{db_no}.",
        }

    def _write_axis_group_to_plc(
        self,
        client,
        db_no: int,
        start_byte: int,
        axis_ids: List[int],
        targets: Dict[str, Any],
        step: int = 4,
    ):
        for idx, axis_id in enumerate(axis_ids):
            axis_key = f"axis_{axis_id:02d}"
            target = targets.get(axis_key)

            if isinstance(target, dict):
                value = target.get("value", None)
            else:
                value = target

            if value is None or value == "":
                continue

            byte = int(start_byte) + idx * int(step)

            self._write_plc_value(
                client=client,
                db_no=db_no,
                byte=byte,
                data_type="REAL",
                value=float(value),
            )

    def _write_plc_value(self, client, db_no: int, byte: int, data_type: str, value):
        """
        Generic PLC DB write.

        Supports:
            REAL  -> 4 bytes
            INT   -> 2 bytes signed
            DINT  -> 4 bytes signed
            WORD  -> 2 bytes unsigned
            BYTE  -> 1 byte unsigned
        """

        data_type = str(data_type or "REAL").strip().upper()

        if client is None:
            raise RuntimeError("PLC client is not available.")

        if data_type == "REAL":
            data = bytearray(struct.pack(">f", float(value)))
            client.db_write(int(db_no), int(byte), data)
            return

        if data_type == "INT":
            data = bytearray(struct.pack(">h", int(value)))
            client.db_write(int(db_no), int(byte), data)
            return

        if data_type == "DINT":
            data = bytearray(struct.pack(">i", int(value)))
            client.db_write(int(db_no), int(byte), data)
            return

        if data_type == "WORD":
            data = bytearray(struct.pack(">H", int(value)))
            client.db_write(int(db_no), int(byte), data)
            return

        if data_type == "BYTE":
            data = bytearray([int(value) & 0xFF])
            client.db_write(int(db_no), int(byte), data)
            return

        raise RuntimeError(f"Unsupported PLC write type: {data_type}")