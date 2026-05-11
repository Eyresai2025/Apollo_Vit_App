from __future__ import annotations

import json
import re
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, Optional, List

from src.COMMON.common import load_env
from src.COMMON.db import get_collection


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

    # ------------------------------------------------------------
    # PRODUCTION RECIPE TARGET CONFIG
    # ------------------------------------------------------------
    def get_recipe_target_configs(self) -> List[Dict[str, Any]]:
        """
        Reads production recipe target rows from .env.

        Example:
            RECIPE_TARGET_COUNT=17

            RECIPE_TARGET_13_KEY=sidewall1_laser_fwd_rev
            RECIPE_TARGET_13_GROUP=LASER
            RECIPE_TARGET_13_AXIS_ID=5
            RECIPE_TARGET_13_NAME=Sidewall 1 Laser FWD/REV Target
            RECIPE_TARGET_13_WRITE_DB=130
            RECIPE_TARGET_13_WRITE_BYTE=100
            RECIPE_TARGET_13_TYPE=REAL

        Returns one row per recipe target.
        One physical servo axis can appear multiple times.
        """
        count = _env_int(self.env, "RECIPE_TARGET_COUNT", 0)
        targets: List[Dict[str, Any]] = []

        axis_configs = self.get_all_axis_configs()

        for idx in range(1, count + 1):
            prefix = f"RECIPE_TARGET_{idx}_"

            key = _env_str(self.env, prefix + "KEY", "")
            if not key:
                continue

            axis_id = _env_int(self.env, prefix + "AXIS_ID", 0)
            axis_cfg = axis_configs.get(axis_id, {})

            group = _env_str(self.env, prefix + "GROUP", "MACHINE").upper()
            name = _env_str(self.env, prefix + "NAME", key)

            write_db = _env_int(
                self.env,
                prefix + "WRITE_DB",
                _env_int(self.env, "RECIPE_PLC_DB", 130),
            )

            write_byte = _env_int(self.env, prefix + "WRITE_BYTE", -1)
            data_type = _env_str(
                self.env,
                prefix + "TYPE",
                _env_str(self.env, "RECIPE_AXIS_VALUE_TYPE", "REAL"),
            ).upper()

            targets.append(
                {
                    "target_index": idx,
                    "target_key": key,
                    "group": group,
                    "axis_id": axis_id,
                    "axis_key": f"axis_{axis_id:02d}" if axis_id > 0 else "",
                    "axis_name": axis_cfg.get("name", f"Axis {axis_id}"),
                    "axis_ip": axis_cfg.get("ip", ""),
                    "target_name": name,
                    "write_db": write_db,
                    "write_byte": write_byte,
                    "type": data_type,
                }
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

    def _read_plc_value(
        self,
        db_no: int,
        byte: int,
        data_type: str,
        plc_client=None,
    ):
        if snap7 is None:
            raise RuntimeError("snap7 not installed")

        client = plc_client or self.plc_client

        if client is None:
            raise RuntimeError(
                "PLC client not available. Run Test Mode first or pass plc_client to RecipeService."
            )

        try:
            if hasattr(client, "get_connected") and not client.get_connected():
                raise RuntimeError("Shared PLC client is disconnected")

            data_type = str(data_type).upper()

            if data_type == "REAL":
                data = client.db_read(db_no, byte, 4)
                return float(get_real(data, 0))

            if data_type == "DINT":
                data = client.db_read(db_no, byte, 4)
                return int(get_dint(data, 0))

            if data_type == "INT":
                data = client.db_read(db_no, byte, 2)
                return int(get_int(data, 0))

            if data_type == "WORD":
                data = client.db_read(db_no, byte, 2)
                return int(get_word(data, 0))

            raise RuntimeError(f"Unsupported PLC data type: {data_type}")

        except Exception as e:
            raise RuntimeError(
                f"PLC read failed DB{db_no}, byte {byte}, type {data_type}: {e}"
            )

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

        inserted = self.recipe_col.insert_one(recipe_doc)

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
    # PLC RECIPE WRITE
    # ------------------------------------------------------------
    def write_recipe_to_plc(
        self,
        recipe_doc: Dict[str, Any],
        plc_client=None,
    ) -> Dict[str, Any]:
        """
        Writes recipe targets to PLC.

        Priority:
        1. Production new field:
            recipe_axis_targets
            Uses exact write_db/write_byte per target.

        2. Legacy fallback:
            camera_axis_targets / laser_axis_targets
            Uses RECIPE_CAMERA_AXIS_START_BYTE / RECIPE_LASER_AXIS_START_BYTE.
        """
        if not self.deployment:
            return {
                "enabled": True,
                "written": False,
                "message": "DEPLOYMENT=False, PLC write skipped.",
            }

        if snap7 is None:
            return {
                "enabled": True,
                "written": False,
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
                return self._write_recipe_targets_to_plc(
                    client=client,
                    recipe_axis_targets=recipe_axis_targets,
                )

            return self._write_legacy_axis_targets_to_plc(
                client=client,
                recipe_doc=recipe_doc,
            )

        except Exception as e:
            return {
                "enabled": True,
                "written": False,
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
                    cfg.get("write_db", self.env.get("RECIPE_PLC_DB", 130)),
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

    def _write_plc_value(
        self,
        client,
        db_no: int,
        byte: int,
        data_type: str,
        value: float,
    ):
        data_type = str(data_type).upper()

        if data_type != "REAL":
            raise RuntimeError(
                f"PLC recipe write currently supports REAL only. Got {data_type}."
            )

        if set_real is None:
            raise RuntimeError("snap7.util.set_real is not available")

        data = bytearray(4)
        set_real(data, 0, float(value))
        client.db_write(int(db_no), int(byte), data)