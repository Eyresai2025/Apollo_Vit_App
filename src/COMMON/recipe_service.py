from __future__ import annotations

import json
import os
import re
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, Optional, List

from src.COMMON.common import load_env
from src.COMMON.db import get_collection


try:
    import snap7  # type: ignore
    from snap7.util import get_real, get_int, get_dint, get_word, set_real  # type: ignore
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


def _parse_int_list(value: str, default: List[int]) -> List[int]:
    value = str(value or "").strip()
    if not value:
        return default

    out = []
    for part in value.split(","):
        part = part.strip()
        if part.isdigit():
            out.append(int(part))
    return out or default


class RecipeService:
    """
    Central backend for:
    - F-016 axis teaching
    - F-020 SKU recipe save and model link
    - F-041 recipe creation
    - F-042 recipe storage and versioning
    - partial F-045 axis target/live/delta view
    """

    def __init__(self, media_path: str, env_path: Optional[str] = None):
        self.media_path = Path(media_path)
        self.project_root = self.media_path.parent
        self.env_path = env_path or str(self.project_root / ".env")
        self.env = load_env(self.env_path)

        self.deployment = _to_bool(self.env.get("DEPLOYMENT", "False"))
        self.recipe_col = get_collection(RECIPE_COLLECTION)
        self.new_sku_col = get_collection("New SKU")

        self.backup_dir = Path(
            self.env.get(
                "RECIPE_BACKUP_DIR",
                str(self.media_path / "recipe_backups")
            )
        )
        if not self.backup_dir.is_absolute():
            self.backup_dir = self.project_root / self.backup_dir

        self.backup_dir.mkdir(parents=True, exist_ok=True)

    def get_axis_count(self) -> int:
        axis_ids = []
        for key in self.env.keys():
            m = re.match(r"AXIS_(\d+)_NAME", str(key))
            if m:
                axis_ids.append(int(m.group(1)))
        return max(axis_ids) if axis_ids else 12

    def get_camera_axis_ids(self) -> List[int]:
        return _parse_int_list(
            self.env.get("CAMERA_AXIS_IDS", ""),
            [1, 2, 3, 4, 5, 6]
        )

    def get_laser_axis_ids(self) -> List[int]:
        return _parse_int_list(
            self.env.get("LASER_AXIS_IDS", ""),
            [7, 8, 9, 10, 11, 12]
        )

    def read_current_axis_positions(self, plc_client=None) -> Dict[str, Dict[str, Any]]:
        """
        Demo:
            returns AXIS_i_RECIPE_POS from .env

        Production:
            reads AXIS_i_POS_DB / AXIS_i_POS_BYTE / AXIS_i_POS_TYPE from PLC.
        """
        result: Dict[str, Dict[str, Any]] = {}

        for axis_id in range(1, self.get_axis_count() + 1):
            axis_key = f"axis_{axis_id:02d}"
            axis_name = self.env.get(f"AXIS_{axis_id}_NAME", f"Axis {axis_id}")

            try:
                value = self._read_one_axis_position(axis_id, plc_client=plc_client)
                status = "OK"
            except Exception as e:
                value = None
                status = f"ERROR: {e}"

            result[axis_key] = {
                "axis_id": axis_id,
                "name": axis_name,
                "value": value,
                "status": status,
                "source": "PLC" if self.deployment else "ENV_DEMO",
            }

        return result

    def _read_one_axis_position(self, axis_id: int, plc_client=None):
        if not self.deployment:
            return float(self.env.get(f"AXIS_{axis_id}_RECIPE_POS", "0.0"))

        db_no = int(self.env.get(f"AXIS_{axis_id}_POS_DB", "0"))
        byte = int(self.env.get(f"AXIS_{axis_id}_POS_BYTE", "0"))
        data_type = str(self.env.get(f"AXIS_{axis_id}_POS_TYPE", "REAL")).upper()

        if db_no <= 0:
            raise RuntimeError(f"AXIS_{axis_id}_POS_DB not configured")

        return self._read_plc_value(
            db_no=db_no,
            byte=byte,
            data_type=data_type,
            plc_client=plc_client,
        )

    def _read_plc_value(self, db_no: int, byte: int, data_type: str, plc_client=None):
        if snap7 is None:
            raise RuntimeError("snap7 not installed")

        own_client = False
        client = plc_client

        if client is None:
            client = snap7.client.Client()
            own_client = True
            client.connect(
                self.env.get("PLC_IP", "192.168.10.1"),
                int(self.env.get("PLC_RACK", "0")),
                int(self.env.get("PLC_SLOT", "1")),
            )

        try:
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

        finally:
            if own_client:
                try:
                    client.disconnect()
                except Exception:
                    pass

    def build_recipe_doc(
        self,
        sku_meta: Dict[str, Any],
        camera_axis_targets: Dict[str, Any],
        laser_axis_targets: Dict[str, Any],
        camera_config_links: Dict[str, Any],
        laser_config_links: Dict[str, Any],
        vit_model_path: str = "",
        training_summary: Optional[Dict[str, Any]] = None,
        validation_result: Optional[Dict[str, Any]] = None,
        author: str = "operator",
    ) -> Dict[str, Any]:
        sku_name = str(sku_meta.get("sku_name") or sku_meta.get("tyre_name") or "").strip()
        if not sku_name:
            raise ValueError("SKU name is required before saving recipe.")

        next_version = self.get_next_version(sku_name)

        validation_result = validation_result or {}
        training_summary = training_summary or {}

        return {
            "type": "sku_recipe",
            "sku_name": sku_name,
            "sku_folder": _safe_name(sku_name),
            "version": next_version,
            "status": "DRAFT" if not validation_result.get("accepted") else "ACCEPTED",

            "tyre_size": sku_meta.get("tyre_size", ""),
            "barcode_pattern": sku_meta.get("barcode_pattern", ""),
            "inspection_zones": int(sku_meta.get("inspection_zones", 5)),
            "image_count_per_zone": int(sku_meta.get("image_count_per_zone", 20)),

            # Important: separate fields as requested by project manager.
            "camera_axis_targets": camera_axis_targets or {},
            "laser_axis_targets": laser_axis_targets or {},

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
            sort=[("version", -1)]
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
            plc_result = self.write_recipe_to_plc(recipe_doc, plc_client=plc_client)

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
        version = recipe_doc.get("version", 1)

        sku_dir = self.backup_dir / sku_folder
        sku_dir.mkdir(parents=True, exist_ok=True)

        backup_path = sku_dir / f"{sku_folder}_recipe_v{version:03d}.json"

        clean_doc = dict(recipe_doc)
        clean_doc.pop("_id", None)

        with open(backup_path, "w", encoding="utf-8") as f:
            json.dump(clean_doc, f, indent=2, ensure_ascii=False)

        return backup_path

    def write_recipe_to_plc(self, recipe_doc: Dict[str, Any], plc_client=None) -> Dict[str, Any]:
        """
        Writes camera and laser axis target values to DB_Recipe.

        This needs final byte mapping confirmation from PLC team.
        Until RECIPE_WRITE_TO_PLC=True, local/Mongo saving is still active.
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

        db_no = int(self.env.get("RECIPE_PLC_DB", "130"))
        camera_start = int(self.env.get("RECIPE_CAMERA_AXIS_START_BYTE", "0"))
        laser_start = int(self.env.get("RECIPE_LASER_AXIS_START_BYTE", "100"))
        step = int(self.env.get("RECIPE_AXIS_STEP_BYTES", "4"))

        own_client = False
        client = plc_client

        if client is None:
            client = snap7.client.Client()
            own_client = True
            client.connect(
                self.env.get("PLC_IP", "192.168.10.1"),
                int(self.env.get("PLC_RACK", "0")),
                int(self.env.get("PLC_SLOT", "1")),
            )

        try:
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
                "message": f"Recipe written to PLC DB{db_no}.",
            }

        except Exception as e:
            return {
                "enabled": True,
                "written": False,
                "message": str(e),
            }

        finally:
            if own_client:
                try:
                    client.disconnect()
                except Exception:
                    pass

    def _write_axis_group_to_plc(
        self,
        client,
        db_no: int,
        start_byte: int,
        axis_ids: List[int],
        targets: Dict[str, Any],
        step: int,
    ) -> None:
        if set_real is None:
            raise RuntimeError("snap7.util.set_real unavailable")

        total_bytes = max(4, len(axis_ids) * step)
        data = bytearray(total_bytes)

        for idx, axis_id in enumerate(axis_ids):
            axis_key = f"axis_{axis_id:02d}"
            raw_value = targets.get(axis_key, 0.0)

            if isinstance(raw_value, dict):
                value = raw_value.get("value", 0.0)
            else:
                value = raw_value

            offset = idx * step
            set_real(data, offset, float(value or 0.0))

        client.db_write(db_no, start_byte, data)

    def get_latest_recipe(self, sku_name: str) -> Optional[Dict[str, Any]]:
        return self.recipe_col.find_one(
            {
                "type": "sku_recipe",
                "sku_name": sku_name,
            },
            sort=[("version", -1)]
        )