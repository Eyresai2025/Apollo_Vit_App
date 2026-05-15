from __future__ import annotations

import copy
import json
from datetime import datetime
from typing import Any, Dict, List, Optional

from bson import ObjectId  # type: ignore

from PyQt5.QtCore import Qt  # type: ignore
from PyQt5.QtWidgets import (  # type: ignore
    QWidget, QVBoxLayout, QHBoxLayout, QFrame, QLabel,
    QPushButton, QMessageBox, QComboBox, QTextEdit,
    QTableWidget, QTableWidgetItem, QHeaderView, QSizePolicy
)

from src.COMMON.recipe_service import RecipeService
from src.COMMON.db import get_collection


def _json_default(obj):
    if isinstance(obj, ObjectId):
        return str(obj)
    try:
        return str(obj)
    except Exception:
        return ""


def _safe_text(value: Any, default: str = "-") -> str:
    if value is None:
        return default

    text = str(value).strip()

    if text == "":
        return default

    return text


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


class RecipeManagementPage(QWidget):
    """
    F-043 / F-044 Recipe Management page.

    Production purpose:
        - View saved SKU recipes
        - View version history
        - View recipe_axis_targets from MongoDB
        - Edit target values safely
        - Save edited values as a new version
        - Load selected recipe to PLC DB53
        - Show DB53 write/read-back verification result

    Important:
        This page does NOT decide production active SKU.
        Production active SKU will later come from PLC/DB75.
    """

    def __init__(
        self,
        media_path: str,
        env_path: str = "",
        on_close=None,
        on_edit_recipe=None,  # kept only for backward compatibility, not used
        parent=None,
    ):
        super().__init__(parent)

        self.media_path = media_path
        self.env_path = env_path
        self.on_close = on_close
        self.on_edit_recipe = None

        self.recipe_service = RecipeService(
            media_path=self.media_path,
            env_path=self.env_path or None,
        )

        self.recipe_col = get_collection("SKU Recipes")
        self.active_recipe_col = get_collection("Active Recipe")

        self.current_recipes: List[Dict[str, Any]] = []
        self.selected_recipe: Optional[Dict[str, Any]] = None

        self.edit_mode = False

        self.sku_combo: Optional[QComboBox] = None
        self.version_combo: Optional[QComboBox] = None
        self.summary_lbl: Optional[QLabel] = None
        self.axis_table: Optional[QTableWidget] = None
        self.raw_json: Optional[QTextEdit] = None

        self.edit_values_btn: Optional[QPushButton] = None
        self.save_version_btn: Optional[QPushButton] = None
        self.cancel_edit_btn: Optional[QPushButton] = None
        self.load_plc_btn: Optional[QPushButton] = None

        self._build_ui()
        self.refresh_recipes()

    # =========================================================
    # UI THEME
    # =========================================================

    def _primary_btn(self, text: str) -> QPushButton:
        btn = QPushButton(text)
        btn.setCursor(Qt.PointingHandCursor)
        btn.setFixedHeight(38)
        btn.setStyleSheet("""
            QPushButton {
                background:#571c86;
                color:white;
                border:none;
                border-radius:19px;
                padding:0 18px;
                font:700 11px 'Segoe UI';
            }
            QPushButton:hover { background:#6b2aa3; }
            QPushButton:disabled {
                background:#cfc3e0;
                color:#f0ecf5;
            }
        """)
        return btn

    def _secondary_btn(self, text: str) -> QPushButton:
        btn = QPushButton(text)
        btn.setCursor(Qt.PointingHandCursor)
        btn.setFixedHeight(38)
        btn.setStyleSheet("""
            QPushButton {
                background:white;
                color:#571c86;
                border:1px solid #d8cce8;
                border-radius:19px;
                padding:0 18px;
                font:700 11px 'Segoe UI';
            }
            QPushButton:hover {
                background:#faf7fd;
                border-color:#bfa7dc;
            }
            QPushButton:disabled {
                color:#aaa0b8;
                border-color:#e9e1f2;
                background:#fafafa;
            }
        """)
        return btn

    def _danger_btn(self, text: str) -> QPushButton:
        btn = QPushButton(text)
        btn.setCursor(Qt.PointingHandCursor)
        btn.setFixedHeight(38)
        btn.setStyleSheet("""
            QPushButton {
                background:#d93f3f;
                color:white;
                border:none;
                border-radius:19px;
                padding:0 18px;
                font:700 11px 'Segoe UI';
            }
            QPushButton:hover { background:#bf3535; }
            QPushButton:disabled {
                background:#e9b1b1;
                color:#fff;
            }
        """)
        return btn

    def _build_ui(self):
        self.setStyleSheet("""
            QWidget {
                background:#f6f3f9;
                font:10pt 'Segoe UI';
                color:#302a38;
            }

            QFrame#MainCard {
                background:#ffffff;
                border:1px solid #e6deef;
                border-radius:18px;
            }

            QFrame#InnerCard {
                background:#fbf9fd;
                border:1px solid #eee6f6;
                border-radius:14px;
            }

            QLabel#Title {
                font:700 22px 'Segoe UI';
                color:#571c86;
                background:transparent;
                border:none;
            }

            QLabel#SubTitle {
                font:500 11px 'Segoe UI';
                color:#7a7288;
                background:transparent;
                border:none;
            }

            QLabel#SectionTitle {
                font:700 13px 'Segoe UI';
                color:#571c86;
                background:transparent;
                border:none;
            }

            QLabel#StatusBox {
                background:#f4eefb;
                color:#49305f;
                border:1px solid #dfd2ef;
                border-radius:14px;
                padding:14px;
                font:600 11px 'Segoe UI';
            }

            QComboBox {
                background:white;
                border:1px solid #d8cce8;
                border-radius:10px;
                min-height:36px;
                padding:0 12px;
                color:#2f2a36;
            }

            QComboBox:focus {
                border:2px solid #571c86;
            }

            QTableWidget {
                background:white;
                border:1px solid #dfd6ea;
                border-radius:12px;
                gridline-color:#ece5f4;
                alternate-background-color:#faf8fd;
                selection-background-color:#eee4f8;
                selection-color:#2f2a36;
            }

            QHeaderView::section {
                background:#f3edf9;
                color:#571c86;
                padding:8px;
                border:none;
                border-bottom:1px solid #ddd3ea;
                font:700 11px 'Segoe UI';
            }

            QTextEdit {
                background:#ffffff;
                border:1px solid #dfd6ea;
                border-radius:12px;
                padding:10px;
                font:10px 'Consolas';
                color:#36303f;
            }
        """)

        root = QVBoxLayout(self)
        root.setContentsMargins(18, 12, 18, 12)
        root.setSpacing(12)

        main_card = QFrame()
        main_card.setObjectName("MainCard")
        main_l = QVBoxLayout(main_card)
        main_l.setContentsMargins(22, 18, 22, 18)
        main_l.setSpacing(16)

        title = QLabel("Recipe Management")
        title.setObjectName("Title")
        main_l.addWidget(title)

        sub = QLabel(
            "View saved recipes, edit target values as a new version, "
            "and load the selected recipe to PLC DB53 with read-back verification."
        )
        sub.setObjectName("SubTitle")
        sub.setWordWrap(True)
        main_l.addWidget(sub)

        # =====================================================
        # TOP SELECTION CARD
        # =====================================================
        select_card = QFrame()
        select_card.setObjectName("InnerCard")
        select_l = QHBoxLayout(select_card)
        select_l.setContentsMargins(16, 14, 16, 14)
        select_l.setSpacing(12)

        sku_lbl = QLabel("SKU")
        sku_lbl.setObjectName("SectionTitle")
        select_l.addWidget(sku_lbl)

        self.sku_combo = QComboBox()
        self.sku_combo.setMinimumWidth(240)
        self.sku_combo.currentIndexChanged.connect(self._on_sku_changed)
        select_l.addWidget(self.sku_combo)

        ver_lbl = QLabel("Version")
        ver_lbl.setObjectName("SectionTitle")
        select_l.addWidget(ver_lbl)

        self.version_combo = QComboBox()
        self.version_combo.setMinimumWidth(180)
        self.version_combo.currentIndexChanged.connect(self._on_version_changed)
        select_l.addWidget(self.version_combo)

        refresh_btn = self._secondary_btn("Refresh")
        refresh_btn.clicked.connect(self.refresh_recipes)
        select_l.addWidget(refresh_btn)

        select_l.addStretch(1)
        main_l.addWidget(select_card)

        # =====================================================
        # SUMMARY CARD
        # =====================================================
        self.summary_lbl = QLabel("No recipe selected.")
        self.summary_lbl.setObjectName("StatusBox")
        self.summary_lbl.setWordWrap(True)
        main_l.addWidget(self.summary_lbl)

        # =====================================================
        # TARGET TABLE
        # =====================================================
        table_title = QLabel("Recipe Target Values")
        table_title.setObjectName("SectionTitle")
        main_l.addWidget(table_title)

        self.axis_table = QTableWidget()
        self.axis_table.setColumnCount(9)
        self.axis_table.setHorizontalHeaderLabels([
            "Group",
            "Target Key",
            "Target Name",
            "Axis",
            "Target Value",
            "DB",
            "Byte",
            "Type",
            "Captured / Updated At",
        ])
        self.axis_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.axis_table.setAlternatingRowColors(True)
        self.axis_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.axis_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.axis_table.setMinimumHeight(260)
        main_l.addWidget(self.axis_table)

        # =====================================================
        # RAW JSON
        # =====================================================
        json_title = QLabel("Recipe JSON")
        json_title.setObjectName("SectionTitle")
        main_l.addWidget(json_title)

        self.raw_json = QTextEdit()
        self.raw_json.setReadOnly(True)
        self.raw_json.setMinimumHeight(190)
        self.raw_json.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        main_l.addWidget(self.raw_json, 1)

        # =====================================================
        # BUTTON ROW
        # =====================================================
        btn_row = QHBoxLayout()
        btn_row.setSpacing(10)

        self.edit_values_btn = self._secondary_btn("Edit Values")
        self.edit_values_btn.clicked.connect(self.enter_edit_mode)

        self.save_version_btn = self._primary_btn("Save as New Version")
        self.save_version_btn.clicked.connect(self.save_edited_recipe_as_new_version)
        self.save_version_btn.setEnabled(False)

        self.cancel_edit_btn = self._danger_btn("Cancel Edit")
        self.cancel_edit_btn.clicked.connect(self.cancel_edit_mode)
        self.cancel_edit_btn.setEnabled(False)

        mark_test_active_btn = self._secondary_btn("Mark Test Active")
        mark_test_active_btn.clicked.connect(self.mark_selected_recipe_as_test_active)

        self.load_plc_btn = self._primary_btn("Load Recipe to Machine")
        self.load_plc_btn.clicked.connect(self.load_selected_recipe_to_machine)

        close_btn = self._secondary_btn("Back")
        close_btn.clicked.connect(self.close_page)

        btn_row.addWidget(self.edit_values_btn)
        btn_row.addWidget(self.save_version_btn)
        btn_row.addWidget(self.cancel_edit_btn)
        btn_row.addWidget(mark_test_active_btn)
        btn_row.addStretch(1)
        btn_row.addWidget(self.load_plc_btn)
        btn_row.addWidget(close_btn)

        main_l.addLayout(btn_row)
        root.addWidget(main_card)

    # =========================================================
    # DATA LOADING
    # =========================================================

    def refresh_recipes(self):
        try:
            previous_sku = self.sku_combo.currentText().strip() if self.sku_combo else ""

            recipes = list(
                self.recipe_col.find(
                    {"type": "sku_recipe"},
                    sort=[("sku_name", 1), ("version", -1)]
                )
            )

            self.current_recipes = recipes

            sku_names = sorted({
                str(r.get("sku_name", "")).strip()
                for r in recipes
                if str(r.get("sku_name", "")).strip()
            })

            self.sku_combo.blockSignals(True)
            self.sku_combo.clear()
            self.sku_combo.addItems(sku_names)
            self.sku_combo.blockSignals(False)

            if sku_names:
                if previous_sku in sku_names:
                    self.sku_combo.setCurrentText(previous_sku)
                else:
                    self.sku_combo.setCurrentIndex(0)
                self._on_sku_changed()
            else:
                self.selected_recipe = None
                self.version_combo.clear()
                self.summary_lbl.setText("No recipes found. Create and save a recipe from Run New SKU page first.")
                self.axis_table.setRowCount(0)
                self.raw_json.setPlainText("")

        except Exception as e:
            QMessageBox.critical(self, "Recipe Load Error", str(e))

    def _recipes_for_sku(self, sku_name: str) -> List[Dict[str, Any]]:
        return [
            r for r in self.current_recipes
            if str(r.get("sku_name", "")).strip() == sku_name
        ]

    def _on_sku_changed(self):
        self.edit_mode = False
        self._update_edit_buttons()

        sku_name = self.sku_combo.currentText().strip()
        recipes = self._recipes_for_sku(sku_name)

        self.version_combo.blockSignals(True)
        self.version_combo.clear()

        for r in recipes:
            version = r.get("version", "-")
            status = r.get("status", "DRAFT")
            self.version_combo.addItem(f"v{version} | {status}", r)

        self.version_combo.blockSignals(False)

        if recipes:
            self.version_combo.setCurrentIndex(0)
            self._on_version_changed()
        else:
            self.selected_recipe = None
            self._render_recipe(None)

    def _on_version_changed(self):
        self.edit_mode = False
        self._update_edit_buttons()

        recipe = self.version_combo.currentData()
        self.selected_recipe = recipe if isinstance(recipe, dict) else None
        self._render_recipe(self.selected_recipe)

    # =========================================================
    # RENDER
    # =========================================================

    def _render_recipe(self, recipe: Optional[Dict[str, Any]]):
        if not recipe:
            self.summary_lbl.setText("No recipe selected.")
            self.axis_table.setRowCount(0)
            self.raw_json.setPlainText("")
            return

        sku_name = _safe_text(recipe.get("sku_name"))
        version = _safe_text(recipe.get("version"))
        status = _safe_text(recipe.get("status"), "DRAFT")
        recipe_number = _safe_text(
            recipe.get("recipe_number") or recipe.get("plc_recipe_number"),
            "-"
        )
        tyre_size = _safe_text(recipe.get("tyre_size"))
        barcode = _safe_text(recipe.get("barcode_pattern") or recipe.get("barcode"))
        model_path = _safe_text(recipe.get("vit_model_path"), "Not linked yet")
        val_score = _safe_text(recipe.get("validation_score"), "Pending")
        created_at = _safe_text(recipe.get("created_at"))
        updated_at = _safe_text(recipe.get("updated_at"))
        author = _safe_text(recipe.get("author"), "operator")

        recipe_axis_targets = recipe.get("recipe_axis_targets", {}) or {}
        camera_targets = recipe.get("camera_axis_targets", {}) or {}
        laser_targets = recipe.get("laser_axis_targets", {}) or {}

        if recipe_axis_targets:
            target_count = len(recipe_axis_targets)
            target_mode = "Production recipe_axis_targets"
        else:
            target_count = len(camera_targets) + len(laser_targets)
            target_mode = "Legacy camera/laser targets"

        summary = (
            f"SKU: {sku_name}    |    Recipe No: {recipe_number}    |    Version: {version}    |    Status: {status}\n"
            f"Tyre Size: {tyre_size}    |    Barcode: {barcode}    |    Validation F1: {val_score}\n"
            f"Author: {author}    |    Created: {created_at}    |    Updated: {updated_at}\n"
            f"Targets: {target_count} ({target_mode})\n"
            f"Model Path: {model_path}"
        )
        self.summary_lbl.setText(summary)

        self._render_axis_table(recipe)

        pretty = json.dumps(
            recipe,
            indent=2,
            ensure_ascii=False,
            default=_json_default
        )
        self.raw_json.setPlainText(pretty)

    def _target_rows_from_recipe(self, recipe: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Prefer production recipe_axis_targets.
        Fallback to legacy camera_axis_targets / laser_axis_targets for old recipes.
        """

        rows: List[Dict[str, Any]] = []

        recipe_targets = recipe.get("recipe_axis_targets", {}) or {}

        if recipe_targets:
            for target_key, info in recipe_targets.items():
                if not isinstance(info, dict):
                    continue

                rows.append({
                    "group": _safe_text(info.get("group")),
                    "target_key": target_key,
                    "target_name": _safe_text(info.get("target_name")),
                    "axis": _safe_text(info.get("axis_key") or f"axis_{int(info.get('axis_id', 0)):02d}"),
                    "value": info.get("value"),
                    "db": info.get("write_db"),
                    "byte": info.get("write_byte"),
                    "type": _safe_text(info.get("type"), "REAL"),
                    "captured_at": _safe_text(info.get("captured_at")),
                    "target_index": info.get("target_index", 9999),
                    "raw": info,
                })

            rows.sort(key=lambda r: int(r.get("target_index") or 9999))
            return rows

        # Legacy fallback for old recipes.
        camera_targets = recipe.get("camera_axis_targets", {}) or {}
        laser_targets = recipe.get("laser_axis_targets", {}) or {}

        for axis_key, info in sorted(camera_targets.items()):
            if isinstance(info, dict):
                rows.append({
                    "group": "CAMERA",
                    "target_key": axis_key,
                    "target_name": _safe_text(info.get("name")),
                    "axis": axis_key,
                    "value": info.get("value"),
                    "db": "",
                    "byte": "",
                    "type": "REAL",
                    "captured_at": _safe_text(info.get("captured_at")),
                    "target_index": 9999,
                    "raw": info,
                })

        for axis_key, info in sorted(laser_targets.items()):
            if isinstance(info, dict):
                rows.append({
                    "group": "LASER",
                    "target_key": axis_key,
                    "target_name": _safe_text(info.get("name")),
                    "axis": axis_key,
                    "value": info.get("value"),
                    "db": "",
                    "byte": "",
                    "type": "REAL",
                    "captured_at": _safe_text(info.get("captured_at")),
                    "target_index": 9999,
                    "raw": info,
                })

        return rows

    def _render_axis_table(self, recipe: Dict[str, Any]):
        rows = self._target_rows_from_recipe(recipe)

        self.axis_table.setRowCount(len(rows))

        for row_idx, row in enumerate(rows):
            values = [
                row.get("group", "-"),
                row.get("target_key", "-"),
                row.get("target_name", "-"),
                row.get("axis", "-"),
                self._fmt_value(row.get("value")),
                row.get("db", "-"),
                row.get("byte", "-"),
                row.get("type", "REAL"),
                row.get("captured_at", "-"),
            ]

            for col_idx, value in enumerate(values):
                item = QTableWidgetItem(str(value))
                item.setTextAlignment(Qt.AlignCenter)

                # Only target value column is editable in edit mode.
                if self.edit_mode and col_idx == 4:
                    item.setFlags(item.flags() | Qt.ItemIsEditable)
                else:
                    item.setFlags(item.flags() & ~Qt.ItemIsEditable)

                self.axis_table.setItem(row_idx, col_idx, item)

        if self.edit_mode:
            self.axis_table.setEditTriggers(
                QTableWidget.DoubleClicked |
                QTableWidget.EditKeyPressed |
                QTableWidget.AnyKeyPressed
            )
        else:
            self.axis_table.setEditTriggers(QTableWidget.NoEditTriggers)

    def _fmt_value(self, value: Any) -> str:
        if value is None or value == "":
            return "-"
        try:
            return f"{float(value):.3f}"
        except Exception:
            return str(value)

    # =========================================================
    # EDIT MODE / VERSIONING
    # =========================================================

    def _update_edit_buttons(self):
        if self.edit_values_btn is not None:
            self.edit_values_btn.setEnabled(not self.edit_mode)

        if self.save_version_btn is not None:
            self.save_version_btn.setEnabled(self.edit_mode)

        if self.cancel_edit_btn is not None:
            self.cancel_edit_btn.setEnabled(self.edit_mode)

        if self.load_plc_btn is not None:
            self.load_plc_btn.setEnabled(not self.edit_mode)

    def enter_edit_mode(self):
        if not self.selected_recipe:
            QMessageBox.warning(self, "Recipe", "Please select a recipe first.")
            return

        self.edit_mode = True
        self._update_edit_buttons()
        self._render_recipe(self.selected_recipe)

        QMessageBox.information(
            self,
            "Edit Mode",
            "Edit only the Target Value column.\n\nAfter editing, click 'Save as New Version'."
        )

    def cancel_edit_mode(self):
        self.edit_mode = False
        self._update_edit_buttons()
        self._render_recipe(self.selected_recipe)

    def _build_edited_recipe_from_table(self) -> Dict[str, Any]:
        if not self.selected_recipe:
            raise RuntimeError("No recipe selected.")

        new_doc = copy.deepcopy(self.selected_recipe)
        new_doc.pop("_id", None)

        sku_name = str(new_doc.get("sku_name", "")).strip()
        if not sku_name:
            raise RuntimeError("Recipe SKU name is missing.")

        recipe_targets = new_doc.get("recipe_axis_targets", {}) or {}

        if not recipe_targets:
            raise RuntimeError(
                "Selected recipe does not have recipe_axis_targets. "
                "Please recreate/save this SKU from the updated New SKU page before editing/loading."
            )

        for row in range(self.axis_table.rowCount()):
            target_key_item = self.axis_table.item(row, 1)
            value_item = self.axis_table.item(row, 4)

            if target_key_item is None or value_item is None:
                continue

            target_key = target_key_item.text().strip()
            value_text = value_item.text().strip()

            if not target_key:
                continue

            if target_key not in recipe_targets:
                raise RuntimeError(f"Target key not found in recipe_axis_targets: {target_key}")

            try:
                value = float(value_text)
            except Exception:
                raise RuntimeError(f"Invalid target value for {target_key}: {value_text}")

            recipe_targets[target_key]["value"] = value
            recipe_targets[target_key]["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            recipe_targets[target_key]["source"] = "MANUAL_EDIT_RECIPE_MANAGEMENT"

        new_doc["recipe_axis_targets"] = recipe_targets
        new_doc["version"] = self.recipe_service.get_next_version(sku_name)
        new_doc["status"] = "DRAFT"
        new_doc["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        new_doc["created_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        new_doc["modified_from_version"] = self.selected_recipe.get("version")
        new_doc["modified_from_recipe_id"] = str(self.selected_recipe.get("_id"))

        # Keep legacy fields roughly in sync for old displays/pages.
        self._sync_legacy_targets_from_recipe_axis_targets(new_doc)

        return new_doc

    def _sync_legacy_targets_from_recipe_axis_targets(self, recipe_doc: Dict[str, Any]) -> None:
        recipe_targets = recipe_doc.get("recipe_axis_targets", {}) or {}

        camera_targets = {}
        laser_targets = {}

        for target_key, target in recipe_targets.items():
            if not isinstance(target, dict):
                continue

            group = str(target.get("group", "")).upper()
            axis_key = target.get("axis_key") or f"axis_{int(target.get('axis_id', 0)):02d}"

            legacy_doc = {
                "target_key": target_key,
                "axis_id": target.get("axis_id"),
                "name": target.get("target_name") or target.get("axis_name") or axis_key,
                "value": target.get("value"),
                "captured_at": target.get("captured_at") or target.get("updated_at"),
                "source": target.get("source", ""),
                "write_db": target.get("write_db"),
                "write_byte": target.get("write_byte"),
                "type": target.get("type", "REAL"),
            }

            if group == "LASER":
                laser_targets[axis_key] = legacy_doc
            else:
                camera_targets[axis_key] = legacy_doc

        recipe_doc["camera_axis_targets"] = camera_targets
        recipe_doc["laser_axis_targets"] = laser_targets

    def save_edited_recipe_as_new_version(self):
        try:
            new_doc = self._build_edited_recipe_from_table()

            result = self.recipe_service.save_recipe(
                new_doc,
                plc_client=None,
                write_to_plc=False,
            )

            self.edit_mode = False
            self._update_edit_buttons()
            self.refresh_recipes()

            QMessageBox.information(
                self,
                "Recipe Version Saved",
                (
                    f"Edited recipe saved as a new version.\n\n"
                    f"SKU: {result.get('sku_name')}\n"
                    f"Version: {result.get('version')}\n"
                    f"Backup:\n{result.get('backup_path')}"
                )
            )

        except Exception as e:
            QMessageBox.critical(self, "Save Version Error", str(e))

    # =========================================================
    # ACTIONS
    # =========================================================

    def mark_selected_recipe_as_test_active(self):
        """
        This is only for engineering/testing state.
        Production active SKU comes from PLC/DB75 later.
        """
        recipe = self.selected_recipe

        if not recipe:
            QMessageBox.warning(self, "Recipe", "Please select a recipe first.")
            return

        try:
            self.active_recipe_col.update_one(
                {"type": "test_active_recipe"},
                {
                    "$set": {
                        "type": "test_active_recipe",
                        "sku_name": recipe.get("sku_name"),
                        "recipe_id": str(recipe.get("_id")),
                        "recipe_version": recipe.get("version"),
                        "status": recipe.get("status"),
                        "vit_model_path": recipe.get("vit_model_path", ""),
                        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "source": "MANUAL_ENGINEERING_TEST",
                    }
                },
                upsert=True
            )

            QMessageBox.information(
                self,
                "Test Active Recipe",
                (
                    "Recipe marked as test active.\n\n"
                    "Note: In production, PLC active SKU remains the source of truth."
                )
            )

        except Exception as e:
            QMessageBox.critical(self, "Active Recipe Error", str(e))

    def load_selected_recipe_to_machine(self):
        """
        Manual/engineering PLC load.

        Writes selected recipe to PLC DB53 using RecipeService.
        RecipeService now also performs DB53 read-back verification.
        """
        recipe = self.selected_recipe

        if not recipe:
            QMessageBox.warning(self, "Recipe", "Please select a recipe first.")
            return

        if not recipe.get("recipe_axis_targets"):
            QMessageBox.warning(
                self,
                "Recipe Format",
                (
                    "This recipe does not contain recipe_axis_targets.\n\n"
                    "Please recreate/save this SKU from the updated New SKU page before loading to PLC."
                )
            )
            return

        reply = QMessageBox.question(
            self,
            "Load Recipe to Machine",
            (
                "This will write the selected recipe target values to PLC DB53, "
                "verify DB53 read-back, write the recipe number to DB75.DBW288, "
                "and verify the recipe number read-back.\n\n"
                "Continue?"
            ),
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )

        if reply != QMessageBox.Yes:
            return

        try:
            result = self.recipe_service.write_recipe_to_plc(
                recipe_doc=recipe,
                plc_client=None,
            )

            msg = self._format_plc_result_message(result)

            QMessageBox.information(
                self,
                "PLC Recipe Load",
                msg
            )

        except Exception as e:
            QMessageBox.critical(self, "PLC Recipe Load Error", str(e))

    def _format_plc_result_message(self, result: Dict[str, Any]) -> str:
        verify_result = result.get("verify_result", {}) or {}
        recipe_number_result = result.get("recipe_number_result", {}) or {}
        plc_enabled = bool(result.get("enabled", False))
        plc_written = bool(result.get("written", False))
        plc_verified = bool(result.get("verified", False))

        written_items = result.get("written_items", []) or []
        skipped_items = result.get("skipped_items", []) or []
        mismatches = result.get("mismatches", []) or verify_result.get("mismatches", []) or []

        if not plc_enabled:
            return (
                "PLC Write: Disabled\n"
                f"PLC Message: {result.get('message', '')}"
            )

        msg = (
            f"PLC Write: {'OK' if plc_written else 'NOT OK'}\n"
            f"PLC Verify: {'OK' if plc_verified else 'NOT OK / SKIPPED'}\n"
            f"Recipe Number Write: {'OK' if recipe_number_result.get('written') else 'NOT OK / SKIPPED'}\n"
            f"Recipe Number Verify: {'OK' if recipe_number_result.get('verified') else 'NOT OK / SKIPPED'}\n"
            f"Targets Written: {len(written_items)}\n"
            f"Targets Skipped: {len(skipped_items)}\n"
            f"Verify Count: {verify_result.get('verified_count', 0)}\n"
            f"Mismatch Count: {verify_result.get('mismatch_count', len(mismatches))}\n"
            f"PLC Message: {result.get('message', '')}"
        )

        if mismatches:
            mismatch_lines = []

            for item in mismatches[:8]:
                mismatch_lines.append(
                    f"- {item.get('target_key')} | "
                    f"Expected={item.get('expected')} | "
                    f"Actual={item.get('actual')} | "
                    f"DB{item.get('db')}.DBD{item.get('byte')}"
                )

            if len(mismatches) > 8:
                mismatch_lines.append(f"... and {len(mismatches) - 8} more mismatches")

            msg += "\n\nPLC Mismatches:\n" + "\n".join(mismatch_lines)

        return msg


    # Backward-compatible alias if any old GUI code calls this.
    def edit_selected_recipe(self):
        self.enter_edit_mode()

    def close_page(self):
        if self.on_close:
            self.on_close()