from __future__ import annotations

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


class RecipeManagementPage(QWidget):
    """
    F-043 / F-044 Recipe Management page.

    Purpose:
        - View saved SKU recipes
        - View latest/versioned recipe details
        - Check camera axis targets and laser axis targets separately
        - Check linked VIT model path
        - Engineering/manual load test to PLC if enabled

    Production note:
        This page is NOT the production active-recipe source.
        In production, PLC active SKU is the source of truth.
    """

    def __init__(
        self,
        media_path: str,
        env_path: str = "",
        on_close=None,
        on_edit_recipe=None,
        parent=None,
    ):
        super().__init__(parent)

        self.media_path = media_path
        self.env_path = env_path
        self.on_close = on_close
        self.on_edit_recipe = on_edit_recipe

        self.recipe_service = RecipeService(
            media_path=self.media_path,
            env_path=self.env_path or None,
        )

        self.recipe_col = get_collection("SKU Recipes")
        self.active_recipe_col = get_collection("Active Recipe")

        self.current_recipes: List[Dict[str, Any]] = []
        self.selected_recipe: Optional[Dict[str, Any]] = None

        self.sku_combo = None
        self.version_combo = None
        self.summary_lbl = None
        self.axis_table = None
        self.raw_json = None

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
            "View saved SKU recipes, check recipe versions, inspect axis targets, "
            "and perform engineering/manual recipe load tests."
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
        self.version_combo.setMinimumWidth(160)
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
        self.summary_lbl.setWordWrap(True)
        self.summary_lbl.setStyleSheet("""
            QLabel {
                background:#f4eefb;
                color:#49305f;
                border:1px solid #dfd2ef;
                border-radius:14px;
                padding:14px;
                font:600 11px 'Segoe UI';
            }
        """)
        main_l.addWidget(self.summary_lbl)

        # =====================================================
        # AXIS TABLE
        # =====================================================
        table_title = QLabel("Axis Targets")
        table_title.setObjectName("SectionTitle")
        main_l.addWidget(table_title)

        self.axis_table = QTableWidget()
        self.axis_table.setColumnCount(5)
        self.axis_table.setHorizontalHeaderLabels([
            "Group", "Axis", "Name", "Target Position", "Captured At"
        ])
        self.axis_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.axis_table.setAlternatingRowColors(True)
        self.axis_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.axis_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.axis_table.setMinimumHeight(230)
        main_l.addWidget(self.axis_table)

        # =====================================================
        # RAW JSON
        # =====================================================
        json_title = QLabel("Recipe JSON")
        json_title.setObjectName("SectionTitle")
        main_l.addWidget(json_title)

        self.raw_json = QTextEdit()
        self.raw_json.setReadOnly(True)
        self.raw_json.setMinimumHeight(210)
        self.raw_json.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        main_l.addWidget(self.raw_json, 1)

        # =====================================================
        # BUTTON ROW
        # =====================================================
        btn_row = QHBoxLayout()
        btn_row.setSpacing(10)

        edit_btn = self._secondary_btn("Edit / Open in New SKU")
        edit_btn.clicked.connect(self.edit_selected_recipe)

        mark_test_active_btn = self._secondary_btn("Mark Test Active")
        mark_test_active_btn.clicked.connect(self.mark_selected_recipe_as_test_active)

        load_plc_btn = self._primary_btn("Load Recipe to Machine")
        load_plc_btn.clicked.connect(self.load_selected_recipe_to_machine)

        close_btn = self._secondary_btn("Back")
        close_btn.clicked.connect(self.close_page)

        btn_row.addWidget(edit_btn)
        btn_row.addWidget(mark_test_active_btn)
        btn_row.addStretch(1)
        btn_row.addWidget(load_plc_btn)
        btn_row.addWidget(close_btn)

        main_l.addLayout(btn_row)

        root.addWidget(main_card)

    # =========================================================
    # DATA LOADING
    # =========================================================

    def refresh_recipes(self):
        try:
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
        tyre_size = _safe_text(recipe.get("tyre_size"))
        barcode = _safe_text(recipe.get("barcode_pattern") or recipe.get("barcode"))
        model_path = _safe_text(recipe.get("vit_model_path"), "Not linked yet")
        val_score = _safe_text(recipe.get("validation_score"), "Pending")
        created_at = _safe_text(recipe.get("created_at"))
        updated_at = _safe_text(recipe.get("updated_at"))
        author = _safe_text(recipe.get("author"), "operator")

        summary = (
            f"SKU: {sku_name}    |    Version: {version}    |    Status: {status}\n"
            f"Tyre Size: {tyre_size}    |    Barcode: {barcode}    |    Validation F1: {val_score}\n"
            f"Author: {author}    |    Created: {created_at}    |    Updated: {updated_at}\n"
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

    def _render_axis_table(self, recipe: Dict[str, Any]):
        rows = []

        camera_targets = recipe.get("camera_axis_targets", {}) or {}
        laser_targets = recipe.get("laser_axis_targets", {}) or {}

        def fmt_value(value):
            if value is None:
                return "-"
            try:
                return f"{float(value):.3f}"
            except Exception:
                return str(value)

        for axis_key, info in sorted(camera_targets.items()):
            if isinstance(info, dict):
                rows.append([
                    "CAMERA",
                    axis_key,
                    _safe_text(info.get("name")),
                    fmt_value(info.get("value")),
                    _safe_text(info.get("captured_at")),
                ])

        for axis_key, info in sorted(laser_targets.items()):
            if isinstance(info, dict):
                rows.append([
                    "LASER",
                    axis_key,
                    _safe_text(info.get("name")),
                    fmt_value(info.get("value")),
                    _safe_text(info.get("captured_at")),
                ])

        self.axis_table.setRowCount(len(rows))

        for row_idx, row_data in enumerate(rows):
            for col_idx, value in enumerate(row_data):
                item = QTableWidgetItem(str(value))
                item.setTextAlignment(Qt.AlignCenter)
                self.axis_table.setItem(row_idx, col_idx, item)

    # =========================================================
    # ACTIONS
    # =========================================================

    def mark_selected_recipe_as_test_active(self):
        """
        This is only for engineering/testing state.
        Production active SKU comes from PLC.
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
        Will write only if:
            DEPLOYMENT=True
            RECIPE_WRITE_TO_PLC=True
            PLC DB mapping is correct
        """
        recipe = self.selected_recipe

        if not recipe:
            QMessageBox.warning(self, "Recipe", "Please select a recipe first.")
            return

        reply = QMessageBox.question(
            self,
            "Load Recipe to Machine",
            (
                "This will attempt to write the selected recipe axis targets "
                "to PLC DB_Recipe if PLC writing is enabled.\n\n"
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

            QMessageBox.information(
                self,
                "PLC Recipe Load",
                result.get("message", str(result))
            )

        except Exception as e:
            QMessageBox.critical(self, "PLC Recipe Load Error", str(e))

    def edit_selected_recipe(self):
        recipe = self.selected_recipe

        if not recipe:
            QMessageBox.warning(self, "Recipe", "Please select a recipe first.")
            return

        if self.on_edit_recipe is None:
            QMessageBox.information(
                self,
                "Edit Recipe",
                "Edit callback is not connected."
            )
            return

        sku_meta = {
            "sku_name": recipe.get("sku_name", ""),
            "tyre_name": recipe.get("sku_name", ""),
            "tyre_size": recipe.get("tyre_size", ""),
            "barcode": recipe.get("barcode_pattern", ""),
            "barcode_pattern": recipe.get("barcode_pattern", ""),
            "operator": recipe.get("author", ""),
            "inspection_zones": recipe.get("inspection_zones", 5),
            "image_count_per_zone": recipe.get("image_count_per_zone", 20),
            "train_good_count": recipe.get("training_summary", {}).get("train_good_count", 10),
        }

        self.on_edit_recipe(sku_meta=sku_meta)

    def close_page(self):
        if self.on_close:
            self.on_close()