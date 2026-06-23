# src/PAGES/action_code_plan_page.py
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from PyQt5.QtCore import Qt, QSize  # type: ignore
from PyQt5.QtGui import QColor, QPixmap  # type: ignore
from PyQt5.QtWidgets import (  # type: ignore
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QFrame, QLabel,
    QSizePolicy, QToolButton, QTableWidget, QTableWidgetItem,
    QAbstractItemView, QHeaderView, QLineEdit, QScrollArea,
    QGraphicsDropShadowEffect, QPushButton, QComboBox, QMessageBox,
    QCheckBox, QFileDialog, QDialog
)

from src.COMMON.action_code_catalog_db import (
    ensure_action_catalog_collections,
    seed_default_action_catalog,
    get_catalog_versions,
    get_current_catalog_version,
    get_action_catalog_header,
    get_action_catalog_sections,
    create_draft_from_version,
    delete_draft_catalog_version,
    publish_catalog_version,
    save_catalog_rows,
    save_header,
    import_catalog_payload,
    get_catalog_image_bytes,
)

try:
    # Optional: used only for Import PDF button.
    from tools.import_osc_catalog_from_pdf import parse_catalog_tables, crop_section_images  # type: ignore
except Exception:  # pragma: no cover
    parse_catalog_tables = None
    crop_section_images = None


PURPLE = "#571c86"
PURPLE_2 = "#764ba2"
GREEN = "#198754"
RED = "#dc3545"
BG = "#f7f4ff"


class ActionCodePlanPage(QWidget):
    """Versioned, editable OSC Action Code Catalog UI.

    Production behavior:
    - ACTIVE versions are locked/read-only.
    - Operator changes are made only in a cloned DRAFT.
    - Publish makes the draft current and archives the previous active version.
    """

    def __init__(self, parent=None, operator: str = "operator"):
        super().__init__(parent)
        self.operator = operator
        self.current_version_id: Optional[str] = None
        self.is_draft = False
        self.header_fields: Dict[str, QLineEdit] = {}
        self.section_tables: Dict[str, QTableWidget] = {}
        self.section_widgets: Dict[str, QFrame] = {}
        self.section_meta: Dict[str, Dict[str, Any]] = {}
        self._build_ui()
        self.reload_all()

    # ------------------------------------------------------------------
    # Style helpers
    # ------------------------------------------------------------------
    def _card(self) -> QFrame:
        frame = QFrame()
        frame.setStyleSheet("QFrame { background-color: white; border-radius: 12px; border: none; }")
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(18)
        shadow.setXOffset(0)
        shadow.setYOffset(3)
        shadow.setColor(QColor(0, 0, 0, 25))
        frame.setGraphicsEffect(shadow)
        return frame

    def _button(self, text: str, kind: str = "primary") -> QPushButton:
        btn = QPushButton(text)
        btn.setMinimumHeight(34)
        bg = PURPLE if kind == "primary" else GREEN if kind == "success" else RED if kind == "danger" else "#ffffff"
        fg = "white" if kind in ("primary", "success", "danger") else PURPLE
        border = "none" if kind in ("primary", "success", "danger") else f"1px solid {PURPLE}"
        btn.setStyleSheet(f"""
            QPushButton {{
                background: {bg}; color: {fg}; border: {border}; border-radius: 8px;
                padding: 6px 12px; font: 700 10px 'Segoe UI';
            }}
            QPushButton:disabled {{ background: #e9ecef; color: #6c757d; border: none; }}
            QPushButton:hover:!disabled {{ opacity: 0.9; }}
        """)
        return btn

    def _label(self, text: str) -> QLabel:
        lab = QLabel(text)
        lab.setStyleSheet("font: 700 9px 'Segoe UI'; color: #495057; letter-spacing: 0.4px;")
        return lab

    def _line(self, text: str = "") -> QLineEdit:
        le = QLineEdit(text)
        le.setMinimumHeight(28)
        le.setStyleSheet("""
            QLineEdit { font: 10px 'Segoe UI'; padding: 4px 8px; border: 1px solid #dee2e6;
                        border-radius: 6px; background: #f8f9fa; color: #212529; }
            QLineEdit:focus { border: 1px solid #571c86; background: white; }
            QLineEdit:read-only { color: #6c757d; background: #f8f9fa; }
        """)
        return le

    def _style_table(self, table: QTableWidget):
        table.setAlternatingRowColors(True)
        table.setShowGrid(True)
        table.setStyleSheet("""
            QTableWidget {
                font: 11px 'Segoe UI';
                gridline-color: #eadfff;
                border: 1px solid #eadfff;
                border-radius: 9px;
                background-color: #ffffff;
                alternate-background-color: #faf6ff;
                selection-background-color: #e6d8ff;
            }
            QTableWidget::item {
                padding: 6px;
                color: #212529;
                border-bottom: 1px solid #f0e6ff;
            }
            QTableWidget::item:selected {
                background-color: #e6d8ff;
                color: #212529;
            }
            QHeaderView::section {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #571c86, stop:1 #3a2fa3);
                color: white;
                padding: 8px 6px;
                border: none;
                font: 800 10px 'Segoe UI';
                min-height: 28px;
            }
        """)
        table.verticalHeader().setVisible(False)
        table.setSelectionBehavior(QAbstractItemView.SelectRows)
        table.setWordWrap(True)
        table.setAlternatingRowColors(True)

    # ------------------------------------------------------------------
    # Accordion
    # ------------------------------------------------------------------
    class AccordionSection(QFrame):  # type: ignore
        def __init__(self, title: str, default_open: bool = False, parent=None):
            super().__init__(parent)
            self.main_layout = QVBoxLayout(self)
            self.main_layout.setContentsMargins(0, 0, 0, 0)
            self.main_layout.setSpacing(5)

            self.header_button = QToolButton()
            self.header_button.setText(title)
            self.header_button.setCheckable(True)
            self.header_button.setChecked(default_open)
            self.header_button.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
            self.header_button.setArrowType(Qt.DownArrow if default_open else Qt.RightArrow)
            self.header_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            self.header_button.setMinimumHeight(42)
            self.header_button.setStyleSheet("""
                QToolButton {
                    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #571c86, stop:1 #764ba2);
                    border-radius: 8px; border: none; padding: 0 12px; color: white;
                    font: 700 11px 'Segoe UI'; text-align: left;
                }
            """)
            self.content_widget = QFrame()
            self.content_widget.setStyleSheet("QFrame { border: none; background-color: transparent; }")
            self.content_layout = QVBoxLayout(self.content_widget)
            self.content_layout.setContentsMargins(10, 10, 10, 14)
            self.content_layout.setSpacing(10)
            self.content_widget.setVisible(default_open)
            self.header_button.clicked.connect(self._toggle)
            self.main_layout.addWidget(self.header_button)
            self.main_layout.addWidget(self.content_widget)

        def _toggle(self, checked: bool):
            self.content_widget.setVisible(checked)
            self.header_button.setArrowType(Qt.DownArrow if checked else Qt.RightArrow)

    # ------------------------------------------------------------------
    # Build UI
    # ------------------------------------------------------------------
    def _build_ui(self):
        self.setStyleSheet(f"""
            QWidget {{
                background: {BG};
            }}

            QToolTip {{
                background-color: #ffffff;
                color: #571c86;
                border: 1px solid #d7c7ff;
                padding: 6px 8px;
                border-radius: 6px;
                font: 10px 'Segoe UI';
            }}
        """)
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(14, 14, 14, 14)
        main_layout.setSpacing(10)

        # ------------------------------------------------------------------
        # Top toolbar - two-line layout to avoid congested button crowding
        # ------------------------------------------------------------------
        toolbar = self._card()
        tl = QGridLayout(toolbar)
        tl.setContentsMargins(14, 10, 14, 10)
        tl.setHorizontalSpacing(10)
        tl.setVerticalSpacing(8)

        self.version_combo = QComboBox()
        self.version_combo.setMinimumWidth(300)
        self.version_combo.currentIndexChanged.connect(self.on_version_changed)
        self.version_combo.setStyleSheet("""
            QComboBox {
                background: white; border: 1px solid #ddd; border-radius: 7px;
                padding: 6px 8px; font: 10px 'Segoe UI'; min-height: 24px;
            }
        """)

        self.search_line = self._line()
        self.search_line.setPlaceholderText("Search condition, description, action code...")
        self.search_line.textChanged.connect(self.apply_filter)

        self.side_filter = QComboBox()
        self.side_filter.addItems(["ALL", "tread", "shoulder", "sidewall", "bead", "innerliner", "curing", "foreign_material"])
        self.side_filter.currentIndexChanged.connect(self.apply_filter)
        self.side_filter.setMinimumWidth(150)
        self.side_filter.setStyleSheet("""
            QComboBox {
                background: white; border: 1px solid #ddd; border-radius: 7px;
                padding: 6px 8px; font: 10px 'Segoe UI'; min-height: 24px;
            }
        """)

        self.btn_import_pdf = self._button("Import SOP PDF", "outline")
        self.btn_draft = self._button("Create Editable Draft", "primary")
        self.btn_save = self._button("Save Draft", "success")
        self.btn_delete_draft = self._button("Delete Draft", "danger")
        self.btn_publish = self._button("Publish Draft", "danger")
        self.btn_reload = self._button("Reload", "outline")

        self.btn_import_pdf.clicked.connect(self.import_pdf_clicked)
        self.btn_draft.clicked.connect(self.create_draft_clicked)
        self.btn_save.clicked.connect(self.save_clicked)
        self.btn_delete_draft.clicked.connect(self.delete_draft_clicked)
        self.btn_publish.clicked.connect(self.publish_clicked)
        self.btn_reload.clicked.connect(self.reload_all)

        rev_label = QLabel("Revision")
        rev_label.setStyleSheet("font: 700 10px 'Segoe UI'; color: #495057;")
        filter_label = QLabel("Filter")
        filter_label.setStyleSheet("font: 700 10px 'Segoe UI'; color: #495057;")

        tl.addWidget(rev_label, 0, 0)
        tl.addWidget(self.version_combo, 0, 1)
        tl.addWidget(self.search_line, 0, 2)
        tl.addWidget(filter_label, 0, 3)
        tl.addWidget(self.side_filter, 0, 4)
        tl.addWidget(self.btn_reload, 0, 5)

        action_row = QWidget()
        ar = QHBoxLayout(action_row)
        ar.setContentsMargins(0, 0, 0, 0)
        ar.setSpacing(8)
        ar.addStretch()
        ar.addWidget(self.btn_import_pdf)
        ar.addWidget(self.btn_draft)
        ar.addWidget(self.btn_save)
        ar.addWidget(self.btn_delete_draft)
        ar.addWidget(self.btn_publish)
        tl.addWidget(action_row, 1, 0, 1, 6)
        tl.setColumnStretch(2, 1)

        main_layout.addWidget(toolbar)

        # ------------------------------------------------------------------
        # Document header card
        # ------------------------------------------------------------------
        self.header_frame = self._card()
        hl = QGridLayout(self.header_frame)
        hl.setContentsMargins(16, 12, 16, 12)
        hl.setHorizontalSpacing(18)
        hl.setVerticalSpacing(7)

        title = QLabel("Action Code Plan - Quality Control Documentation")
        title.setStyleSheet(f"font: 800 13px 'Segoe UI'; color: {PURPLE};")
        hl.addWidget(title, 0, 0, 1, 4)

        fields = [
            ("document_name", "DOCUMENT NAME"), ("date_of_release", "DATE OF RELEASE"),
            ("document_no", "DOCUMENT NO."), ("date_of_applicability", "DATE OF APPLICABILITY"),
            ("revision_no", "REVISION NO."), ("process_owner", "PROCESS OWNER"),
            ("document_status", "DOCUMENT STATUS"), ("security_classification", "SECURITY CLASSIFICATION"),
        ]
        for idx, (key, label) in enumerate(fields, start=1):
            row = 1 + (idx - 1) // 2
            col = 0 if idx % 2 == 1 else 2
            hl.addWidget(self._label(label), row, col)
            le = self._line()
            self.header_fields[key] = le
            hl.addWidget(le, row, col + 1)
        hl.setColumnStretch(1, 1)
        hl.setColumnStretch(3, 1)
        main_layout.addWidget(self.header_frame)

        # ------------------------------------------------------------------
        # Body scroll: page-level scrolling only. Tables do not scroll inside.
        # ------------------------------------------------------------------
        self.body_frame = self._card()
        bl = QVBoxLayout(self.body_frame)
        bl.setContentsMargins(14, 14, 14, 14)
        bl.setSpacing(10)

        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("""
            QLabel {
                font: 700 10px 'Segoe UI'; color: #495057;
                background: #f8f5ff; border: 1px solid #eadfff;
                border-radius: 7px; padding: 7px 10px;
            }
        """)
        bl.addWidget(self.status_label)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFrameShape(QFrame.NoFrame)
        self.scroll_area.setStyleSheet("QScrollArea { border: none; background: transparent; }")
        self.scroll_content = QWidget()
        self.scroll_layout = QVBoxLayout(self.scroll_content)
        self.scroll_layout.setContentsMargins(0, 0, 0, 0)
        self.scroll_layout.setSpacing(12)
        self.scroll_area.setWidget(self.scroll_content)
        bl.addWidget(self.scroll_area, 1)
        main_layout.addWidget(self.body_frame, 1)

    # ------------------------------------------------------------------
    # Load/reload
    # ------------------------------------------------------------------
    def reload_all(self):
        try:
            ensure_action_catalog_collections()
            seed_default_action_catalog(force=False)
            self.reload_versions()
            self.load_current_version()
        except Exception as e:
            QMessageBox.critical(self, "OSC Catalog Error", str(e))

    def reload_versions(self):
        self.version_combo.blockSignals(True)
        self.version_combo.clear()
        versions = get_catalog_versions(include_archived=True)
        if not versions:
            self.version_combo.blockSignals(False)
            return
        for v in versions:
            tag = "CURRENT" if v.get("is_current") else v.get("status", "")
            text = f"{v.get('version_id')}  [{tag}]"
            self.version_combo.addItem(text, v.get("version_id"))
        self.version_combo.blockSignals(False)

    def on_version_changed(self):
        version_id = self.version_combo.currentData()
        if version_id:
            self.load_current_version(version_id)

    def load_current_version(self, version_id: Optional[str] = None):
        if not version_id:
            current = get_current_catalog_version()
            version_id = current.get("version_id") if current else self.version_combo.currentData()
        self.current_version_id = version_id
        header = get_action_catalog_header(version_id)
        for key, le in self.header_fields.items():
            le.setText(str(header.get(key, "")))

        versions = get_catalog_versions(include_archived=True)
        selected = next((v for v in versions if v.get("version_id") == version_id), {})
        self.is_draft = selected.get("status") == "DRAFT" and not selected.get("locked", False)
        self.set_edit_mode(self.is_draft)
        self.build_sections()
        self.status_label.setText(f"Loaded {version_id or ''} | Status: {selected.get('status', 'UNKNOWN')} | Editable: {'YES' if self.is_draft else 'NO'}")

    def set_edit_mode(self, editable: bool):
        for le in self.header_fields.values():
            le.setReadOnly(not editable)

        self.btn_save.setEnabled(editable)
        self.btn_delete_draft.setEnabled(editable)
        self.btn_publish.setEnabled(editable)
        self.btn_draft.setEnabled(not editable)

    # ------------------------------------------------------------------
    # Section rendering
    # ------------------------------------------------------------------
    def clear_sections(self):
        self.section_tables.clear()
        self.section_widgets.clear()
        self.section_meta.clear()

        while self.scroll_layout.count():
            item = self.scroll_layout.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()

    def build_sections(self):
        self.clear_sections()
        sections = get_action_catalog_sections(self.current_version_id, include_images=True, include_inactive=True)
        if not sections:
            msg = QLabel("No OSC rows found. Use Import SOP PDF to load the client catalogue.")
            msg.setAlignment(Qt.AlignCenter)
            msg.setStyleSheet(f"font: 700 13px 'Segoe UI'; color: {RED}; padding: 30px;")
            self.scroll_layout.addWidget(msg)
            self.scroll_layout.addStretch()
            return

        for idx, sec in enumerate(sections):
            title = f"{sec.get('catalog_code')} | {sec.get('section_name')}    [{sec.get('side')}]"
            acc = self.AccordionSection(title, default_open=(idx < 2))
            table = self.create_section_table(sec.get("rows", []))
            code = str(sec.get("catalog_code"))
            self.section_tables[code] = table
            self.section_widgets[code] = acc
            self.section_meta[code] = {
                "side": str(sec.get("side", "")),
                "catalog_code": code,
                "section_name": str(sec.get("section_name", "")),
            }

            acc.content_layout.addWidget(table)
            img_widget = self.create_image_gallery(sec.get("images", []))
            if img_widget:
                acc.content_layout.addWidget(img_widget)
            self.scroll_layout.addWidget(acc)
        self.scroll_layout.addStretch()
        self.apply_filter()

    def create_section_table(self, rows: List[Dict[str, Any]]) -> QTableWidget:
        headers = ["Condition", "Description", "Action", "OE", "Replacement", "Scrap", "Active"]
        table = QTableWidget(len(rows), len(headers))
        table.setHorizontalHeaderLabels(headers)
        self._style_table(table)

        # Important production UI rule:
        # page scroll only; each table expands to show all rows.
        table.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        table.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        table.setEditTriggers(
            QAbstractItemView.DoubleClicked | QAbstractItemView.EditKeyPressed
            if self.is_draft else QAbstractItemView.NoEditTriggers
        )
        table.setProperty("rows_meta", rows)

        h = table.horizontalHeader()
        h.setStretchLastSection(False)
        h.setSectionResizeMode(0, QHeaderView.Fixed)
        h.setSectionResizeMode(1, QHeaderView.Stretch)
        h.setSectionResizeMode(2, QHeaderView.Fixed)
        h.setSectionResizeMode(3, QHeaderView.Fixed)
        h.setSectionResizeMode(4, QHeaderView.Fixed)
        h.setSectionResizeMode(5, QHeaderView.Fixed)
        h.setSectionResizeMode(6, QHeaderView.Fixed)
        h.resizeSection(0, 90)
        h.resizeSection(2, 80)
        h.resizeSection(3, 48)
        h.resizeSection(4, 92)
        h.resizeSection(5, 58)
        h.resizeSection(6, 64)

        total_row_height = 0
        for r, row in enumerate(rows):
            values = [
                str(row.get("condition_code", "")),
                str(row.get("description", "")),
                str(row.get("action_code", "")),
                "X" if row.get("oe") else "",
                "X" if row.get("replacement") else "",
                "X" if row.get("scrap") else "",
                "YES" if row.get("active", True) else "NO",
            ]
            for c, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setTextAlignment(Qt.AlignLeft | Qt.AlignVCenter if c == 1 else Qt.AlignCenter)
                if c == 0:
                    item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                table.setItem(r, c, item)

            desc_len = len(values[1])
            row_h = 38 if desc_len <= 85 else 54 if desc_len <= 150 else 68
            table.setRowHeight(r, row_h)
            total_row_height += row_h

        # Header + rows + border. This removes the congested internal scrollbar.
        table_height = 34 + total_row_height + 4
        table.setMinimumHeight(table_height)
        table.setMaximumHeight(table_height)
        return table

    def _pixmap_from_catalog_image(self, image_doc: Dict[str, Any]) -> QPixmap:
        pix = QPixmap()

        try:
            data = get_catalog_image_bytes(image_doc)
            if data:
                pix.loadFromData(data)
                return pix
        except Exception:
            pass

        path = image_doc.get("image_path")
        if path and os.path.exists(str(path)):
            pix.load(str(path))

        return pix
    
    def create_image_gallery(self, images: List[Dict[str, Any]]) -> Optional[QFrame]:
        if not images:
            return None

        frame = QFrame()
        frame.setStyleSheet("""
            QFrame {
                background: #fbfaff;
                border: 1px solid #eadfff;
                border-radius: 10px;
            }
        """)

        outer = QVBoxLayout(frame)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(6)

        title = QLabel(f"Reference images ({len(images)})")
        title.setStyleSheet("""
            font: 700 10px 'Segoe UI';
            color: #571c86;
            border: none;
            background: transparent;
        """)
        outer.addWidget(title)

        grid_holder = QWidget()
        grid_holder.setStyleSheet("background: transparent; border: none;")

        grid = QGridLayout(grid_holder)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(10)

        max_show = min(len(images), 8)

        for idx, img in enumerate(images[:max_show]):
            tile = QLabel()
            tile.setMinimumSize(200, 125)
            tile.setMaximumSize(240, 150)
            tile.setAlignment(Qt.AlignCenter)
            tile.setCursor(Qt.PointingHandCursor)

            tile.setStyleSheet("""
                QLabel {
                    background: white;
                    border: 1px solid #d7c7ff;
                    border-radius: 8px;
                    color: #6c757d;
                    padding: 3px;
                }
            """)

            # No tooltip, to avoid black hover popup.
            tile.setToolTip("")

            # New support:
            # 1. MongoDB GridFS image using gridfs_file_id
            # 2. Fallback old local image_path
            pix = self._pixmap_from_catalog_image(img)

            if not pix.isNull():
                tile.setPixmap(
                    pix.scaled(
                        QSize(228, 138),
                        Qt.KeepAspectRatio,
                        Qt.SmoothTransformation,
                    )
                )
            else:
                tile.setText("Image\nnot loaded")

            # Open popup using full image document.
            # Popup should load from GridFS or fallback path.
            tile.mousePressEvent = lambda event, d=dict(img): self.open_image_popup(d)

            grid.addWidget(tile, idx // 4, idx % 4)

        if len(images) > max_show:
            more = QLabel(f"+{len(images) - max_show} more images available")
            more.setStyleSheet("""
                font: 700 10px 'Segoe UI';
                color: #571c86;
                border: none;
                background: transparent;
            """)
            grid.addWidget(more, (max_show // 4) + 1, 0, 1, 4)

        outer.addWidget(grid_holder)

        return frame


    def open_image_popup(self, image_ref):
        pix = QPixmap()
        title_text = "OSC Reference Image"

        if isinstance(image_ref, dict):
            title_text = str(
                image_ref.get("image_name")
                or image_ref.get("file_name")
                or image_ref.get("gridfs_file_id")
                or "OSC Reference Image"
            )

            try:
                data = get_catalog_image_bytes(image_ref)
                if data:
                    pix.loadFromData(data)
            except Exception:
                pass

        else:
            image_path = str(image_ref or "")
            title_text = os.path.basename(image_path)

            if image_path and os.path.exists(image_path):
                pix.load(image_path)

        if pix.isNull():
            QMessageBox.warning(
                self,
                "Image Load Failed",
                "Could not load OSC reference image from MongoDB/GridFS."
            )
            return

        dialog = QDialog(self)
        dialog.setWindowTitle(title_text)
        dialog.setModal(False)
        dialog.resize(980, 720)
        dialog.setStyleSheet("""
            QDialog { background: #f7f4ff; }
            QLabel#titleLabel {
                font: 800 13px 'Segoe UI';
                color: #571c86;
                padding: 8px;
            }
            QLabel#imageLabel {
                background: white;
                border: 1px solid #d7c7ff;
                border-radius: 10px;
                padding: 10px;
            }
            QPushButton {
                background: #571c86;
                color: white;
                border: none;
                border-radius: 8px;
                padding: 8px 18px;
                font: 800 11px 'Segoe UI';
            }
        """)

        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)

        title = QLabel(title_text)
        title.setObjectName("titleLabel")
        layout.addWidget(title)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)

        img_label = QLabel()
        img_label.setObjectName("imageLabel")
        img_label.setAlignment(Qt.AlignCenter)
        img_label.setMinimumSize(900, 580)
        img_label.setPixmap(
            pix.scaled(
                QSize(900, 580),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
        )

        scroll.setWidget(img_label)
        layout.addWidget(scroll, 1)

        btn_row = QHBoxLayout()
        btn_row.addStretch()

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dialog.close)
        btn_row.addWidget(close_btn)

        layout.addLayout(btn_row)

        dialog.show()
        dialog.raise_()
        dialog.activateWindow()

    # ------------------------------------------------------------------
    # Filtering
    # ------------------------------------------------------------------
    def apply_filter(self):
        search = self.search_line.text().strip().lower() if hasattr(self, "search_line") else ""
        selected_side = self.side_filter.currentText() if hasattr(self, "side_filter") else "ALL"

        visible_sections = 0

        for code, table in self.section_tables.items():
            section_widget = self.section_widgets.get(code)
            meta = self.section_meta.get(code, {})
            section_side = str(meta.get("side", ""))

            side_ok = selected_side == "ALL" or section_side == selected_side

            row_match_count = 0

            for r in range(table.rowCount()):
                row_text = " ".join(
                    table.item(r, c).text() if table.item(r, c) else ""
                    for c in range(table.columnCount())
                ).lower()

                search_ok = not search or search in row_text
                show_row = side_ok and search_ok

                table.setRowHidden(r, not show_row)

                if show_row:
                    row_match_count += 1

            show_section = side_ok and row_match_count > 0

            if section_widget:
                section_widget.setVisible(show_section)

            if show_section:
                visible_sections += 1

        if visible_sections == 0:
            self.status_label.setText(
                f"Loaded {self.current_version_id or ''} | No matching OSC sections found for filter/search."
            )
        else:
            self.status_label.setText(
                f"Loaded {self.current_version_id or ''} | Showing sections: {visible_sections}"
            )

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------
    def create_draft_clicked(self):
        try:
            base_version_id = self.current_version_id                                         

            # Avoid duplicate draft creation for same active version.
            versions = get_catalog_versions(include_archived=True)
            existing_draft = next(
                (
                    v for v in versions
                    if v.get("status") == "DRAFT"
                    and v.get("source") == f"draft_from:{base_version_id}"
                ),
                None,
            )

            if existing_draft:
                answer = QMessageBox.question(
                    self,
                    "Draft Already Exists",
                    f"A draft already exists for this catalog:\n\n"
                    f"{existing_draft.get('version_id')}\n\n"
                    f"Open existing draft instead of creating another duplicate?",
                    QMessageBox.Yes | QMessageBox.No,
                )

                if answer == QMessageBox.Yes:
                    idx = self.version_combo.findData(existing_draft.get("version_id"))
                    if idx >= 0:
                        self.version_combo.setCurrentIndex(idx)
                    self.load_current_version(existing_draft.get("version_id"))
                    return

            draft = create_draft_from_version(base_version_id, operator=self.operator)

            QMessageBox.information(
                self,
                "Draft Created",
                f"Editable draft created:\n{draft.get('version_id')}"
            )

            self.reload_versions()
            idx = self.version_combo.findData(draft.get("version_id"))
            if idx >= 0:
                self.version_combo.setCurrentIndex(idx)
            self.load_current_version(draft.get("version_id"))

        except Exception as e:
            QMessageBox.critical(self, "Create Draft Failed", str(e))

    def save_clicked(self):
        if not self.current_version_id or not self.is_draft:
            QMessageBox.warning(self, "Read Only", "Create an editable draft before saving changes.")
            return
        try:
            header_updates = {k: v.text().strip() for k, v in self.header_fields.items()}
            save_header(self.current_version_id, header_updates, operator=self.operator)

            rows_to_save: List[Dict[str, Any]] = []
            for table in self.section_tables.values():
                for r in range(table.rowCount()):
                    rows_to_save.append({
                        "condition_code": table.item(r, 0).text().strip() if table.item(r, 0) else "",
                        "description": table.item(r, 1).text().strip() if table.item(r, 1) else "",
                        "action_code": table.item(r, 2).text().strip() if table.item(r, 2) else "",
                        "oe": (table.item(r, 3).text().strip().upper() == "X") if table.item(r, 3) else False,
                        "replacement": (table.item(r, 4).text().strip().upper() == "X") if table.item(r, 4) else False,
                        "scrap": (table.item(r, 5).text().strip().upper() == "X") if table.item(r, 5) else False,
                        "active": (table.item(r, 6).text().strip().upper() != "NO") if table.item(r, 6) else True,
                    })
            result = save_catalog_rows(self.current_version_id, rows_to_save, operator=self.operator)
            QMessageBox.information(self, "Saved", f"Draft saved. Updated rows: {result.get('updated_rows')}")
            self.load_current_version(self.current_version_id)
        except Exception as e:
            QMessageBox.critical(self, "Save Failed", str(e))

    def delete_draft_clicked(self):
        if not self.current_version_id or not self.is_draft:
            QMessageBox.warning(
                self,
                "Cannot Delete",
                "Only an unpublished DRAFT version can be deleted."
            )
            return

        answer = QMessageBox.question(
            self,
            "Delete Draft",
            f"Delete this draft from MongoDB?\n\n{self.current_version_id}\n\nThis will remove draft rows and draft image references. Active catalog will not be affected.",
            QMessageBox.Yes | QMessageBox.No,
        )

        if answer != QMessageBox.Yes:
            return

        try:
            result = delete_draft_catalog_version(
                self.current_version_id,
                operator=self.operator,
            )

            QMessageBox.information(
                self,
                "Draft Deleted",
                f"Draft deleted successfully.\n\n"
                f"Rows deleted: {result.get('deleted_rows')}\n"
                f"Image refs deleted: {result.get('deleted_images')}"
            )

            self.reload_versions()
            self.load_current_version()

        except Exception as e:
            QMessageBox.critical(self, "Delete Draft Failed", str(e))

    def publish_clicked(self):
        if not self.current_version_id or not self.is_draft:
            return
        answer = QMessageBox.question(
            self,
            "Publish Draft",
            "Publish this draft as CURRENT ACTIVE catalog?\nPrevious active version will be archived. This is the production traceability point.",
            QMessageBox.Yes | QMessageBox.No,
        )
        if answer != QMessageBox.Yes:
            return
        try:
            self.save_clicked()
            publish_catalog_version(self.current_version_id, operator=self.operator)
            QMessageBox.information(self, "Published", f"Catalog published:\n{self.current_version_id}")
            self.reload_all()
        except Exception as e:
            QMessageBox.critical(self, "Publish Failed", str(e))

    def import_pdf_clicked(self):
        if parse_catalog_tables is None or crop_section_images is None:
            QMessageBox.warning(self, "Importer Missing", "tools/import_osc_catalog_from_pdf.py is not available. Copy the tools folder into project root.")
            return
        pdf_path, _ = QFileDialog.getOpenFileName(self, "Select OSC SOP PDF", "", "PDF Files (*.pdf)")
        if not pdf_path:
            return
        out_dir = os.path.join("media", "osc_catalog", "rev03")
        try:
            payload = parse_catalog_tables(__import__("pathlib").Path(pdf_path))
            payload["images"] = crop_section_images(__import__("pathlib").Path(pdf_path), payload, __import__("pathlib").Path(out_dir))
            result = import_catalog_payload(payload, replace=True, publish=True, operator=self.operator)
            QMessageBox.information(self, "Import Completed", f"Imported rows: {result.get('row_count')}\nImages: {result.get('image_count')}\nVersion: {result.get('version_id')}")
            self.reload_all()
        except Exception as e:
            QMessageBox.critical(self, "Import Failed", str(e))
