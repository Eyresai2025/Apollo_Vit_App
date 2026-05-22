# src/PAGES/action_code_plan_page.py
from PyQt5.QtCore import Qt  # type: ignore
from PyQt5.QtGui import QColor  # type: ignore
from PyQt5.QtWidgets import (  # type: ignore
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QFrame, QLabel,
    QSizePolicy, QToolButton, QTableWidget, QTableWidgetItem,
    QAbstractItemView, QHeaderView, QLineEdit, QScrollArea,
    QGraphicsDropShadowEffect
)

from src.COMMON.action_code_catalog_db import (
    seed_default_action_catalog,
    get_action_catalog_header,
    get_action_catalog_sections,
)
class ActionCodePlanPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()

    # ---------- Helper: table style ----------
    def _style_action_table_variant1(self, table: QTableWidget):
        table.setAlternatingRowColors(False)
        table.setStyleSheet("""
            QTableWidget {
                font: 11px 'Segoe UI';
                gridline-color: #e0d3ff;
                border: none;
                border-radius: 8px;
                background-color: #f3e8ff;
            }
            QTableWidget::item {
                background-color: #f3e8ff;
                padding: 6px;
                border: none;
                color: #212529;
            }
            QTableWidget::item:alternate {
                background-color: #f3e8ff;
            }
            QTableWidget::item:selected {
                background-color: #e0d0ff;
                color: #212529;
            }
            QHeaderView::section {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #571c86, stop:1 #3a2fa3);
                color: white;
                padding: 10px 6px;
                border: none;
                font: 600 10px 'Segoe UI';
                text-transform: uppercase;
                letter-spacing: 0.6px;
            }
        """)

    # ---------- Helper: Accordion section ----------
    class AccordionSection(QFrame):  # type: ignore
        def __init__(self, section_code: str, title: str, default_open=True, parent=None):
            super().__init__(parent)
            self.setStyleSheet("QFrame { border: none; }")

            self.main_layout = QVBoxLayout(self)  # type: ignore
            self.main_layout.setContentsMargins(0, 0, 0, 0)
            self.main_layout.setSpacing(4)

            self.header_button = QToolButton()  # type: ignore
            self.header_button.setText(f"{section_code}  |  {title}")
            self.header_button.setCheckable(True)
            self.header_button.setChecked(default_open)
            self.header_button.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)  # type: ignore
            self.header_button.setArrowType(Qt.DownArrow if default_open else Qt.RightArrow)  # type: ignore
            self.header_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)  # type: ignore
            self.header_button.setMinimumHeight(38)
            self.header_button.setStyleSheet("""
                QToolButton {
                    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                        stop:0 #571c86, stop:1 #764ba2);
                    border-radius: 8px;
                    border: none;
                    padding: 0 12px;
                    color: white;
                    font: 600 11px 'Segoe UI';
                    text-align: left;
                }
                QToolButton::checked {
                    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                        stop:0 #4b1174, stop:1 #6b3a9d);
                }
            """)

            self.content_widget = QFrame()  # type: ignore
            self.content_widget.setStyleSheet("QFrame { border: none; background-color: transparent; }")
            self.content_layout = QVBoxLayout(self.content_widget)  # type: ignore
            self.content_layout.setContentsMargins(8, 8, 8, 16)
            self.content_layout.setSpacing(10)
            self.content_widget.setVisible(default_open)

            self.header_button.clicked.connect(self._toggle)

            self.main_layout.addWidget(self.header_button)
            self.main_layout.addWidget(self.content_widget)

        def _toggle(self, checked: bool):
            self.content_widget.setVisible(checked)
            self.header_button.setArrowType(Qt.DownArrow if checked else Qt.RightArrow)  # type: ignore

    # ---------- Helper: image placeholder ----------
    def _make_image_placeholder(self):
        frame = QFrame()  # type: ignore
        frame.setStyleSheet("""
            QFrame {
                border: 2px dashed #ced4da;
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #f8f9fa, stop:1 #e9ecef);
                border-radius: 8px;
            }
        """)
        lay = QVBoxLayout(frame)  # type: ignore
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setAlignment(Qt.AlignCenter)  # type: ignore

        icon_lbl = QLabel("📷")  # type: ignore
        icon_lbl.setAlignment(Qt.AlignCenter)  # type: ignore
        icon_lbl.setStyleSheet("font-size: 34px;")
        lay.addWidget(icon_lbl)

        return frame

    # ---------- Helper: section table ----------
    def _create_section_table(self, rows):
        num_rows = len(rows) + 1

        table = QTableWidget(num_rows, 6)  # type: ignore
        table.setHorizontalHeaderLabels([
            "Condition", "Description of condition",
            "Action code", "Classification", "Replacement", "Scrap"
        ])
        table.verticalHeader().setVisible(False)
        table.setSelectionBehavior(QAbstractItemView.SelectRows)  # type: ignore
        table.setEditTriggers(QAbstractItemView.NoEditTriggers)  # type: ignore

        self._style_action_table_variant1(table)

        header = table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Interactive)  # type: ignore
        header.setStretchLastSection(True)

        header.resizeSection(0, 90)
        header.resizeSection(1, 420)
        header.resizeSection(2, 110)
        header.resizeSection(3, 130)
        header.resizeSection(4, 120)
        header.resizeSection(5, 100)

        for row_index, row_doc in enumerate(rows):
            values = [
                str(row_doc.get("condition_code", "")),
                str(row_doc.get("description", "")),
                str(row_doc.get("action_code", "")),
                str(row_doc.get("classification", "")),
                str(row_doc.get("replacement", "")),
                str(row_doc.get("scrap", "")),
            ]

            for col_index, value in enumerate(values):
                item = QTableWidgetItem(value)  # type: ignore

                if col_index == 1:
                    item.setTextAlignment(Qt.AlignVCenter | Qt.AlignLeft)  # type: ignore
                else:
                    item.setTextAlignment(Qt.AlignCenter)  # type: ignore

                table.setItem(row_index, col_index, item)

        img_row = num_rows - 1
        table.setRowHeight(img_row, 170)

        container = QWidget()  # type: ignore
        hbox = QHBoxLayout(container)  # type: ignore
        hbox.setContentsMargins(40, 8, 40, 8)
        hbox.setSpacing(24)

        hbox.addWidget(self._make_image_placeholder(), 1)
        hbox.addWidget(self._make_image_placeholder(), 1)

        table.setCellWidget(img_row, 0, container)
        table.setSpan(img_row, 0, 1, 6)

        return table

    # ---------- UI ----------
    def _build_ui(self):
        self.setStyleSheet("""
            QWidget {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #f7f4ff, stop:1 #ece6ff);
            }
        """)

        main_layout = QVBoxLayout(self)  # type: ignore
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(8)

        # ===== Header Card =====
        header_frame = QFrame()  # type: ignore
        header_frame.setMaximumHeight(150)
        header_frame.setStyleSheet("""
            QFrame {
                background-color: white;
                border-radius: 12px;
                border: none;
            }
        """)

        shadow = QGraphicsDropShadowEffect()  # type: ignore
        shadow.setBlurRadius(18)
        shadow.setXOffset(0)
        shadow.setYOffset(3)
        shadow.setColor(QColor(0, 0, 0, 25))  # type: ignore
        header_frame.setGraphicsEffect(shadow)

        header_layout = QGridLayout(header_frame)  # type: ignore
        header_layout.setContentsMargins(14, 8, 14, 8)
        header_layout.setHorizontalSpacing(18)
        header_layout.setVerticalSpacing(6)

        title_label = QLabel("Action Code Plan - Quality Control Documentation")  # type: ignore
        title_label.setStyleSheet("""
            font: bold 13px 'Segoe UI';
            color: #571c86;
            padding-bottom: 4px;
        """)
        header_layout.addWidget(title_label, 0, 0, 1, 4)

        def make_label(text):
            lab = QLabel(text)  # type: ignore
            lab.setStyleSheet("""
                font: 600 9px 'Segoe UI';
                color: #495057;
                letter-spacing: 0.6px;
            """)
            return lab

        def make_line(text=""):
            le = QLineEdit(text)  # type: ignore
            le.setMinimumHeight(26)
            le.setStyleSheet("""
                QLineEdit {
                    font: 10px 'Segoe UI';
                    padding: 2px 8px;
                    border: 1px solid #e0e0e0;
                    border-radius: 5px;
                    background-color: #f8f9fa;
                    color: #212529;
                }
                QLineEdit:focus {
                    border: 1px solid #571c86;
                    background-color: #ffffff;
                }
            """)
            return le

        row = 1
        header_layout.addWidget(make_label("DOCUMENT NAME"), row, 0)
        header_layout.addWidget(make_line("Global Off Standard Catalogue for PCR Tyres"), row, 1)
        header_layout.addWidget(make_label("DATE OF RELEASE"), row, 2)
        header_layout.addWidget(make_line(), row, 3)

        row += 1
        header_layout.addWidget(make_label("DOCUMENT NO."), row, 0)
        header_layout.addWidget(make_line("SOP-GQ&BE-001"), row, 1)
        header_layout.addWidget(make_label("DATE OF APPLICABILITY"), row, 2)
        header_layout.addWidget(make_line(), row, 3)

        row += 1
        header_layout.addWidget(make_label("REVISION NO."), row, 0)
        header_layout.addWidget(make_line("03"), row, 1)
        header_layout.addWidget(make_label("PROCESS OWNER"), row, 2)
        header_layout.addWidget(make_line("Corporate"), row, 3)

        row += 1
        header_layout.addWidget(make_label("DOCUMENT STATUS"), row, 0)
        status_line = make_line("Approved")
        status_line.setStyleSheet(status_line.styleSheet() + """
            QLineEdit { color: #28a745; font-weight: 600; }
        """)
        header_layout.addWidget(status_line, row, 1)
        header_layout.addWidget(make_label("SECURITY CLASSIFICATION"), row, 2)
        header_layout.addWidget(make_line("Internal"), row, 3)

        main_layout.addWidget(header_frame, stretch=0)

        # ===== Body Card (Scroll + Accordion) =====
        body_frame = QFrame()  # type: ignore
        body_frame.setStyleSheet("""
            QFrame {
                background-color: white;
                border-radius: 12px;
                border: none;
            }
        """)

        shadow2 = QGraphicsDropShadowEffect()  # type: ignore
        shadow2.setBlurRadius(20)
        shadow2.setXOffset(0)
        shadow2.setYOffset(4)
        shadow2.setColor(QColor(0, 0, 0, 30))  # type: ignore
        body_frame.setGraphicsEffect(shadow2)

        body_layout = QVBoxLayout(body_frame)  # type: ignore
        body_layout.setContentsMargins(16, 16, 16, 12)
        body_layout.setSpacing(10)

        scroll_area = QScrollArea()  # type: ignore
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.NoFrame)  # type: ignore
        scroll_area.setStyleSheet("""
            QScrollArea { border: none; background: transparent; }
            QScrollArea > QWidget > QWidget { background: transparent; }
        """)

        scroll_content = QWidget()  # type: ignore
        scroll_layout = QVBoxLayout(scroll_content)  # type: ignore
        scroll_layout.setContentsMargins(0, 0, 0, 0)
        scroll_layout.setSpacing(12)
        # ----- load section data from MongoDB -----
        try:
            # Safe: if DB is empty, it inserts your current 3 OQC sections.
            seed_default_action_catalog(force=False)

            sections = get_action_catalog_sections(revision_no="03")

            if not sections:
                empty_label = QLabel("No Action Code Catalog data found in MongoDB.")  # type: ignore
                empty_label.setAlignment(Qt.AlignCenter)  # type: ignore
                empty_label.setStyleSheet("font: 600 13px 'Segoe UI'; color: #dc3545;")
                scroll_layout.addWidget(empty_label)
            else:
                for index, section in enumerate(sections):
                    sec = self.AccordionSection(
                        str(section.get("catalog_code", "")),
                        str(section.get("section_name", "")),
                        default_open=(index == 0),
                    )
                    sec.content_layout.addWidget(
                        self._create_section_table(section.get("rows", []))
                    )
                    scroll_layout.addWidget(sec)

        except Exception as e:
            error_label = QLabel(f"Failed to load Action Code Catalog from MongoDB: {e}")  # type: ignore
            error_label.setAlignment(Qt.AlignCenter)  # type: ignore
            error_label.setStyleSheet("font: 600 13px 'Segoe UI'; color: #dc3545;")
            scroll_layout.addWidget(error_label)

        scroll_layout.addStretch()
        scroll_area.setWidget(scroll_content)

        body_layout.addWidget(scroll_area, stretch=1)
        main_layout.addWidget(body_frame, stretch=9)
