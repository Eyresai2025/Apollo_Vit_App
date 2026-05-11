# src/Pages/axis_status_page.py

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFrame, QLabel, QPushButton,
    QTableWidget, QTableWidgetItem, QHeaderView,
    QMessageBox
)
from src.COMMON.axis_status_service import AxisStatusService


class AxisStatusPage(QWidget):
    """
    Axis Status page.

    Read-only page:
    - Shows selected SKU / PLC SKU reference.
    - Shows 12 servo axis values.
    - Compares current position with recipe position/tolerance.
    - Does not write to PLC.
    """

    def __init__(self, media_path, env_path=None, on_close=None, parent=None):
        super().__init__(parent)

        self.media_path = media_path
        self.env_path = env_path
        self.on_close = on_close

        self.service = AxisStatusService(
            media_path=self.media_path,
            env_path=self.env_path,
        )

        self.refresh_timer = QTimer(self)
        self.refresh_timer.timeout.connect(self.refresh_axis_status)

        self._build_ui()

    # ------------------------------------------------------------
    # UI
    # ------------------------------------------------------------
    def _build_ui(self):
        self.setStyleSheet("QWidget { background-color: #f5f5f5; }")

        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(10)

        # Header
        header = QFrame()
        header.setStyleSheet("""
            QFrame {
                background:#571c86;
                border-radius:12px;
            }
        """)
        h = QHBoxLayout(header)
        h.setContentsMargins(16, 10, 16, 10)
        h.setSpacing(10)

        title = QLabel("Axis Status Monitor")
        title.setStyleSheet("font: 900 15px 'Segoe UI'; color:white; border:none;")
        h.addWidget(title)

        h.addStretch()

        self.refresh_state_lbl = QLabel("Auto Refresh: OFF")
        self.refresh_state_lbl.setStyleSheet(
            "font: 900 11px 'Segoe UI'; color:#ffcc00; border:none;"
        )
        h.addWidget(self.refresh_state_lbl)

        root.addWidget(header)

        # Top info panel
        info_panel = QFrame()
        info_panel.setStyleSheet("""
            QFrame {
                background:white;
                border-radius:14px;
                border:1px solid #ececec;
            }
        """)
        info = QHBoxLayout(info_panel)
        info.setContentsMargins(14, 10, 14, 10)
        info.setSpacing(12)

        sku_title = QLabel("Active PLC SKU:")
        sku_title.setStyleSheet("font: 800 12px 'Segoe UI'; color:#222; border:none;")
        info.addWidget(sku_title)

        self.active_sku_value_lbl = QLabel("UNKNOWN")
        self.active_sku_value_lbl.setStyleSheet("""
            QLabel {
                background:#f1f3f5;
                color:#111;
                border-radius:8px;
                padding:6px 12px;
                font: 900 12px 'Segoe UI';
            }
        """)
        info.addWidget(self.active_sku_value_lbl)

        self.active_sku_lbl = QLabel("Recipe SKU: UNKNOWN")
        self.active_sku_lbl.setStyleSheet("font: 800 12px 'Segoe UI'; color:#333; border:none;")
        info.addWidget(self.active_sku_lbl)

        self.recipe_status_lbl = QLabel("Recipe: UNKNOWN")
        self.recipe_status_lbl.setStyleSheet("font: 800 12px 'Segoe UI'; color:#333; border:none;")
        info.addWidget(self.recipe_status_lbl)

        self.overall_status_lbl = QLabel("Overall: UNKNOWN")
        self.overall_status_lbl.setAlignment(Qt.AlignCenter)
        self.overall_status_lbl.setStyleSheet("""
            QLabel {
                background:#eeeeee;
                color:#333;
                border-radius:10px;
                padding:6px 12px;
                font: 900 12px 'Segoe UI';
            }
        """)
        info.addWidget(self.overall_status_lbl)

        info.addStretch()

        root.addWidget(info_panel)

        # Table panel
        table_panel = QFrame()
        table_panel.setStyleSheet("""
            QFrame {
                background:white;
                border-radius:14px;
                border:1px solid #ececec;
            }
        """)
        table_layout = QVBoxLayout(table_panel)
        table_layout.setContentsMargins(10, 10, 10, 10)

        self.table = QTableWidget()
        self.table.setColumnCount(8)
        self.table.setHorizontalHeaderLabels([
            "Axis",
            "Current Pos",
            "Recipe Pos",
            "Tolerance",
            "Enabled",
            "Homed",
            "Fault",
            "Status",
        ])
        self.table.setRowCount(12)
        self.table.setAlternatingRowColors(True)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setSelectionMode(QTableWidget.SingleSelection)
        self.table.setShowGrid(True)

        self.table.setStyleSheet("""
            QTableWidget {
                background:white;
                gridline-color:#dddddd;
                font: 700 11px 'Segoe UI';
                alternate-background-color:#fafafa;
                selection-background-color:#eee6f7;
                selection-color:#111;
            }
            QHeaderView::section {
                background:#571c86;
                color:white;
                font: 900 11px 'Segoe UI';
                padding:7px;
                border:none;
                border-right:1px solid #6f2aa1;
            }
        """)

        header_view = self.table.horizontalHeader()
        header_view.setSectionResizeMode(QHeaderView.Fixed)
        header_view.setStretchLastSection(False)

        self.table.verticalHeader().setDefaultSectionSize(30)
        self.table.setMinimumHeight(380)

        table_layout.addWidget(self.table)
        root.addWidget(table_panel, 1)

        # Buttons
        btn_row = QHBoxLayout()
        btn_row.setSpacing(10)

        def mkbtn(text, bg, hover, fn):
            b = QPushButton(text)
            b.setFixedHeight(40)
            b.setCursor(Qt.PointingHandCursor)
            b.setStyleSheet(f"""
                QPushButton {{
                    background:{bg};
                    color:white;
                    border:none;
                    border-radius:10px;
                    font: 800 12px 'Segoe UI';
                    padding: 0 16px;
                }}
                QPushButton:hover {{
                    background:{hover};
                }}
            """)
            b.clicked.connect(fn)
            return b

        btn_row.addWidget(
            mkbtn(
                "Refresh Now",
                "#7C19EE",
                "#873DDD",
                self.refresh_axis_status
            )
        )

        btn_row.addStretch()

        btn_row.addWidget(
            mkbtn(
                "Close",
                "#130F0F",
                "#555555",
                self.close_page
            )
        )

        root.addLayout(btn_row)

        self.status_msg_lbl = QLabel("Status: Waiting...")
        self.status_msg_lbl.setStyleSheet("font: 800 11px 'Segoe UI'; color:#444;")
        root.addWidget(self.status_msg_lbl)

        QTimer.singleShot(0, self._resize_table_columns)

    def _resize_table_columns(self):
        """
        Keep the table full-width and balanced.
        This avoids the large blank area on the right.
        """
        try:
            total_width = self.table.viewport().width()

            if total_width <= 100:
                return

            # Small safety margin for grid/scrollbar
            total_width = total_width - 6

            ratios = [
                0.22,  # Axis
                0.13,  # Current Pos
                0.13,  # Recipe Pos
                0.10,  # Tolerance
                0.10,  # Enabled
                0.10,  # Homed
                0.09,  # Fault
                0.13,  # Status
            ]

            used = 0
            for col, ratio in enumerate(ratios):
                if col == len(ratios) - 1:
                    width = max(90, total_width - used)
                else:
                    width = max(70, int(total_width * ratio))
                    used += width

                self.table.setColumnWidth(col, width)

        except Exception:
            pass



    # ------------------------------------------------------------
    # REFRESH
    # ------------------------------------------------------------
    def start_refresh(self):
        interval = max(500, int(self.service.refresh_ms))
        self.refresh_timer.start(interval)
        self.refresh_state_lbl.setText(f"Auto Refresh: ON ({interval} ms)")
        self.refresh_axis_status()

    def stop_refresh(self):
        self.refresh_timer.stop()
        self.refresh_state_lbl.setText("Auto Refresh: OFF")


    def refresh_axis_status(self):
        try:
            result = self.service.get_axis_status()
            self._apply_result(result)

        except Exception as e:
            QMessageBox.warning(
                self,
                "Axis Status Error",
                f"Failed to refresh Axis Status:\n{e}"
            )

    # ------------------------------------------------------------
    # APPLY RESULT
    # ------------------------------------------------------------
    def _fmt(self, value):
        if value is None:
            return "UNKNOWN"
        if isinstance(value, bool):
            return "YES" if value else "NO"
        return str(value)

    def _set_item(self, row, col, text, status=None):
        item = QTableWidgetItem(str(text))
        item.setTextAlignment(Qt.AlignCenter)

        if status in ("OK", "LIVE ONLY"):
            item.setForeground(Qt.darkGreen)

        elif status in ("FAULT", "NOT HOMED", "OUT OF RANGE"):
            item.setForeground(Qt.red)

        elif status in ("DISABLED", "UNKNOWN"):
            item.setForeground(Qt.darkYellow)

        self.table.setItem(row, col, item)

    def _apply_result(self, result):
        active_sku = result.get("active_sku", "UNKNOWN")
        sku_message = result.get("sku_message", "-")
        recipe_status = result.get("recipe_status", "UNKNOWN")
        overall_ok = bool(result.get("overall_ok", False))
        axes = result.get("axes", [])

        sku_info = result.get("sku_info", {})
        plc_raw_value = sku_info.get("plc_raw_value", None)

        self.active_sku_value_lbl.setText(str(active_sku))

        if plc_raw_value not in (None, "", "UNKNOWN"):
            self.active_sku_lbl.setText(
                f"PLC Active SKU: {active_sku} | Raw PLC Value: {plc_raw_value}"
            )
        else:
            self.active_sku_lbl.setText(
                f"PLC Active SKU: {active_sku}"
            )

        self.recipe_status_lbl.setText(f"MongoDB Recipe: {recipe_status}")
        self.status_msg_lbl.setText(f"Status: {sku_message}")

        if overall_ok:
            self.overall_status_lbl.setText("Overall: OK")
            self.overall_status_lbl.setStyleSheet("""
                QLabel {
                    background:#2f9e44;
                    color:white;
                    border-radius:10px;
                    padding:6px 12px;
                    font: 900 12px 'Segoe UI';
                }
            """)
        else:
            self.overall_status_lbl.setText("Overall: CHECK REQUIRED")
            self.overall_status_lbl.setStyleSheet("""
                QLabel {
                    background:#e03131;
                    color:white;
                    border-radius:10px;
                    padding:6px 12px;
                    font: 900 12px 'Segoe UI';
                }
            """)

        self.table.setRowCount(max(12, len(axes)))

        for row, axis in enumerate(axes):
            status = axis.get("status", "UNKNOWN")

            self._set_item(row, 0, axis.get("name", f"Axis {row + 1}"), status)
            self._set_item(row, 1, self._fmt(axis.get("current_position")), status)
            self._set_item(row, 2, self._fmt(axis.get("recipe_position")), status)
            self._set_item(row, 3, self._fmt(axis.get("tolerance")), status)
            self._set_item(row, 4, self._fmt(axis.get("enabled")), status)
            self._set_item(row, 5, self._fmt(axis.get("homed")), status)
            self._set_item(row, 6, self._fmt(axis.get("fault")), status)
            self._set_item(row, 7, status, status)

        self.status_msg_lbl.setText(f"Status: {sku_message}")
        self._resize_table_columns()

    # ------------------------------------------------------------
    # CLOSE / EVENTS
    # ------------------------------------------------------------
    def close_page(self):
        self.stop_refresh()

        if callable(self.on_close):
            self.on_close()

    def showEvent(self, event):
        super().showEvent(event)

        self.start_refresh()
        QTimer.singleShot(100, self._resize_table_columns)

    def hideEvent(self, event):
        self.stop_refresh()
        super().hideEvent(event)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._resize_table_columns()