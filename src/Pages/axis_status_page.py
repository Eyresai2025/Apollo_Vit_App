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
    - Reads MongoDB Active Recipe / last_loaded_recipe.
    - Does NOT read DB75.DBW288.
    - Shows DB74 live actual position.
    - Shows DB75 running servo recipe value.
    - Shows MongoDB saved recipe target value.
    - Shows delta comparisons.
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

        recipe_no_title = QLabel("Active Recipe:")
        recipe_no_title.setStyleSheet("font: 800 12px 'Segoe UI'; color:#222; border:none;")
        info.addWidget(recipe_no_title)

        self.loaded_recipe_no_lbl = QLabel("UNKNOWN")
        self.loaded_recipe_no_lbl.setStyleSheet("""
            QLabel {
                background:#f1f3f5;
                color:#111;
                border-radius:8px;
                padding:6px 12px;
                font: 900 12px 'Segoe UI';
            }
        """)
        info.addWidget(self.loaded_recipe_no_lbl)

        self.active_sku_lbl = QLabel("SKU: UNKNOWN")
        self.active_sku_lbl.setStyleSheet("font: 800 12px 'Segoe UI'; color:#333; border:none;")
        info.addWidget(self.active_sku_lbl)

        self.recipe_status_lbl = QLabel("MongoDB State: UNKNOWN")
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
        self.table.setColumnCount(10)
        self.table.setHorizontalHeaderLabels([
            "Target",
            "Axis",
            "Position",
            "Current Live Value",
            "Active Recipe Value",
            "MongoDB Value",
            "Enabled",
            "Homed",
            "Fault",
            "Status",
        ])
        self.table.setRowCount(37)
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
                font: 700 10px 'Segoe UI';
                alternate-background-color:#fafafa;
                selection-background-color:#eee6f7;
                selection-color:#111;
            }
            QHeaderView::section {
                background:#571c86;
                color:white;
                font: 900 10px 'Segoe UI';
                padding:7px;
                border:none;
                border-right:1px solid #6f2aa1;
            }
        """)

        header_view = self.table.horizontalHeader()
        header_view.setSectionResizeMode(QHeaderView.Fixed)
        header_view.setStretchLastSection(False)

        self.table.verticalHeader().setDefaultSectionSize(30)
        self.table.setMinimumHeight(420)

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
                self.refresh_axis_status,
            )
        )

        btn_row.addStretch()

        btn_row.addWidget(
            mkbtn(
                "Close",
                "#130F0F",
                "#555555",
                self.close_page,
            )
        )

        root.addLayout(btn_row)

        self.status_msg_lbl = QLabel("Status: Waiting...")
        self.status_msg_lbl.setStyleSheet("font: 800 11px 'Segoe UI'; color:#444;")
        root.addWidget(self.status_msg_lbl)

        QTimer.singleShot(0, self._resize_table_columns)

    def _resize_table_columns(self):
        try:
            total_width = self.table.viewport().width()

            if total_width <= 100:
                return

            total_width = total_width - 6

            ratios = [
                0.08,  # Target
                0.18,  # Axis
                0.09,  # Position
                0.12,  # Current Live Value
                0.12,  # Active Recipe Value
                0.12,  # MongoDB Value
                0.07,  # Enabled
                0.07,  # Homed
                0.06,  # Fault
                0.09,  # Status
            ]

            used = 0
            for col, ratio in enumerate(ratios):
                if col == len(ratios) - 1:
                    width = max(110, total_width - used)
                else:
                    width = max(65, int(total_width * ratio))
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
                f"Failed to refresh Axis Status:\n{e}",
            )

    # ------------------------------------------------------------
    # APPLY RESULT
    # ------------------------------------------------------------
    def _fmt(self, value):
        if value is None:
            return "UNKNOWN"

        if isinstance(value, bool):
            return "YES" if value else "NO"

        try:
            return f"{float(value):.3f}"
        except Exception:
            return str(value)

    def _set_item(self, row, col, text, status=None):
        item = QTableWidgetItem(str(text))
        item.setTextAlignment(Qt.AlignCenter)

        if status in ("OK", "LIVE ONLY"):
            item.setForeground(Qt.darkGreen)

        elif status in (
            "FAULT",
            "NOT HOMED",
            "OUT OF RANGE",
            "RUNNING/MONGO MISMATCH",
            "PLC/MONGO MISMATCH",
        ):
            item.setForeground(Qt.red)

        elif status in (
            "DISABLED",
            "UNKNOWN",
            "DB75 UNKNOWN",
            "MONGO MISSING",
            "LIVE UNKNOWN",
        ):
            item.setForeground(Qt.darkYellow)

        self.table.setItem(row, col, item)

    def _apply_result(self, result):
        loaded_recipe_number = result.get(
            "plc_active_recipe_number",
            result.get("loaded_recipe_number", result.get("active_recipe_number", "UNKNOWN"))
        )

        active_sku = result.get(
            "active_sku",
            result.get("loaded_sku", "UNKNOWN")
        )

        recipe_version = result.get(
            "recipe_version",
            result.get("loaded_recipe_version", "-")
        )

        plc_written = result.get("plc_written")
        plc_verified = result.get("plc_verified")

        sku_message = result.get("sku_message", "-")
        recipe_status = result.get("recipe_status", "UNKNOWN")
        overall_ok = bool(result.get("overall_ok", False))
        targets = result.get("targets", [])

        self.loaded_recipe_no_lbl.setText(str(loaded_recipe_number))
        self.active_sku_lbl.setText(
            f"SKU: {active_sku} | Version: {recipe_version} | "
            f"PLC Written: {plc_written} | PLC Verified: {plc_verified}"
        )
        self.recipe_status_lbl.setText(f"MongoDB State: {recipe_status}")
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

        self.table.setRowCount(max(37, len(targets)))

        for row, target in enumerate(targets):
            status = target.get("status", "UNKNOWN")

            group = str(target.get("group", "-")).upper()
            axis_name = target.get("axis_name", f"Axis {target.get('axis_id', '-')}")

            running_db75 = target.get("running_db75", target.get("active_db75"))

            self._set_item(row, 0, group, status)
            self._set_item(row, 1, axis_name, status)
            self._set_item(row, 2, target.get("position", "-"), status)
            self._set_item(row, 3, self._fmt(target.get("live_db74")), status)
            self._set_item(row, 4, self._fmt(running_db75), status)
            self._set_item(row, 5, self._fmt(target.get("mongo_target")), status)
            self._set_item(row, 6, self._fmt(target.get("enabled")), status)
            self._set_item(row, 7, self._fmt(target.get("homed")), status)
            self._set_item(row, 8, self._fmt(target.get("fault")), status)
            self._set_item(row, 9, status, status)

        for row in range(len(targets), self.table.rowCount()):
            for col in range(self.table.columnCount()):
                self._set_item(row, col, "", None)

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