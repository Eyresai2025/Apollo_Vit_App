# src/Pages/test_mode_page.py

import os
from datetime import datetime

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QFrame, QLabel,
    QPushButton, QProgressBar, QMessageBox, QScrollArea, QSizePolicy,
    QCheckBox
)
from PyQt5.QtGui import QPixmap

from src.COMMON.full_hardware_check import start_full_hardware_check_from_test_page


def _card():
    fr = QFrame()
    fr.setStyleSheet("""
        QFrame {
            background: white;
            border-radius: 14px;
            border: 1px solid #ececec;
        }
    """)
    return fr


def _set(dot: QLabel, txt: QLabel, state: str, msg: str):
    colors = {
        "ok": "#2f9e44",
        "warn": "#ff9800",
        "err": "#e03131",
        "off": "#666666",
    }

    c = colors.get(state, "#666666")

    dot.setStyleSheet(f"QLabel {{ font: 900 16px 'Segoe UI'; color: {c}; }}")
    txt.setStyleSheet(f"""
        QLabel {{
            font: 700 11px 'Segoe UI';
            color: {c};
            background: transparent;
            border: none;
        }}
    """)
    txt.setText(msg)


class TestModePage(QWidget):
    def __init__(self, reports_dir, expected_serials=None, on_close=None, media_path=None, parent=None):
        super().__init__(parent)

        self.reports_dir = reports_dir
        self.expected_serials = expected_serials or []
        self.on_close = on_close
        self.media_path = media_path

        self.last_hardware_check_result = None
        self._hardware_check_thread = None
        self._hardware_check_worker = None
        self.poll_timer = None

        self.light_checks = {}

        self._build_ui()

    def _build_ui(self):
        self.setStyleSheet("QWidget { background-color: #f5f5f5; }")

        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(10)

        top = QFrame()
        top.setStyleSheet("QFrame { background:#571c86; border-radius:12px; }")
        tl = QHBoxLayout(top)
        tl.setContentsMargins(16, 10, 16, 10)

        title = QLabel("System Test Monitor")
        title.setStyleSheet("font: 900 14px 'Segoe UI'; color:white; border:none;")
        tl.addWidget(title)

        tl.addStretch()

        badge = QLabel("● HARDWARE CHECK")
        badge.setStyleSheet("font: 900 11px 'Segoe UI'; color:#ffcc00; border:none;")
        tl.addWidget(badge)

        root.addWidget(top)

        page_scroll = QScrollArea()
        page_scroll.setWidgetResizable(True)
        page_scroll.setStyleSheet("""
            QScrollArea {
                border: none;
                background: #f5f5f5;
            }
            QScrollBar:vertical {
                border: none;
                background: #f1f1f1;
                width: 10px;
                border-radius: 5px;
            }
            QScrollBar::handle:vertical {
                background: #c9c9c9;
                border-radius: 5px;
                min-height: 30px;
            }
        """)

        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)
        scroll_layout.setContentsMargins(0, 0, 0, 0)
        scroll_layout.setSpacing(10)

        grid_wrap = _card()
        grid = QGridLayout(grid_wrap)
        grid.setContentsMargins(14, 14, 14, 14)
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(12)

        def status_card(name, icon_file, detail_height=150, with_light_checkboxes=False):
            fr = _card()
            fr.setMinimumHeight(detail_height + 85)
            fr.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

            v = QVBoxLayout(fr)
            v.setContentsMargins(14, 12, 14, 12)
            v.setSpacing(8)

            row = QHBoxLayout()
            row.setSpacing(10)

            icon_label = QLabel()
            icon_label.setFixedSize(40, 40)
            icon_label.setAlignment(Qt.AlignCenter)
            icon_label.setStyleSheet("""
                QLabel {
                    background: #f7f7f7;
                    border: 1px solid #e6e6e6;
                    border-radius: 10px;
                }
            """)

            icon_path = ""
            if self.media_path:
                icon_path = os.path.join(self.media_path, "img", icon_file)

            if icon_path and os.path.exists(icon_path):
                pm = QPixmap(icon_path).scaled(
                    28, 28,
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation
                )
                icon_label.setPixmap(pm)
            else:
                icon_label.setText("🖥️")

            name_label = QLabel(name)
            name_label.setStyleSheet("""
                QLabel {
                    font: 900 12px 'Segoe UI';
                    color:#222;
                    border:none;
                    background: transparent;
                }
            """)

            dot = QLabel("●")
            dot.setStyleSheet("QLabel { font:900 16px 'Segoe UI'; color:#666; border:none; }")

            row.addWidget(icon_label)
            row.addWidget(name_label)
            row.addStretch()
            row.addWidget(dot)

            v.addLayout(row)

            content_widget = QWidget()
            content_layout = QVBoxLayout(content_widget)
            content_layout.setContentsMargins(4, 4, 4, 4)
            content_layout.setSpacing(6)

            if with_light_checkboxes:
                cb_style = """
                    QCheckBox {
                        font: 700 12px 'Segoe UI';
                        color: #333;
                        spacing: 8px;
                    }
                    QCheckBox::indicator {
                        width: 18px;
                        height: 18px;
                    }
                """

                for i in range(1, 6):
                    key = f"light{i}"
                    cb = QCheckBox(f"Light {i} working")
                    cb.setStyleSheet(cb_style)
                    self.light_checks[key] = cb
                    content_layout.addWidget(cb)

            detail_label = QLabel("Waiting...")
            detail_label.setWordWrap(True)
            detail_label.setAlignment(Qt.AlignLeft | Qt.AlignTop)
            detail_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
            detail_label.setStyleSheet("""
                QLabel {
                    font: 700 11px 'Segoe UI';
                    color:#666;
                    background: white;
                    border:none;
                    padding: 4px;
                }
            """)

            content_layout.addWidget(detail_label)
            content_layout.addStretch()

            detail_scroll = QScrollArea()
            detail_scroll.setWidgetResizable(True)
            detail_scroll.setMinimumHeight(detail_height)
            detail_scroll.setStyleSheet("""
                QScrollArea {
                    background: white;
                    border: 1px solid #eeeeee;
                    border-radius: 10px;
                }
                QScrollBar:vertical {
                    border: none;
                    background: #f0f0f0;
                    width: 8px;
                    border-radius: 4px;
                }
                QScrollBar::handle:vertical {
                    background: #c7c7c7;
                    border-radius: 4px;
                    min-height: 25px;
                }
            """)
            detail_scroll.setWidget(content_widget)

            v.addWidget(detail_scroll)

            return fr, dot, detail_label

        w1, self.lights_dot, self.lights_txt = status_card(
            "Lighting System",
            "lightbulb.png",
            detail_height=170,
            with_light_checkboxes=True,
        )

        w2, self.laser_dot, self.laser_txt = status_card(
            "Lasers",
            "production.png",
            detail_height=120
        )

        w3, self.cam_dot, self.cam_txt = status_card(
            "Camera Array",
            "camera.png",
            detail_height=180
        )

        w4, self.m99_dot, self.m99_txt = status_card(
            "PLC",
            "plc.png",
            detail_height=180
        )

        grid.addWidget(w1, 0, 0)
        grid.addWidget(w2, 0, 1)
        grid.addWidget(w3, 1, 0)
        grid.addWidget(w4, 1, 1)

        scroll_layout.addWidget(grid_wrap)

        page_scroll.setWidget(scroll_widget)
        root.addWidget(page_scroll, 1)

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
                    font: 700 12px 'Segoe UI';
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
                "Run Full Hardware Check",
                "#7C19EE",
                "#873DDD",
                self.run_full_hardware_check
            )
        )

        btn_row.addWidget(
            mkbtn(
                "Emergency Stop",
                "#7C19EE",
                "#873DDD",
                self.emergency_stop
            )
        )

        btn_row.addWidget(
            mkbtn(
                "Generate Report",
                "#7C19EE",
                "#873DDD",
                self.generate_report
            )
        )

        btn_row.addStretch()

        btn_row.addWidget(
            mkbtn(
                "Close",
                "#130F0F",
                "#555555",
                self.close_and_reset
            )
        )

        root.addLayout(btn_row)

        pwrap = QFrame()
        pwrap.setStyleSheet("""
            QFrame {
                background:#ffffff;
                border-radius:12px;
                border:1px solid #ececec;
            }
        """)

        pl = QHBoxLayout(pwrap)
        pl.setContentsMargins(14, 8, 14, 8)

        self.p_label = QLabel("System Status: WAITING FOR HARDWARE CHECK")
        self.p_label.setStyleSheet("""
            QLabel {
                font: 900 11px 'Segoe UI';
                color:#333;
                border:none;
                background: transparent;
            }
        """)
        pl.addWidget(self.p_label)

        self.pbar = QProgressBar()
        self.pbar.setRange(0, 100)
        self.pbar.setValue(0)
        self.pbar.setTextVisible(False)
        self.pbar.setFixedHeight(10)
        self.pbar.setStyleSheet("""
            QProgressBar {
                background:#eee;
                border-radius:5px;
                border:none;
            }
            QProgressBar::chunk {
                background:#666;
                border-radius:5px;
            }
        """)
        pl.addWidget(self.pbar, 1)

        root.addWidget(pwrap)

        _set(self.m99_dot, self.m99_txt, "off", "PLC check not started.")
        _set(self.lights_dot, self.lights_txt, "off", "Select Light 1 to Light 5 if working.")
        _set(self.laser_dot, self.laser_txt, "off", "Laser check not started.")
        _set(self.cam_dot, self.cam_txt, "off", "Camera check not started.")

    def get_light_feedback(self):
        return {
            key: cb.isChecked()
            for key, cb in self.light_checks.items()
        }

    def run_full_hardware_check(self):
        start_full_hardware_check_from_test_page(
            test_page=self,
            media_path=self.media_path,
        )

    def emergency_stop(self):
        _set(self.m99_dot, self.m99_txt, "warn", "Emergency stop requested from Test Mode page.")
        self.pbar.setValue(100)
        self.pbar.setStyleSheet("""
            QProgressBar {
                background:#eee;
                border-radius:5px;
                border:none;
            }
            QProgressBar::chunk {
                background:#ff9800;
                border-radius:5px;
            }
        """)
        self.p_label.setText("System Status: EMERGENCY STOP REQUESTED")

        QMessageBox.warning(
            self,
            "Emergency Stop",
            "Emergency stop clicked.\n\nConnect this button to actual PLC emergency stop/reset logic if required."
        )

    def generate_report(self):
        os.makedirs(self.reports_dir, exist_ok=True)

        result = getattr(self, "last_hardware_check_result", None)

        if not result:
            QMessageBox.warning(
                self,
                "No Hardware Check",
                "Please run Full Hardware Check before generating the report."
            )
            return

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fp = os.path.join(self.reports_dir, f"Hardware_Check_Report_{ts}.txt")

        details = result.get("details", {})
        messages = result.get("messages", [])

        lights = details.get("lights", {})
        plc = details.get("plc", {})
        camera = details.get("camera", {})
        laser = details.get("laser", {})
        app_bit = details.get("application_ok_bit", {})

        camera_lines = []
        for cam in camera.get("camera_status", []):
            camera_lines.append(
                f"{cam.get('side')} | Serial: {cam.get('serial')} | Connected: {cam.get('connected')} | {cam.get('message')}"
            )

        content = f"""FULL HARDWARE CHECK REPORT
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

OVERALL STATUS: {"PASS" if result.get("overall_ok") else "FAIL"}
Deployment Mode: {result.get("deployment", "-")}
Check Time: {result.get("timestamp", "-")}

LIGHT USER FEEDBACK:
{lights}

PLC:
PLC Type: {plc.get("plc_type", "-")}
PLC IP: {plc.get("ip", "-")}
Connected: {plc.get("connected", "-")}
Last Error: {plc.get("last_error", "-")}

APPLICATION OK BIT:
Address: {app_bit.get("address", "-")}
Sent: {app_bit.get("sent", "-")}
Value Written: {app_bit.get("value_written", "-")}
Read Back Value: {app_bit.get("read_back_value", "-")}
Verified: {app_bit.get("verified", "-")}
Message: {app_bit.get("message", "-")}

LASER:
Connected: {laser.get("connected", "-")}
Message: {laser.get("message", "-")}

CAMERAS:
{chr(10).join(camera_lines) if camera_lines else "-"}

MESSAGES:
{chr(10).join(messages) if messages else "-"}
"""

        try:
            with open(fp, "w", encoding="utf-8") as f:
                f.write(content)

            QMessageBox.information(
                self,
                "Report Saved",
                f"Hardware check report saved:\n{fp}"
            )

        except Exception as e:
            QMessageBox.critical(self, "Save Error", str(e))

    def close_and_reset(self):
        existing_thread = getattr(self, "_hardware_check_thread", None)

        if existing_thread is not None and existing_thread.isRunning():
            QMessageBox.warning(
                self,
                "Hardware Check Running",
                "Hardware check is still running. Please wait until it completes."
            )
            return

        if QMessageBox.question(
            self,
            "Close",
            "Close Test Mode and return to Live page?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes
        ) == QMessageBox.Yes:
            if callable(self.on_close):
                self.on_close()