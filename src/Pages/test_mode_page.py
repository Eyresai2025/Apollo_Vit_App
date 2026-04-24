# test_mode_page.py
import os, random
from datetime import datetime
from PyQt5.QtCore import Qt, QTimer # type: ignore
from PyQt5.QtWidgets import ( # type: ignore
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QFrame, QLabel,
    QPushButton, QProgressBar, QMessageBox
)
from PyQt5.QtGui import QPixmap,QIcon # type: ignore

DEFAULT_EXPECTED_SERIALS = [
    "244802149","244802163","251102086","251401655","251300826","251500640"
]

def _card():
    fr = QFrame()
    fr.setStyleSheet("""
        QFrame { background: white; border-radius: 14px; border: 1px solid #ececec; }
    """)
    return fr

def _set(dot: QLabel, txt: QLabel, state: str, msg: str):
    colors = {"ok":"#4CAF50","warn":"#ff9800","err":"#f44336","off":"#666666"}
    c = colors.get(state, "#666666")
    dot.setStyleSheet(f"QLabel {{ font: 700 16px 'Segoe UI'; color: {c}; }}")
    txt.setStyleSheet(f"QLabel {{ font: 600 12px 'Segoe UI'; color: {c}; }}")
    txt.setText(msg)

class TestModePage(QWidget):
    """
    This is a QWidget page (NOT dialog) → meant to be inserted into QStackedWidget.
    """
    def __init__(self, reports_dir, expected_serials=None, on_close=None,media_path=None, parent=None):
        super().__init__(parent)
        self.reports_dir = reports_dir
        self.expected_serials = expected_serials or DEFAULT_EXPECTED_SERIALS
        self.on_close = on_close  # callback to go back dashboard
        self.media_path = media_path 
        self.is_polling = True
        self.dummy_m99_state = True
        self.poll_count = 0

        self.found = []
        self.missing = []

        self._build_ui()

        # timers
        self.poll_timer = QTimer(self)
        self.poll_timer.timeout.connect(self._poll)
        self.poll_timer.start(1000)

        QTimer.singleShot(400, self.scan_cameras)

    def _build_ui(self):
        self.setStyleSheet("QWidget { background-color: #f5f5f5; }")
        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(10)

        # header inside page (optional)
        top = QFrame()
        top.setStyleSheet("QFrame { background:#571c86; border-radius:12px; }")
        tl = QHBoxLayout(top)
        tl.setContentsMargins(16, 12, 16, 12)

        title = QLabel(" System Test Monitor (Simulation Mode)")
        title.setStyleSheet("font: 900 14px 'Segoe UI'; color:white;")
        tl.addWidget(title)

        tl.addStretch()

        badge = QLabel("● DUMMY MODE")
        badge.setStyleSheet("font: 900 11px 'Segoe UI'; color:#ff9800;")
        tl.addWidget(badge)

        root.addWidget(top)

        # cards grid
        grid_wrap = _card()
        grid = QGridLayout(grid_wrap)
        grid.setContentsMargins(14, 14, 14, 14)
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(12)

        def status_card(name, icon_file):
            fr = _card()
            v = QVBoxLayout(fr)
            v.setContentsMargins(14, 12, 14, 12)

            row = QHBoxLayout()

            # ✅ icon from media/img
            icon_label = QLabel()
            icon_label.setFixedSize(28, 28)
            icon_label.setAlignment(Qt.AlignCenter)
            icon_label.setStyleSheet("""
                QLabel {
                    background: #f5f5f5;
                    border: 1px solid #e6e6e6;
                    border-radius: 10px;
                }
            """)

            icon_path = ""
            if self.media_path:
                icon_path = os.path.join(self.media_path, "img", icon_file)

            if icon_path and os.path.exists(icon_path):
                pm = QPixmap(icon_path).scaled(26, 26, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                icon_label.setPixmap(pm)
            else:
                icon_label.setText("🖼️")  # fallback

            nm = QLabel(name)
            nm.setStyleSheet("font: 900 12px 'Segoe UI'; color:#222;")

            dot = QLabel("●")
            dot.setStyleSheet("QLabel { font:700 16px 'Segoe UI'; color:#666; }")

            row.addWidget(icon_label)
            row.addSpacing(10)
            row.addWidget(nm)
            row.addStretch()
            row.addWidget(dot)
            v.addLayout(row)

            st = QLabel("Initializing...")
            st.setStyleSheet("font: 600 12px 'Segoe UI'; color:#888;")
            v.addWidget(st)

            return fr, dot, st

        w1, self.lights_dot, self.lights_txt = status_card("Lighting System", "lightbulb.png")
        w2, self.conv_dot,  self.conv_txt    = status_card("Conveyor Belt", "production.png")
        w3, self.ar_dot,    self.ar_txt      = status_card("Accept & Reject", "failure.png")
        w4, self.cam_dot,   self.cam_txt     = status_card("Camera Array", "camera.png")
        w5, self.m99_dot,   self.m99_txt     = status_card("Master Control (M99)", "plc.png")


        grid.addWidget(w1, 0, 0)
        grid.addWidget(w2, 0, 1)
        grid.addWidget(w3, 1, 0)
        grid.addWidget(w4, 1, 1)
        grid.addWidget(w5, 2, 0, 1, 2)

        root.addWidget(grid_wrap, 1)

        # buttons
        btn_row = QHBoxLayout()

        def mkbtn(text, bg, hover, fn):
            b = QPushButton(text)
            b.setFixedHeight(40)
            b.setCursor(Qt.PointingHandCursor)
            b.setStyleSheet(f"""
                QPushButton {{
                    background:{bg}; color:white; border:none; border-radius:10px;
                    font: 600 13px 'Segoe UI'; padding: 0 14px;
                }}
                QPushButton:hover {{ background:{hover}; }}
            """)
            b.clicked.connect(fn)
            return b

        btn_row.addWidget(mkbtn(" Refresh Cameras", "#7C19EE", "#873DDD", self.scan_cameras))
        btn_row.addWidget(mkbtn("Emergency Stop", "#7C19EE", "#873DDD", self.emergency_stop))
        btn_row.addWidget(mkbtn("Generate Report", "#7C19EE", "#873DDD", self.generate_report))
        btn_row.addStretch()
        btn_row.addWidget(mkbtn("Close", "#130F0F", "#555555", self.close_and_reset))

        root.addLayout(btn_row)

        # progress
        pwrap = QFrame()
        pwrap.setStyleSheet("QFrame { background:#ffffff; border-radius:12px; border:1px solid #ececec; }")
        pl = QHBoxLayout(pwrap)
        pl.setContentsMargins(14, 10, 14, 10)

        self.p_label = QLabel("System Status:")
        self.p_label.setStyleSheet("font: 800 11px 'Segoe UI'; color:#333;")
        pl.addWidget(self.p_label)

        self.pbar = QProgressBar()
        self.pbar.setRange(0, 100)
        self.pbar.setValue(80)
        self.pbar.setTextVisible(False)
        self.pbar.setFixedHeight(10)
        self.pbar.setStyleSheet("""
            QProgressBar { background:#eee; border-radius:5px; }
            QProgressBar::chunk { background:#4CAF50; border-radius:5px; }
        """)
        pl.addWidget(self.pbar, 1)
        root.addWidget(pwrap)

        # initial state
        _set(self.m99_dot, self.m99_txt, "warn", "INITIALIZING...")
        _set(self.lights_dot, self.lights_txt, "off", "STANDBY")
        _set(self.conv_dot, self.conv_txt, "off", "STANDBY")
        _set(self.ar_dot, self.ar_txt, "off", "STANDBY")
        _set(self.cam_dot, self.cam_txt, "off", "WAITING...")

    # -------- logic --------
    def scan_cameras(self):
        _set(self.cam_dot, self.cam_txt, "warn", "Scanning devices...")
        total = len(self.expected_serials)
        found_count = random.randint(max(1, total - 2), total)
        self.found = random.sample(self.expected_serials, k=found_count)
        self.missing = sorted(list(set(self.expected_serials) - set(self.found)))

        if found_count >= total:
            _set(self.cam_dot, self.cam_txt, "ok", f"ALL CONNECTED ({found_count}/{total})")
        else:
            preview = ", ".join(self.missing[:3])
            more = f" (+{len(self.missing)-3} more)" if len(self.missing) > 3 else ""
            _set(self.cam_dot, self.cam_txt, "err", f"MISSING ({found_count}/{total}) • {preview}{more}")

    def emergency_stop(self):
        self.dummy_m99_state = False
        _set(self.m99_dot, self.m99_txt, "warn", "EMERGENCY STOPPED")

    def _poll(self):
        if not self.is_polling:
            return
        self.poll_count += 1

        if self.dummy_m99_state:
            _set(self.m99_dot, self.m99_txt, "ok", f"ACTIVE (Poll #{self.poll_count})")
            _set(self.lights_dot, self.lights_txt, "ok", "ILLUMINATED")
            _set(self.conv_dot, self.conv_txt, "ok", "RUNNING")
            _set(self.ar_dot, self.ar_txt, "ok", "READY")
            self.pbar.setValue(100)
            self.p_label.setText("System Status: SIMULATION MODE")
        else:
            _set(self.m99_dot, self.m99_txt, "warn", "INACTIVE")
            _set(self.lights_dot, self.lights_txt, "off", "OFF")
            _set(self.conv_dot, self.conv_txt, "off", "STOPPED")
            _set(self.ar_dot, self.ar_txt, "off", "DISABLED")
            self.pbar.setValue(60)
            self.p_label.setText("System Status: STOPPED")

    def generate_report(self):
        os.makedirs(self.reports_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fp = os.path.join(self.reports_dir, f"Test_Report_{ts}.txt")

        cam_status = "PASS" if len(self.found) >= len(self.expected_serials) else "FAIL"
        content = f"""SYSTEM TEST REPORT
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

Camera System: {cam_status}
Expected: {len(self.expected_serials)}
Found: {len(self.found)}
Missing: {len(self.missing)}

Expected Serials:
{chr(10).join(self.expected_serials)}

Found Serials:
{chr(10).join(self.found) if self.found else "None"}
"""
        try:
            with open(fp, "w", encoding="utf-8") as f:
                f.write(content)
            QMessageBox.information(self, "Report Saved", f"Saved:\n{fp}")
        except Exception as e:
            QMessageBox.critical(self, "Save Error", str(e))

    def close_and_reset(self):
        self.is_polling = False
        self.poll_timer.stop()

        if QMessageBox.question(self, "Close", "Close Test Mode and return to Dashboard?",
                                QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes) == QMessageBox.Yes:
            if callable(self.on_close):
                self.on_close()
