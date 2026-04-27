# repeatability_page.py
import os
import shutil
from datetime import datetime

from PyQt5.QtCore import Qt, QTimer  # type: ignore
from PyQt5.QtGui import QPixmap  # type: ignore
from PyQt5.QtWidgets import (  # type: ignore
    QWidget, QVBoxLayout, QHBoxLayout, QFrame, QLabel,
    QPushButton, QProgressBar, QMessageBox, QSizePolicy
)

from src.COMMON.db import insert_repeatability_log


IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp")


def _ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def _latest_n_images(folder: str, n: int = 4):
    if not folder or (not os.path.exists(folder)):
        return []
    files = [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith(IMAGE_EXTS)
    ]
    files.sort(key=os.path.getmtime, reverse=True)
    return files[:n]


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
    colors = {"ok": "#4CAF50", "warn": "#ff9800", "err": "#f44336", "off": "#666666"}
    c = colors.get(state, "#666666")
    dot.setStyleSheet(f"QLabel {{ font: 900 16px 'Segoe UI'; color: {c}; }}")
    txt.setStyleSheet(f"QLabel {{ font: 800 11px 'Segoe UI'; color: {c}; }}")
    txt.setText(msg)


class RepeatabilityPage(QWidget):
    """
    Repeatability page (UI-only + dummy run):
    - Auto "connect PLC" when page opens.
    - 1x4 image strip.
    - Indicators below images.
    - Start/Stop/Close only.
    - Only ProgressBar for progress.
    """

    def __init__(
        self,
        media_path: str,
        raw_dir: str,
        save_root_dir: str,
        on_close=None,
        parent=None
    ):
        super().__init__(parent)
        self.media_path = media_path
        self.raw_dir = raw_dir
        self.save_root_dir = save_root_dir
        self.on_close = on_close

        self.labels = ["SIDE WALL 1", "SIDE WALL 2", "INNER SIDE", "TOP", "BEAD"]
        self.img_labels = []
        self.output_label = None

        # state (reset each time page is created)
        self.plc_connected = False
        self.running = False
        self.target_cycles = 10
        self.current_cycle = 0

        self._build_ui()

        # auto preview refresh
        self.preview_timer = QTimer(self)
        self.preview_timer.timeout.connect(self.refresh_preview_only)
        self.preview_timer.start(1500)

        # run timer (dummy cycle simulation)
        self.run_timer = QTimer(self)
        self.run_timer.timeout.connect(self._tick_run)

        # AUTO CONNECT PLC on open
        self._auto_connect_plc()
        
    def reset_page(self, refresh_preview: bool = True):
        # stop run timer
        if hasattr(self, "run_timer") and self.run_timer.isActive():
            self.run_timer.stop()

        self.running = False
        self.current_cycle = 0

        # reset progress
        self.pbar.setValue(0)
        self.progress_txt.setText(f"Progress: 0/{self.target_cycles}")

        # reset images to blank
        for lbl in self.img_labels:
            lbl.setPixmap(QPixmap())
            lbl.setText("🖼️")

        # reset buttons
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)

        # reset indicators (THIS IS WHAT YOU ARE MISSING)
        _set(self.model_dot, self.model_txt, "off", "IDLE")
        _set(self.db_dot, self.db_txt, "off", "IDLE")

        # re-auto connect PLC + Camera indicator (so it always shows connected/ready fresh)
        self._auto_connect_plc()

        if refresh_preview:
            self.refresh_preview_only()


    # ---------------- UI ----------------
    def _build_ui(self):
        self.setStyleSheet("QWidget { background-color: #f5f5f5; }")
        root = QVBoxLayout(self)
        root.setContentsMargins(14, 12, 14, 12)
        root.setSpacing(10)

        # Header strip
        head = QFrame()
        head.setStyleSheet("QFrame{ background:#571c86; border-radius:12px; }")
        hl = QHBoxLayout(head)
        hl.setContentsMargins(16, 10, 16, 10)

        title = QLabel("Repeatability")
        title.setStyleSheet("font: 900 13px 'Segoe UI'; color:white;")
        hl.addWidget(title)
        hl.addStretch()

        mode = QLabel("• Simulation Mode")
        mode.setStyleSheet("font: 900 11px 'Segoe UI'; color:#ffcc66;")
        hl.addWidget(mode)

        root.addWidget(head)

        # ======================
        # 1x4 image strip
        # ======================
        images_wrap = _card()
        images_l = QVBoxLayout(images_wrap)
        images_l.setContentsMargins(8, 8, 8, 8)
        images_l.setSpacing(6)

        row = QHBoxLayout()
        row.setSpacing(6)

        for i in range(len(self.labels)):
            card = _card()
            card_l = QVBoxLayout(card)
            card_l.setContentsMargins(6, 6, 6, 6)
            card_l.setSpacing(4)

            t = QLabel(self.labels[i])
            t.setAlignment(Qt.AlignCenter)
            t.setStyleSheet("font: 900 11px 'Segoe UI'; color:#571c86;")
            card_l.addWidget(t)

            img = QLabel("🖼️")
            img.setAlignment(Qt.AlignCenter)
            img.setMinimumHeight(260)
            img.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            img.setStyleSheet("""
                QLabel {
                    background:#ffffff;
                    border: 1px solid #eeeeee;
                    border-radius: 12px;
                    font: 900 28px 'Segoe UI';
                    color:#bbb;
                }
            """)
            card_l.addWidget(img, 1)
            self.img_labels.append(img)

            row.addWidget(card)

        images_l.addLayout(row)
        root.addWidget(images_wrap, 1)

        # ======================
        # Indicators BELOW images
        # ======================
        ind_wrap = _card()
        ind_l = QHBoxLayout(ind_wrap)
        ind_l.setContentsMargins(14, 10, 14, 10)
        ind_l.setSpacing(18)

        def indicator(name):
            box = QFrame()
            bl = QHBoxLayout(box)
            bl.setContentsMargins(0, 0, 0, 0)
            bl.setSpacing(8)

            dot = QLabel("●")
            dot.setStyleSheet("QLabel { font:900 16px 'Segoe UI'; color:#666; }")
            txt = QLabel("IDLE")
            txt.setStyleSheet("QLabel { font:800 11px 'Segoe UI'; color:#666; }")

            nm = QLabel(name)
            nm.setStyleSheet("font: 900 11px 'Segoe UI'; color:#222;")

            bl.addWidget(nm)
            bl.addSpacing(6)
            bl.addWidget(dot)
            bl.addWidget(txt)
            bl.addStretch()
            return box, dot, txt

        plc_box, self.plc_dot, self.plc_txt = indicator("PLC")
        cam_box, self.cam_dot, self.cam_txt = indicator("CAMERA")
        model_box, self.model_dot, self.model_txt = indicator("MODELS")
        db_box, self.db_dot, self.db_txt = indicator("MONGODB")

        ind_l.addWidget(plc_box, 1)
        ind_l.addWidget(cam_box, 1)
        ind_l.addWidget(model_box, 1)
        ind_l.addWidget(db_box, 1)

        root.addWidget(ind_wrap)

        # ======================
        # Buttons + Progress bar
        # ======================
        bottom = QFrame()
        bottom_l = QHBoxLayout(bottom)
        bottom_l.setContentsMargins(0, 0, 0, 0)
        bottom_l.setSpacing(10)

        def mkbtn(text, primary=False):
            b = QPushButton(text)
            b.setCursor(Qt.PointingHandCursor)
            b.setFixedHeight(40)
            if primary:
                b.setStyleSheet("""
                    QPushButton {
                        background:#571c86; color:white; border:none;
                        border-radius:12px; font: 900 12px 'Segoe UI';
                        padding: 0 18px;
                    }
                    QPushButton:hover { background:#6b2aa3; }
                    QPushButton:disabled { background:#b8a6c9; }
                """)
            else:
                b.setStyleSheet("""
                    QPushButton {
                        background: rgba(87,28,134,18);
                        color:#571c86;
                        border: 1px solid rgba(87,28,134,120);
                        border-radius:12px;
                        font: 900 12px 'Segoe UI';
                        padding: 0 18px;
                    }
                    QPushButton:hover { background: rgba(87,28,134,35); }
                    QPushButton:disabled { color:#999; border-color:#ddd; }
                """)
            return b

        self.btn_start = mkbtn("Start", primary=True)
        self.btn_start.clicked.connect(self.start_repeatability)
        bottom_l.addWidget(self.btn_start)

        self.btn_stop = mkbtn("Stop", primary=False)
        self.btn_stop.clicked.connect(self.stop_repeatability)
        self.btn_stop.setEnabled(False)
        bottom_l.addWidget(self.btn_stop)

        bottom_l.addStretch()

        self.btn_close = mkbtn("Close", primary=False)
        self.btn_close.clicked.connect(self.close_page)
        bottom_l.addWidget(self.btn_close)

        root.addWidget(bottom)

        # progress bar + progress text
        pwrap = _card()
        pl = QHBoxLayout(pwrap)
        pl.setContentsMargins(12, 10, 12, 10)
        pl.setSpacing(10)

        self.progress_txt = QLabel("Progress: 0/10")
        self.progress_txt.setStyleSheet("font: 900 11px 'Segoe UI'; color:#333;")
        pl.addWidget(self.progress_txt)

        self.pbar = QProgressBar()
        self.pbar.setRange(0, 100)
        self.pbar.setValue(0)
        self.pbar.setTextVisible(False)
        self.pbar.setFixedHeight(10)
        self.pbar.setStyleSheet("""
            QProgressBar { background:#eee; border-radius:5px; }
            QProgressBar::chunk { background:#4CAF50; border-radius:5px; }
        """)
        pl.addWidget(self.pbar, 1)

        root.addWidget(pwrap)

        self.pbar.setRange(0, 100)
        self.pbar.setValue(0)
        self.pbar.setTextVisible(False)
        self.pbar.setFixedHeight(10)
        self.pbar.setStyleSheet("""
            QProgressBar { background:#eee; border-radius:5px; }
            QProgressBar::chunk { background:#4CAF50; border-radius:5px; }
        """)
        pl.addWidget(self.pbar, 1)
        root.addWidget(pwrap)

        # initial status
        _set(self.plc_dot, self.plc_txt, "off", "DISCONNECTED")
        _set(self.cam_dot, self.cam_txt, "off", "WAITING")
        _set(self.model_dot, self.model_txt, "off", "IDLE")
        _set(self.db_dot, self.db_txt, "off", "IDLE")

        self.refresh_preview_only()

    # ---------------- AUTO CONNECT PLC ----------------
    def _auto_connect_plc(self):
        # dummy connect
        self.plc_connected = True
        _set(self.plc_dot, self.plc_txt, "ok", "CONNECTED")
        _set(self.cam_dot, self.cam_txt, "warn", "READY")
        _set(self.model_dot, self.model_txt, "off", "IDLE")
        _set(self.db_dot, self.db_txt, "off", "IDLE")

        insert_repeatability_log({
            "event": "plc_connected_auto",
            "created_at": datetime.utcnow(),
        })

    # ---------------- actions ----------------
    def start_repeatability(self):
        if not self.plc_connected:
            QMessageBox.warning(self, "PLC", "PLC not connected.")
            return

        reply = QMessageBox.question(
            self,
            "Instruction",
            "Run the Master Tyre for 10 times to check Model Efficiency.\n\nContinue?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        if reply != QMessageBox.Yes:
            return

        # reset progress
        self.running = True
        self.current_cycle = 0
        self.pbar.setValue(0)
        self.progress_txt.setText(f"Progress: 0/{self.target_cycles}")
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)

        _set(self.model_dot, self.model_txt, "warn", "RUNNING...")
        _set(self.db_dot, self.db_txt, "warn", "LOGGING...")

        insert_repeatability_log({
            "event": "repeatability_started",
            "target_cycles": self.target_cycles,
            "created_at": datetime.utcnow(),
        })

        self.run_timer.start(900)

    def stop_repeatability(self):
        self.run_timer.stop()
        self.running = False
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)

        _set(self.model_dot, self.model_txt, "off", "STOPPED")
        _set(self.db_dot, self.db_txt, "off", "IDLE")

        insert_repeatability_log({
            "event": "repeatability_stopped",
            "cycle_no": self.current_cycle,
            "created_at": datetime.utcnow(),
        })

    def _tick_run(self):
        if not self.running:
            return

        self.current_cycle += 1
        capture_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_id = f"rep_{capture_id}_c{self.current_cycle:02d}"
        out_dir = _ensure_dir(os.path.join(self.save_root_dir, "Repeatability", run_id))

        paths = _latest_n_images(self.raw_dir, 4)

        while len(paths) < 4:
            paths.append(None)

        # BEAD = TOP
        paths.append(paths[3] if len(paths) >= 4 else None)

        saved = []
        for p, lab in zip(paths, self.labels):
            if not p or not os.path.exists(p):
                saved.append(None)
                continue
            ext = os.path.splitext(p)[1].lower()
            dst = os.path.join(out_dir, f"{lab.replace(' ', '_').lower()}{ext}")
            try:
                shutil.copy2(p, dst)
            except Exception:
                with open(p, "rb") as rf, open(dst, "wb") as wf:
                    wf.write(rf.read())
            saved.append(dst)

        # update previews
        for i, sp in enumerate(saved):
            lbl = self.img_labels[i]
            if sp and os.path.exists(sp):
                pm = QPixmap(sp)
                lbl.setPixmap(pm.scaled(lbl.width(), lbl.height(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
            else:
                lbl.setPixmap(QPixmap())
                lbl.setText("🖼️")

        # progress bar ONLY
        self.pbar.setValue(int((self.current_cycle / self.target_cycles) * 100))
        self.progress_txt.setText(f"Progress: {self.current_cycle}/{self.target_cycles}")

        # DB log
        insert_repeatability_log({
            "event": "cycle_done",
            "cycle_no": self.current_cycle,
            "run_id": run_id,
            "folder_path": out_dir,
            "images": {
                "sidewall1": saved[0],
                "sidewall2": saved[1],
                "inner": saved[2],
                "top": saved[3],
                "bead": saved[4],
            },
            "created_at": datetime.utcnow(),
        })

        if self.current_cycle >= self.target_cycles:
            self.run_timer.stop()
            self.running = False
            self.btn_stop.setEnabled(False)
            self.btn_start.setEnabled(True)

            _set(self.model_dot, self.model_txt, "ok", "DONE")
            _set(self.db_dot, self.db_txt, "ok", "SAVED")
            self.pbar.setValue(100)

            insert_repeatability_log({
                "event": "repeatability_completed",
                "total_cycles": self.target_cycles,
                "created_at": datetime.utcnow(),
            })

    def refresh_preview_only(self):
        paths = _latest_n_images(self.raw_dir, 4)

        # Ensure 4 base images exist
        while len(paths) < 4:
            paths.append(None)

        # 👉 Add BEAD = same as TOP (index 3)
        if len(paths) >= 4:
            paths.append(paths[3])  # BEAD uses TOP image
        else:
            paths.append(None)

        for i, p in enumerate(paths):
            lbl = self.img_labels[i]
            if p and os.path.exists(p):
                pm = QPixmap(p)
                lbl.setPixmap(pm.scaled(lbl.width(), lbl.height(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
            else:
                lbl.setPixmap(QPixmap())
                lbl.setText("🖼️")

    def close_page(self):
        reply = QMessageBox.question(
            self,
            "Close",
            "Close Repeatability and return to Dashboard?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        if reply != QMessageBox.Yes:
            return

        # FULL RESET so next open is fresh (also resets indicators)
        self.reset_page(refresh_preview=True)

        if callable(self.on_close):
            self.on_close()

