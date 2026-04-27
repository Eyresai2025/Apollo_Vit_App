import os
import re
import cv2
from pathlib import Path
from datetime import datetime

from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal, QEvent  # type: ignore
from PyQt5.QtGui import QPixmap  # type: ignore
from PyQt5.QtWidgets import (  # type: ignore
    QWidget, QVBoxLayout, QHBoxLayout, QFrame, QLabel,
    QPushButton, QProgressBar, QMessageBox, QSizePolicy, QApplication,
    QGridLayout, QScrollArea, QDialog, QStackedWidget
)

from src.COMMON.db import save_new_sku_image
# from src.camera.cam_connections import capture_images_from_all_cameras
from src.training.central_vit_trainer import run_training_for_sku

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

# =========================
# CAMERA / TRAINING CONFIG
# =========================
CAMERA_PIPELINE_MAP = {
    "254701283": "sidewall1",
    "254701301": "sidewall2",
    "254701300": "innerwall",
    "254701292": "Tread",
    "254701293": "bead",
    
}

BASE_SRC_DIR = Path(__file__).resolve().parents[1]   # .../src
TRAINING_DIR = BASE_SRC_DIR / "training"
VIT_TRAINING_ROOT = str(TRAINING_DIR / "VIT_Training")

# Prefer your currently working R model if present, else fall back
_PREFERRED_R_WEIGHTS = [
    TRAINING_DIR / "best (1) 1.pt",
    TRAINING_DIR / "R_Detection_185_70_R14_AMZ4G.pt",
]
YOLO_R_PATH = str(next((p for p in _PREFERRED_R_WEIGHTS if p.exists()), _PREFERRED_R_WEIGHTS[0]))


def _ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def _safe_name(text: str) -> str:
    text = str(text or "").strip()
    if not text:
        return "unknown_sku"
    text = re.sub(r'[<>:"/\\|?*]+', "_", text)
    text = re.sub(r"\s+", "_", text)
    text = text.strip("._")
    return text or "unknown_sku"


def _card() -> QFrame:
    fr = QFrame()
    fr.setStyleSheet("""
        QFrame {
            background: #ffffff;
            border: 1px solid #e8e3ef;
            border-radius: 14px;
        }
    """)
    return fr


class ImageViewerDialog(QDialog):
    def __init__(self, image_path: str, title: str = "Image Viewer", parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(1200, 800)
        self.scale_factor = 1.0
        self._pixmap = QPixmap(image_path)

        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(8)

        toolbar = QHBoxLayout()
        toolbar.setSpacing(8)

        def mkbtn(text):
            b = QPushButton(text)
            b.setFixedHeight(32)
            b.setStyleSheet("""
                QPushButton {
                    background:#571c86;
                    color:white;
                    border:none;
                    border-radius:16px;
                    font: 700 11px 'Segoe UI';
                    padding: 0 16px;
                }
                QPushButton:hover { background:#6b2aa3; }
            """)
            return b

        zoom_in_btn = mkbtn("Zoom In")
        zoom_out_btn = mkbtn("Zoom Out")
        reset_btn = mkbtn("Reset")
        fit_btn = mkbtn("Fit Width")

        zoom_in_btn.clicked.connect(self.zoom_in)
        zoom_out_btn.clicked.connect(self.zoom_out)
        reset_btn.clicked.connect(self.reset_zoom)
        fit_btn.clicked.connect(self.fit_width)

        toolbar.addWidget(zoom_in_btn)
        toolbar.addWidget(zoom_out_btn)
        toolbar.addWidget(reset_btn)
        toolbar.addWidget(fit_btn)
        toolbar.addStretch()

        self.zoom_lbl = QLabel("100%")
        self.zoom_lbl.setStyleSheet("font: 700 11px 'Segoe UI'; color:#333;")
        toolbar.addWidget(self.zoom_lbl)

        root.addLayout(toolbar)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(False)
        self.scroll_area.setStyleSheet("""
            QScrollArea {
                background: #111;
                border-radius: 12px;
                border: 1px solid #ddd;
            }
        """)

        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("background:#111;")
        self.scroll_area.setWidget(self.image_label)

        self.scroll_area.viewport().installEventFilter(self)
        self.image_label.installEventFilter(self)

        root.addWidget(self.scroll_area, 1)
        self.update_image()

    def update_image(self):
        if self._pixmap.isNull():
            return

        w = max(1, int(self._pixmap.width() * self.scale_factor))
        h = max(1, int(self._pixmap.height() * self.scale_factor))

        scaled = self._pixmap.scaled(
            w,
            h,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.image_label.setPixmap(scaled)
        self.image_label.resize(scaled.size())
        self.zoom_lbl.setText(f"{int(self.scale_factor * 100)}%")

    def zoom_in(self):
        self.scale_factor = min(self.scale_factor * 1.1, 8.0)
        self.update_image()

    def zoom_out(self):
        self.scale_factor = max(self.scale_factor * 0.9, 0.1)
        self.update_image()

    def reset_zoom(self):
        self.scale_factor = 1.0
        self.update_image()

    def fit_width(self):
        if self._pixmap.isNull():
            return
        viewport_w = max(1, self.scroll_area.viewport().width() - 20)
        self.scale_factor = viewport_w / self._pixmap.width()
        self.update_image()

    def eventFilter(self, obj, event):
        if event.type() == QEvent.Wheel and (event.modifiers() & Qt.ControlModifier):
            if event.angleDelta().y() > 0:
                self.scale_factor = min(self.scale_factor * 1.1, 8.0)
            else:
                self.scale_factor = max(self.scale_factor * 0.9, 0.1)
            self.update_image()
            return True
        return super().eventFilter(obj, event)


class AspectImageLabel(QLabel):
    def __init__(self, title: str = "", parent=None):
        super().__init__(parent)
        self._pm = None
        self._image_path = ""
        self._title = title
        self.setAlignment(Qt.AlignCenter)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(220, 420)
        self.setCursor(Qt.PointingHandCursor)

    def set_image_path(self, path: str):
        self._image_path = path or ""
        if path and os.path.exists(path):
            self._pm = QPixmap(path)
        else:
            self._pm = None
        self._update_scaled()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_scaled()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self._image_path and os.path.exists(self._image_path):
            dlg = ImageViewerDialog(self._image_path, self._title, self)
            dlg.exec_()
        super().mousePressEvent(event)

    def _update_scaled(self):
        if self._pm is None or self._pm.isNull():
            self.setPixmap(QPixmap())
            self.setText("")
            self.setStyleSheet("""
                QLabel {
                    background: #faf9fc;
                    border: 1px solid #e9e4f1;
                    border-radius: 12px;
                }
            """)
            return

        scaled = self._pm.scaled(
            self.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.setText("")
        self.setPixmap(scaled)
        self.setStyleSheet("""
            QLabel {
                background: #faf9fc;
                border: 1px solid #e9e4f1;
                border-radius: 12px;
            }
        """)


class TrainingWorker(QThread):
    status_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(dict)
    error_signal = pyqtSignal(str)

    def __init__(
        self,
        media_path: str,
        sku_name: str,
        serial_pipeline_map: dict,
        vit_training_root: str,
        yolo_r_path: str,
        device: str = "cuda",
        rebuild_dataset: bool = True,
        parent=None
    ):
        super().__init__(parent)
        self.media_path = media_path
        self.sku_name = sku_name
        self.serial_pipeline_map = serial_pipeline_map
        self.vit_training_root = vit_training_root
        self.yolo_r_path = yolo_r_path
        self.device = device
        self.rebuild_dataset = rebuild_dataset

    def run(self):
        try:
            summary = run_training_for_sku(
                media_path=self.media_path,
                sku_name=self.sku_name,
                serial_pipeline_map=self.serial_pipeline_map,
                vit_training_root=self.vit_training_root,
                yolo_r_path=self.yolo_r_path,
                device=self.device,
                rebuild_dataset=self.rebuild_dataset,
                logger=self.status_signal.emit,
            )
            self.finished_signal.emit(summary)
        except Exception as e:
            self.error_signal.emit(str(e))


class NewSKUPage(QWidget):
    def __init__(
        self,
        media_path: str,
        raw_dir: str,
        save_root_dir: str,
        mydb=None,
        meta_collection: str = "New SKU",
        gridfs_bucket: str = "fs",
        sku_meta=None,
        on_close=None,
        parent=None
    ):
        super().__init__(parent)

        self.media_path = media_path
        self.raw_dir = raw_dir
        self.save_root_dir = save_root_dir
        self.mydb = mydb
        self.meta_collection = meta_collection
        self.gridfs_bucket = gridfs_bucket
        self.sku_meta = sku_meta or {}
        self.on_close = on_close

        self.labels = ["SIDE WALL 1", "SIDE WALL 2", "INNER SIDE", "TOP","BEAD"]

        self.img_labels = []
        self.status_lbl = None
        self.capture_btn = None
        self.training_btn = None
        self.refresh_btn = None
        self.close_btn = None

        self.capture_in_progress = False
        self.training_in_progress = False
        self.latest_preview_paths = {}
        self.training_worker = None

        self.stack = None
        self.capture_page = None
        self.training_page = None
        self.capture_tab_btn = None
        self.training_tab_btn = None

        self.training_progress = None
        self.training_status_lbl = None
        self.training_summary_lbl = None
        self.training_current_action_lbl = None
        self.training_percent_lbl = None

        self.camera_result_labels = {}
        self.camera_status_boxes = {}
        self.serial_status_state = {}

        self.serial_to_title = {
            "254701283": "Side Wall 1",
            "254701301": "Side Wall 2",
            "254701300": "Inner Side",
            "254701292": "Top",
            "254701293": "Bead",
        }

        self.active_training_serial = None
        self.current_gpu_training_serial = None
        self.enabled_training_serials = []
        self.serial_stage_progress = {}

        self._build_ui()

        QTimer.singleShot(100, self.load_raw_images_for_preview)

        self.preview_timer = QTimer(self)
        self.preview_timer.timeout.connect(self.refresh_preview_only)
        self.preview_timer.start(1500)

        QTimer.singleShot(0, self.refresh_preview_only)

    def set_sku_meta(self, sku_meta: dict):
        self.sku_meta = sku_meta or {}

    def _preview_serial_order(self):
         return [str(i+1) for i in range(len(self.labels))]

    def _ordered_preview_paths(self):
        return [self.latest_preview_paths.get(serial, "") for serial in self._preview_serial_order()]

    def load_raw_images_for_preview(self):
        if self.capture_in_progress or self.training_in_progress:
            return

        self.latest_preview_paths = {}
        preview_keys = self._preview_serial_order()  

        if os.path.exists(self.raw_dir):
            image_files = []
            for file in os.listdir(self.raw_dir):
                if file.lower().endswith(IMAGE_EXTS):
                    image_files.append(file)
            
            
            image_files.sort()
            
            # Match images by index (1.jpg -> key "1", etc.)
            for idx, key in enumerate(preview_keys):
                if idx < len(image_files):
                    image_path = os.path.join(self.raw_dir, image_files[idx])
                    if os.path.exists(image_path):
                        self.latest_preview_paths[key] = image_path
            
            # Also try to match by filename without extension
            if not self.latest_preview_paths:
                for file in image_files:
                    name_without_ext = os.path.splitext(file)[0]
                    if name_without_ext in preview_keys:
                        image_path = os.path.join(self.raw_dir, file)
                        self.latest_preview_paths[name_without_ext] = image_path

        
        self._update_preview_from_latest()
        
        if self.latest_preview_paths:
            self.status_lbl.setText(f"Loaded {len(self.latest_preview_paths)} images from raw folder")
        else:
            self.status_lbl.setText("No images found in raw folder")

    def _get_sku_name(self) -> str:
        for key in ("sku_name", "sku", "name", "pattern_name", "tyre_name"):
            value = self.sku_meta.get(key)
            if value:
                return str(value).strip()

        base_dir = os.path.join(self.media_path, "new_sku_images")
        if os.path.isdir(base_dir):
            folders = [
                name for name in os.listdir(base_dir)
                if os.path.isdir(os.path.join(base_dir, name))
            ]
            if len(folders) == 1:
                return folders[0]

        return "unknown_sku"

    def _tab_button_style(self, active: bool) -> str:
        if active:
            return """
                QPushButton {
                    background: transparent;
                    color: #571c86;
                    border: none;
                    border-bottom: 2px solid #571c86;
                    font: 600 11px 'Segoe UI';
                    padding: 4px 16px 3px 16px;
                }
            """
        return """
            QPushButton {
                background: transparent;
                color: #8a7f9c;
                border: none;
                border-bottom: 2px solid transparent;
                font: 500 11px 'Segoe UI';
                padding: 4px 16px 3px 16px;
            }
            QPushButton:hover {
                color: #571c86;
            }
        """

    def _switch_tab(self, idx: int):
        self.stack.setCurrentIndex(idx)
        self.capture_tab_btn.setStyleSheet(self._tab_button_style(idx == 0))
        self.training_tab_btn.setStyleSheet(self._tab_button_style(idx == 1))

    def _build_ui(self):
        self.setStyleSheet("""
            QWidget {
                background-color: #f8f7fa;
            }
            QStackedWidget {
                background: transparent;
            }
        """)
        root = QVBoxLayout(self)
        root.setContentsMargins(18, 10, 18, 12)
        root.setSpacing(12)

        nav_frame = QFrame()
        nav_frame.setFixedHeight(38)
        nav_frame.setStyleSheet("""
            QFrame {
                background: transparent;
                border: none;
            }
        """)
        nav_l = QHBoxLayout(nav_frame)
        nav_l.setContentsMargins(0, 0, 0, 0)
        nav_l.setSpacing(0)

        self.capture_tab_btn = QPushButton("Capture")
        self.capture_tab_btn.setCursor(Qt.PointingHandCursor)
        self.capture_tab_btn.setFixedHeight(34)
        self.capture_tab_btn.clicked.connect(lambda: self._switch_tab(0))
        nav_l.addWidget(self.capture_tab_btn)

        self.training_tab_btn = QPushButton("Training")
        self.training_tab_btn.setCursor(Qt.PointingHandCursor)
        self.training_tab_btn.setFixedHeight(34)
        self.training_tab_btn.clicked.connect(lambda: self._switch_tab(1))
        nav_l.addWidget(self.training_tab_btn)

        nav_l.addStretch(1)

        version_lbl = QLabel("v1.0")
        version_lbl.setStyleSheet("""
            font: 500 9px 'Segoe UI';
            color: #b9b0c7;
            padding: 0 6px;
        """)
        nav_l.addWidget(version_lbl)

        root.addWidget(nav_frame)

        self.stack = QStackedWidget()
        self.capture_page = QWidget()
        self.training_page = QWidget()

        self._build_capture_page()
        self._build_training_page()

        self.stack.addWidget(self.capture_page)
        self.stack.addWidget(self.training_page)
        root.addWidget(self.stack, 1)

        self._switch_tab(0)

    def _build_capture_page(self):
        root = QVBoxLayout(self.capture_page)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(12)

        main_card = QFrame()
        main_card.setObjectName("captureMainCard")
        main_card.setStyleSheet("""
            QFrame#captureMainCard {
                background: #ffffff;
                border: 1px solid #ebe5f2;
                border-radius: 18px;
            }
            QFrame#previewCard {
                background: #fcfbfe;
                border: 1px solid #eee7f6;
                border-radius: 16px;
            }
            QFrame#imageShell {
                background: #f7f4fb;
                border: 1px solid #e9e1f1;
                border-radius: 14px;
            }
            QFrame#actionBar {
                background: #faf8fd;
                border: 1px solid #eee7f6;
                border-radius: 14px;
            }
            QFrame#statusCard {
                background: #fbfafe;
                border: 1px solid #eee7f6;
                border-radius: 14px;
            }
        """)
        main_l = QVBoxLayout(main_card)
        main_l.setContentsMargins(18, 16, 18, 16)
        main_l.setSpacing(14)

        header_row = QHBoxLayout()
        header_row.setSpacing(12)

        header_left = QVBoxLayout()
        header_left.setSpacing(2)

        title_lbl = QLabel("New SKU Image Capture")
        title_lbl.setStyleSheet("""
            font: 700 18px 'Segoe UI';
            color: #571c86;
        """)
        header_left.addWidget(title_lbl)

        subtitle_lbl = QLabel("Capture and verify all 4 tyre views before starting training.")
        subtitle_lbl.setStyleSheet("""
            font: 500 11px 'Segoe UI';
            color: #7a7288;
        """)
        header_left.addWidget(subtitle_lbl)

        header_row.addLayout(header_left)
        header_row.addStretch(1)

        badge_lbl = QLabel("5 Cameras")
        badge_lbl.setAlignment(Qt.AlignCenter)
        badge_lbl.setFixedHeight(28)
        badge_lbl.setStyleSheet("""
            QLabel {
                background: #f4eefb;
                color: #571c86;
                border: 1px solid #e5d8f4;
                border-radius: 14px;
                font: 700 11px 'Segoe UI';
                padding: 0 12px;
            }
        """)
        header_row.addWidget(badge_lbl)

        main_l.addLayout(header_row)

        preview_grid = QGridLayout()
        preview_grid.setHorizontalSpacing(16)
        preview_grid.setVerticalSpacing(16)
        preview_grid.setContentsMargins(0, 0, 0, 0)

        self.img_labels = []

        for i in range(5):
            card = QFrame()
            card.setObjectName("previewCard")
            card.setMinimumSize(260, 560)
            card.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

            card_l = QVBoxLayout(card)
            card_l.setContentsMargins(12, 12, 12, 12)
            card_l.setSpacing(10)

            top_row = QHBoxLayout()
            top_row.setContentsMargins(0, 0, 0, 0)

            title = QLabel(self.labels[i].title())
            title.setStyleSheet("""
                font: 700 13px 'Segoe UI';
                color: #571c86;
            """)
            top_row.addWidget(title)

            top_row.addStretch(1)

            click_lbl = QLabel("Click to zoom")
            click_lbl.setStyleSheet("""
                font: 500 10px 'Segoe UI';
                color: #9a90aa;
            """)
            top_row.addWidget(click_lbl)

            card_l.addLayout(top_row)

            image_shell = QFrame()
            image_shell.setObjectName("imageShell")
            image_shell_l = QVBoxLayout(image_shell)
            image_shell_l.setContentsMargins(10, 10, 10, 10)
            image_shell_l.setSpacing(0)

            img = AspectImageLabel(title=self.labels[i].title())
            img.setMinimumSize(220, 470)
            image_shell_l.addWidget(img)

            card_l.addWidget(image_shell, 1)

            footer_lbl = QLabel("Latest preview")
            footer_lbl.setAlignment(Qt.AlignCenter)
            footer_lbl.setStyleSheet("""
                font: 500 10px 'Segoe UI';
                color: #8a8198;
                padding-top: 2px;
            """)
            card_l.addWidget(footer_lbl)

            self.img_labels.append(img)
            preview_grid.addWidget(card, 0, i)

        for col in range(4):
            preview_grid.setColumnStretch(col, 1)

        main_l.addLayout(preview_grid)

        action_bar = QFrame()
        action_bar.setObjectName("actionBar")
        action_l = QHBoxLayout(action_bar)
        action_l.setContentsMargins(14, 10, 14, 10)
        action_l.setSpacing(10)

        def mkbtn(text, primary=False):
            b = QPushButton(text)
            b.setCursor(Qt.PointingHandCursor)
            b.setFixedHeight(36)

            if primary:
                b.setStyleSheet("""
                    QPushButton {
                        background:#571c86;
                        color:white;
                        border:none;
                        border-radius:18px;
                        font: 700 11px 'Segoe UI';
                        padding: 0 18px;
                    }
                    QPushButton:hover { background:#6b2aa3; }
                    QPushButton:pressed { background:#481a6e; }
                    QPushButton:disabled {
                        background:#cfc3e0;
                        color:#f0ecf5;
                    }
                """)
            else:
                b.setStyleSheet("""
                    QPushButton {
                        background:#ffffff;
                        color:#571c86;
                        border:1px solid #dfd5ea;
                        border-radius:18px;
                        font: 700 11px 'Segoe UI';
                        padding: 0 18px;
                    }
                    QPushButton:hover {
                        background:#faf7fd;
                        border-color:#c9b7df;
                    }
                    QPushButton:pressed { background:#f3edf9; }
                    QPushButton:disabled {
                        color:#b7adca;
                        border-color:#e6e2ee;
                    }
                """)
            return b

        self.capture_btn = mkbtn("Start Capture", primary=True)
        self.capture_btn.clicked.connect(self.confirm_and_start_capture)
        action_l.addWidget(self.capture_btn)

        self.training_btn = mkbtn("Start Training", primary=False)
        self.training_btn.clicked.connect(self.confirm_and_start_training)
        action_l.addWidget(self.training_btn)

        self.refresh_btn = mkbtn("Refresh Preview", primary=False)
        self.refresh_btn.clicked.connect(self.refresh_preview_with_raw_load)
        action_l.addWidget(self.refresh_btn)

        action_l.addStretch(1)

        self.close_btn = mkbtn("Close", primary=False)
        self.close_btn.clicked.connect(self.close_page)
        action_l.addWidget(self.close_btn)

        main_l.addWidget(action_bar)

        status_card = QFrame()
        status_card.setObjectName("statusCard")
        status_l = QVBoxLayout(status_card)
        status_l.setContentsMargins(14, 10, 14, 10)
        status_l.setSpacing(4)

        status_title = QLabel("Status")
        status_title.setStyleSheet("""
            font: 700 11px 'Segoe UI';
            color: #571c86;
        """)
        status_l.addWidget(status_title)

        self.status_lbl = QLabel("Ready")
        self.status_lbl.setStyleSheet("""
            font: 500 11px 'Segoe UI';
            color:#5f5a6b;
        """)
        self.status_lbl.setWordWrap(True)
        self.status_lbl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        status_l.addWidget(self.status_lbl)

        main_l.addWidget(status_card)

        root.addWidget(main_card, 1)

    def _build_training_page(self):
        root = QVBoxLayout(self.training_page)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(12)

        main_card = QFrame()
        main_card.setObjectName("trainingMainCard")
        main_card.setStyleSheet("""
            QFrame#trainingMainCard {
                background: #ffffff;
                border: 1px solid #ebe5f2;
                border-radius: 18px;
            }
            QFrame#trainTopCard {
                background: #fcfbfe;
                border: 1px solid #eee7f6;
                border-radius: 16px;
            }
            QFrame#trainProgressCard {
                background: #faf8fd;
                border: 1px solid #eee7f6;
                border-radius: 16px;
            }
            QFrame#trainCamCard {
                background: #fcfbfe;
                border: 1px solid #eee7f6;
                border-radius: 16px;
            }
            QFrame#trainStatusBox {
                background: #f7f3fb;
                border: 1px solid #e8def2;
                border-radius: 12px;
            }
        """)
        main_l = QVBoxLayout(main_card)
        main_l.setContentsMargins(18, 16, 18, 16)
        main_l.setSpacing(14)

        header_row = QHBoxLayout()
        header_row.setSpacing(12)

        header_left = QVBoxLayout()
        header_left.setSpacing(2)

        title_lbl = QLabel("Model Training")
        title_lbl.setStyleSheet("""
            font: 700 18px 'Segoe UI';
            color: #571c86;
        """)
        header_left.addWidget(title_lbl)

        subtitle_lbl = QLabel("Monitor dataset preparation, epoch progress, and final result for each camera pipeline.")
        subtitle_lbl.setWordWrap(True)
        subtitle_lbl.setStyleSheet("""
            font: 500 11px 'Segoe UI';
            color: #7a7288;
        """)
        header_left.addWidget(subtitle_lbl)

        header_row.addLayout(header_left)
        header_row.addStretch(1)

        badge_lbl = QLabel("VIT Training")
        badge_lbl.setAlignment(Qt.AlignCenter)
        badge_lbl.setFixedHeight(28)
        badge_lbl.setStyleSheet("""
            QLabel {
                background: #f4eefb;
                color: #571c86;
                border: 1px solid #e5d8f4;
                border-radius: 14px;
                font: 700 11px 'Segoe UI';
                padding: 0 12px;
            }
        """)
        header_row.addWidget(badge_lbl)

        main_l.addLayout(header_row)

        top_card = QFrame()
        top_card.setObjectName("trainTopCard")
        top_l = QVBoxLayout(top_card)
        top_l.setContentsMargins(16, 14, 16, 14)
        top_l.setSpacing(10)

        self.training_status_lbl = QLabel("Training status: Waiting")
        self.training_status_lbl.setStyleSheet("""
            font: 700 15px 'Segoe UI';
            color:#571c86;
        """)
        top_l.addWidget(self.training_status_lbl)

        self.training_summary_lbl = QLabel("No training started yet.")
        self.training_summary_lbl.setWordWrap(True)
        self.training_summary_lbl.setStyleSheet("""
            font: 500 11px 'Segoe UI';
            color:#6f6a7a;
        """)
        top_l.addWidget(self.training_summary_lbl)

        self.training_current_action_lbl = QLabel("Current action: Waiting")
        self.training_current_action_lbl.setWordWrap(True)
        self.training_current_action_lbl.setStyleSheet("""
            font: 500 11px 'Segoe UI';
            color:#8f89a0;
        """)
        top_l.addWidget(self.training_current_action_lbl)

        main_l.addWidget(top_card)

        progress_card = QFrame()
        progress_card.setObjectName("trainProgressCard")
        progress_l = QVBoxLayout(progress_card)
        progress_l.setContentsMargins(16, 14, 16, 14)
        progress_l.setSpacing(10)

        prog_title = QLabel("Overall Progress")
        prog_title.setStyleSheet("""
            font: 700 12px 'Segoe UI';
            color:#571c86;
        """)
        progress_l.addWidget(prog_title)

        prog_row = QHBoxLayout()
        prog_row.setSpacing(10)

        self.training_progress = QProgressBar()
        self.training_progress.setRange(0, 100)
        self.training_progress.setValue(0)
        self.training_progress.setTextVisible(False)
        self.training_progress.setFixedHeight(12)
        self.training_progress.setStyleSheet("""
            QProgressBar {
                background:#eee9f5;
                border-radius:6px;
                border:none;
            }
            QProgressBar::chunk {
                background: qlineargradient(
                    x1:0, y1:0, x2:1, y2:0,
                    stop:0 #571c86,
                    stop:1 #7a43ae
                );
                border-radius:6px;
            }
        """)
        prog_row.addWidget(self.training_progress, 1)

        self.training_percent_lbl = QLabel("0%")
        self.training_percent_lbl.setAlignment(Qt.AlignCenter)
        self.training_percent_lbl.setFixedSize(54, 28)
        self.training_percent_lbl.setStyleSheet("""
            QLabel {
                background:#ffffff;
                border:1px solid #ddd2ea;
                border-radius:14px;
                font: 700 11px 'Segoe UI';
                color:#571c86;
            }
        """)
        prog_row.addWidget(self.training_percent_lbl)

        progress_l.addLayout(prog_row)

        progress_hint = QLabel("Progress updates automatically as each serial completes dataset prep and training.")
        progress_hint.setWordWrap(True)
        progress_hint.setStyleSheet("""
            font: 500 10px 'Segoe UI';
            color:#958ca5;
        """)
        progress_l.addWidget(progress_hint)

        main_l.addWidget(progress_card)

        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(16)
        grid.setVerticalSpacing(16)

        self.camera_result_labels = {}
        self.camera_status_boxes = {}

        serials = ["254701283", "254701301", "254701300", "254701292","254701293"]

        for idx, serial in enumerate(serials):
            card = QFrame()
            card.setObjectName("trainCamCard")
            card.setMinimumHeight(190)
            card.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

            cl = QVBoxLayout(card)
            cl.setContentsMargins(14, 14, 14, 14)
            cl.setSpacing(10)

            top_row = QHBoxLayout()
            top_row.setSpacing(8)

            title = QLabel(self.serial_to_title.get(serial, serial))
            title.setStyleSheet("""
                font: 700 13px 'Segoe UI';
                color:#571c86;
            """)
            top_row.addWidget(title)

            top_row.addStretch(1)

            pipe_name = CAMERA_PIPELINE_MAP.get(serial, "")
            pipe_badge = QLabel(pipe_name if pipe_name else "not configured")
            pipe_badge.setAlignment(Qt.AlignCenter)
            pipe_badge.setFixedHeight(24)
            pipe_badge.setStyleSheet("""
                QLabel {
                    background:#f3edf9;
                    color:#6b4b8f;
                    border:1px solid #e2d8ef;
                    border-radius:12px;
                    font: 600 10px 'Segoe UI';
                    padding: 0 10px;
                }
            """)
            top_row.addWidget(pipe_badge)

            cl.addLayout(top_row)

            serial_lbl = QLabel(f"Camera Serial: {serial}")
            serial_lbl.setWordWrap(True)
            serial_lbl.setStyleSheet("""
                font: 500 10px 'Segoe UI';
                color:#9a90aa;
            """)
            cl.addWidget(serial_lbl)

            status_box = QFrame()
            status_box.setObjectName("trainStatusBox")
            status_box_l = QVBoxLayout(status_box)
            status_box_l.setContentsMargins(12, 10, 12, 10)
            status_box_l.setSpacing(6)

            status_title = QLabel("Pipeline Status")
            status_title.setStyleSheet("""
                font: 700 11px 'Segoe UI';
                color:#6a5a82;
            """)
            status_box_l.addWidget(status_title)

            result_lbl = QLabel("Status: Not started")
            result_lbl.setWordWrap(True)
            result_lbl.setAlignment(Qt.AlignTop | Qt.AlignLeft)
            result_lbl.setMinimumHeight(72)
            result_lbl.setStyleSheet("""
                font: 600 11px 'Segoe UI';
                color:#6f6a7a;
            """)
            status_box_l.addWidget(result_lbl)

            cl.addWidget(status_box, 1)

            footer_lbl = QLabel("Live status will appear here during training.")
            footer_lbl.setWordWrap(True)
            footer_lbl.setStyleSheet("""
                font: 500 10px 'Segoe UI';
                color:#9a90aa;
            """)
            cl.addWidget(footer_lbl)

            self.camera_result_labels[serial] = result_lbl
            self.camera_status_boxes[serial] = status_box

            row = idx // 2
            col = idx % 2
            grid.addWidget(card, row, col)

        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)

        main_l.addLayout(grid, 1)
        root.addWidget(main_card, 1)

    def _status_theme(self, state: str):
        themes = {
            "waiting":  ("#6f6a7a", "#f7f3fb", "#e8def2"),
            "prep":     ("#b16e2c", "#fff7ee", "#efd8b3"),
            "queued":   ("#1f6390", "#eef7ff", "#bfdcf2"),
            "training": ("#1f6390", "#eef4ff", "#c9daf8"),
            "done":     ("#2a6e2f", "#eefaf0", "#cbe8cf"),
            "failed":   ("#bd3b2b", "#fff1ef", "#f3c7c1"),
            "skipped":  ("#9a92ae", "#f5f3f8", "#dfd9e8"),
            "unknown":  ("#333333", "#f7f3fb", "#e8def2"),
        }
        return themes.get(state, themes["unknown"])

    def _refresh_training_summary_counts(self):
        if not self.enabled_training_serials:
            return

        counts = {
            "prep": 0,
            "queued": 0,
            "training": 0,
            "done": 0,
            "failed": 0,
            "skipped": 0,
            "waiting": 0,
        }

        for serial in self.enabled_training_serials:
            state = self.serial_status_state.get(serial, "waiting")
            if state in counts:
                counts[state] += 1
            else:
                counts["waiting"] += 1

        parts = []
        if counts["prep"]:
            parts.append(f"{counts['prep']} preparing")
        if counts["queued"]:
            parts.append(f"{counts['queued']} waiting for GPU")
        if counts["training"]:
            parts.append(f"{counts['training']} training")
        if counts["done"]:
            parts.append(f"{counts['done']} done")
        if counts["failed"]:
            parts.append(f"{counts['failed']} failed")
        if counts["skipped"]:
            parts.append(f"{counts['skipped']} skipped")
        if counts["waiting"]:
            parts.append(f"{counts['waiting']} waiting")

        summary_text = ", ".join(parts) if parts else "Training is in progress for configured cameras."

        if self.training_summary_lbl is not None and self.training_in_progress:
            self.training_summary_lbl.setText(summary_text)

    def _set_camera_status(self, serial: str, text: str, color: str = "#333", state: str = "unknown"):
        lbl = self.camera_result_labels.get(serial)
        box = self.camera_status_boxes.get(serial)
        if lbl is not None:
            lbl.setText(text)
            lbl.setStyleSheet(f"""
                font: 600 11px 'Segoe UI';
                color: {color};
            """)

        fg, bg, border = self._status_theme(state)
        if box is not None:
            box.setStyleSheet(f"""
                QFrame {{
                    background: {bg};
                    border: 1px solid {border};
                    border-radius: 12px;
                }}
            """)

        self.serial_status_state[serial] = state
        self._refresh_training_summary_counts()

    def _compact_training_msg(self, msg: str, max_len: int = 120) -> str:
        msg = (msg or "").strip().replace("\n", " ")
        if len(msg) <= max_len:
            return msg
        return msg[:max_len - 3] + "..."

    def _reset_training_cards(self):
        self.serial_status_state = {}
        for serial in ["254701283", "254701301", "254701300", "254701292","254701293"]:
            if CAMERA_PIPELINE_MAP.get(serial):
                self._set_camera_status(serial, "Status: Waiting", "#6f6a7a", "waiting")
            else:
                self._set_camera_status(serial, "Status: Not configured", "#b3aac5", "skipped")

    def _reset_training_progress(self):
        self.enabled_training_serials = [s for s, p in CAMERA_PIPELINE_MAP.items() if p]
        self.serial_stage_progress = {s: 0.0 for s in self.enabled_training_serials}
        self.active_training_serial = None
        self.current_gpu_training_serial = None

        if self.training_progress is not None:
            self.training_progress.setRange(0, 100)
            self.training_progress.setValue(0)

        if self.training_percent_lbl is not None:
            self.training_percent_lbl.setText("0%")

    def _set_serial_progress(self, serial: str, frac: float):
        if serial not in self.serial_stage_progress or not self.enabled_training_serials:
            return

        frac = max(0.0, min(1.0, frac))
        self.serial_stage_progress[serial] = max(self.serial_stage_progress.get(serial, 0.0), frac)

        total_frac = sum(self.serial_stage_progress.values()) / max(len(self.enabled_training_serials), 1)
        pct = int(total_frac * 100)

        if self.training_progress is not None:
            self.training_progress.setRange(0, 100)
            self.training_progress.setValue(pct)

        if self.training_percent_lbl is not None:
            self.training_percent_lbl.setText(f"{pct}%")

    def _update_training_card_from_log(self, msg: str):
        compact_msg = self._compact_training_msg(msg)

        if self.training_current_action_lbl is not None:
            self.training_current_action_lbl.setText(f"Current action: {compact_msg}")

        if msg.startswith("[MODE]"):
            if self.training_summary_lbl is not None:
                self.training_summary_lbl.setText("Parallel dataset preparation is running. GPU training will start serial by serial.")
            return

        m = re.search(r"\[PREP\]\s+serial=(\d+)", msg)
        if m:
            serial = m.group(1)
            self._set_camera_status(serial, "Status: Preparing dataset...", "#b16e2c", "prep")
            self._set_serial_progress(serial, 0.10)
            return

        m = re.search(r"\[PREP-DONE\]\s+serial=(\d+)", msg)
        if m:
            serial = m.group(1)
            self._set_camera_status(serial, "Status: Dataset ready.\nWaiting for GPU training...", "#1f6390", "queued")
            self._set_serial_progress(serial, 0.45)
            return

        m = re.search(r"\[PREP-FAIL\]\s+serial=(\d+)\s+\|\s+(.*)", msg)
        if m:
            serial = m.group(1)
            err = m.group(2).strip()
            self._set_camera_status(serial, f"Status: Prep failed\n{err[:70]}", "#bd3b2b", "failed")
            self._set_serial_progress(serial, 1.0)
            return

        m = re.search(r"\[TRAIN\]\s+serial=(\d+)", msg)
        if m:
            serial = m.group(1)
            self.active_training_serial = serial
            self.current_gpu_training_serial = serial
            self._set_camera_status(serial, "Status: Training started on GPU...", "#1f6390", "training")
            self._set_serial_progress(serial, 0.60)
            return

        if "[PIPELINE] Dataset preparation complete." in msg and self.current_gpu_training_serial:
            serial = self.current_gpu_training_serial
            self._set_camera_status(serial, "Status: Dataset ready.\nStarting GPU training...", "#1f6390", "training")
            self._set_serial_progress(serial, 0.60)
            return

        m = re.search(r"Epoch\s*\[(\d+)/(\d+)\]", msg)
        if m and self.current_gpu_training_serial:
            ep = int(m.group(1))
            total = int(m.group(2))
            frac = 0.60 + 0.35 * (ep / max(total, 1))
            serial = self.current_gpu_training_serial
            self._set_camera_status(serial, f"Status: Training epoch {ep}/{total}", "#1f6390", "training")
            self._set_serial_progress(serial, frac)
            return

        m = re.search(r"\[DONE\]\s+serial=(\d+)", msg)
        if m:
            serial = m.group(1)
            self._set_camera_status(serial, "Status: Completed", "#2a6e2f", "done")
            self._set_serial_progress(serial, 1.0)
            if self.current_gpu_training_serial == serial:
                self.current_gpu_training_serial = None
            return

        m = re.search(r"\[FAIL\]\s+serial=(\d+)\s+\|\s+(.*)", msg)
        if m:
            serial = m.group(1)
            err = m.group(2).strip()
            self._set_camera_status(serial, f"Status: Failed\n{err[:70]}", "#bd3b2b", "failed")
            self._set_serial_progress(serial, 1.0)
            if self.current_gpu_training_serial == serial:
                self.current_gpu_training_serial = None
            return

        m = re.search(r"\[SKIP\]\s+serial=(\d+)\s+\|\s+(.*)", msg)
        if m:
            serial = m.group(1)
            reason = m.group(2).strip()
            self._set_camera_status(serial, f"Status: Skipped\n{reason[:70]}", "#9a92ae", "skipped")
            self._set_serial_progress(serial, 1.0)
            return

    def _set_controls_enabled(self, enabled: bool):
        self.capture_btn.setEnabled(enabled)
        self.training_btn.setEnabled(enabled)
        self.refresh_btn.setEnabled(enabled)
        self.close_btn.setEnabled(enabled)
        self.capture_tab_btn.setEnabled(enabled or not self.capture_in_progress)
        self.training_tab_btn.setEnabled(enabled or not self.training_in_progress)

    def refresh_preview_only(self):
        if self.capture_in_progress or self.training_in_progress:
            return

        if not self.latest_preview_paths:
            self.load_raw_images_for_preview()

        # Get preview paths in correct order
        preview_paths = []
        for key in self._preview_serial_order():
            preview_paths.append(self.latest_preview_paths.get(key, ""))
        
        # Ensure we have exactly len(self.labels) items
        while len(preview_paths) < len(self.labels):
            preview_paths.append("")

        for i in range(len(self.labels)):
            if i < len(self.img_labels):
                self.img_labels[i].set_image_path(preview_paths[i])

    def refresh_preview_with_raw_load(self):
        if self.capture_in_progress or self.training_in_progress:
            return

        self.load_raw_images_for_preview()
        self.refresh_preview_only()

        if self.latest_preview_paths:
            self.status_lbl.setText(f"Loaded {len(self.latest_preview_paths)} images from raw folder")
        else:
            self.status_lbl.setText("No images found in raw folder")

    def _update_preview_from_latest(self):
        preview_paths = self._ordered_preview_paths()
        # Use len(self.labels) instead of hardcoded 4
        while len(preview_paths) < len(self.labels):
            preview_paths.append("")

        for i in range(len(self.labels)):  # Use dynamic length
            self.img_labels[i].set_image_path(preview_paths[i])

    def confirm_and_start_capture(self):
        if self.capture_in_progress or self.training_in_progress:
            return

        reply = QMessageBox.question(
            self,
            "Start Capture",
            "Capture 20 good images with one good tyre.\n\nAfter placing one good tyre, click OK to start capture.",
            QMessageBox.Ok | QMessageBox.Cancel,
            QMessageBox.Cancel
        )

        if reply == QMessageBox.Ok:
            self.start_capture()

    def start_capture(self):
        if self.capture_in_progress or self.training_in_progress:
            return

        self.capture_in_progress = True
        self._set_controls_enabled(False)
        self.preview_timer.stop()
        self._switch_tab(0)

        IMAGES_PER_CAMERA = 20
        GOOD_FOLDER_COUNT = 10
        EXPECTED_CAMERAS = 5

        sku_name = self._get_sku_name()
        sku_folder = _safe_name(sku_name)

        base_out_dir = _ensure_dir(
            os.path.join(self.media_path, "new_sku_images", sku_folder)
        )

        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.latest_preview_paths = {}

        self.status_lbl.setText(f"Starting for SKU={sku_name} ...")
        QApplication.processEvents()

        try:
            for shot_idx in range(1, IMAGES_PER_CAMERA + 1):
                pct = int((shot_idx / IMAGES_PER_CAMERA) * 100)
                self.status_lbl.setText(
                    f"Capturing set {shot_idx}/{IMAGES_PER_CAMERA} ({pct}%) ..."
                )
                QApplication.processEvents()

                captured_results = capture_images_from_all_cameras(
                    expected_cameras=EXPECTED_CAMERAS
                )

                if not captured_results:
                    raise Exception(f"No images captured in set {shot_idx}")

                if len(captured_results) != EXPECTED_CAMERAS:
                    raise Exception(
                        f"Expected {EXPECTED_CAMERAS} camera images, "
                        f"but got {len(captured_results)} in set {shot_idx}"
                    )

                capture_stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

                for serial, image in captured_results:
                    serial_str = _safe_name(str(serial))
                    serial_base_dir = _ensure_dir(os.path.join(base_out_dir, serial_str))

                    if shot_idx <= GOOD_FOLDER_COUNT:
                        save_dir = _ensure_dir(os.path.join(serial_base_dir, "train", "good"))
                        save_group = "train_good"
                    else:
                        save_dir = serial_base_dir
                        save_group = "serial_root"

                    file_name = f"{serial_str}_{capture_stamp}_{shot_idx:03d}.png"
                    file_path = os.path.join(save_dir, file_name)

                    ok = cv2.imwrite(file_path, image)
                    if not ok:
                        raise Exception(f"Failed to save image: {file_path}")

                    self.latest_preview_paths[serial_str] = file_path

                    db_meta = dict(self.sku_meta or {})
                    db_meta.update({
                        "sku_name": sku_name,
                        "sku_folder": sku_folder,
                        "camera_serial": serial_str,
                        "session_id": session_id,
                        "capture_index": shot_idx,
                        "total_images_per_camera": IMAGES_PER_CAMERA,
                        "save_group": save_group,
                        "saved_dir": save_dir,
                        "saved_file": file_name,
                    })

                    save_new_sku_image(
                        file_path=file_path,
                        label=serial_str,
                        capture_id=session_id,
                        sku_meta=db_meta,
                        meta_collection=self.meta_collection,
                        gridfs_bucket=self.gridfs_bucket
                    )

                self._update_preview_from_latest()
                QApplication.processEvents()

            self.status_lbl.setText(
                f"Completed | 20 images per camera saved for SKU={sku_name}"
            )
            QApplication.processEvents()

            QMessageBox.information(
                self,
                "Capture Complete",
                "20 images per camera captured successfully.\n\nFirst 10 saved in train/good.\nNext 10 saved in camera serial folder.\n\nCapturing stopped."
            )

        except Exception as e:
            self.status_lbl.setText("Failed")
            QMessageBox.critical(self, "Capture Error", str(e))

        finally:
            self.capture_in_progress = False
            self._set_controls_enabled(True)
            self.preview_timer.start(1500)

    def confirm_and_start_training(self):
        if self.capture_in_progress or self.training_in_progress:
            return

        sku_name = self._get_sku_name()
        sku_folder = _safe_name(sku_name)
        sku_root = os.path.join(self.media_path, "new_sku_images", sku_folder)

        if not os.path.exists(sku_root):
            QMessageBox.warning(
                self,
                "Training",
                f"SKU folder not found:\n{sku_root}\n\nCapture first."
            )
            return

        reply = QMessageBox.question(
            self,
            "Start Training",
            "Start VIT training now?\n\nTraining will use:\n- train/good images from each camera serial folder\n- one random image from each serial root as reference\n\nOnly configured camera serials will be trained.",
            QMessageBox.Ok | QMessageBox.Cancel,
            QMessageBox.Cancel
        )

        if reply == QMessageBox.Ok:
            self.start_training()

    def start_training(self):
        if self.capture_in_progress or self.training_in_progress:
            return

        self.capture_in_progress = False
        self.training_in_progress = True
        self._set_controls_enabled(False)
        self.preview_timer.stop()

        sku_name = self._get_sku_name()

        self._switch_tab(1)
        self._reset_training_cards()
        self._reset_training_progress()

        self.training_status_lbl.setText(f"Training status: Starting for SKU={sku_name}")
        self.training_summary_lbl.setText("Dataset preparation and training are in progress.")
        self.training_current_action_lbl.setText("Current action: Waiting for training to start...")
        self.status_lbl.setText(f"Training running for SKU={sku_name} ...")

        print(f"[DEBUG] VIT_TRAINING_ROOT = {VIT_TRAINING_ROOT}")
        print(f"[DEBUG] YOLO_R_PATH = {YOLO_R_PATH}")

        self.training_worker = TrainingWorker(
            media_path=self.media_path,
            sku_name=sku_name,
            serial_pipeline_map=CAMERA_PIPELINE_MAP,
            vit_training_root=VIT_TRAINING_ROOT,
            yolo_r_path=YOLO_R_PATH,
            device="cuda",
            rebuild_dataset=True,
            parent=self
        )
        self.training_worker.status_signal.connect(self._on_training_status)
        self.training_worker.finished_signal.connect(self._on_training_finished)
        self.training_worker.error_signal.connect(self._on_training_error)
        self.training_worker.start()

    def _on_training_status(self, msg: str):
        self.training_status_lbl.setText("Training status: Running")
        self.training_summary_lbl.setText("Dataset preparation and training are in progress.")
        self.status_lbl.setText(self._compact_training_msg(msg, 90))
        self._update_training_card_from_log(msg)

    def _on_training_finished(self, summary: dict):
        self.training_in_progress = False
        self._set_controls_enabled(True)
        self.preview_timer.start(1500)

        self.training_progress.setRange(0, 100)
        self.training_progress.setValue(100)
        self.training_percent_lbl.setText("100%")

        summary_path = summary.get("summary_path", "")
        self.status_lbl.setText("Training completed")
        self.training_status_lbl.setText("Training status: Completed")
        self.training_summary_lbl.setText(f"Summary saved at:\n{summary_path}")
        self.training_current_action_lbl.setText("Current action: All configured training jobs finished.")

        for item in summary.get("results", []):
            serial = str(item.get("camera_serial", ""))
            if serial in self.camera_result_labels:
                if item.get("success", False):
                    run_dir = item.get("run_dir", "")
                    self._set_camera_status(
                        serial,
                        f"Status: Completed\n{run_dir[-70:] if run_dir else ''}",
                        "#2a6e2f",
                        "done"
                    )
                else:
                    err = item.get("error", "Unknown error")
                    self._set_camera_status(
                        serial,
                        f"Status: Failed\n{err[:70]}",
                        "#bd3b2b",
                        "failed"
                    )

        for item in summary.get("skipped", []):
            serial = str(item.get("camera_serial", ""))
            if serial in self.camera_result_labels:
                self._set_camera_status(
                    serial,
                    f"Status: Skipped\n{item.get('reason', '')[:70]}",
                    "#9a92ae",
                    "skipped"
                )

        self._switch_tab(1)

    def _on_training_error(self, err: str):
        self.training_in_progress = False
        self._set_controls_enabled(True)
        self.preview_timer.start(1500)

        self.training_progress.setRange(0, 100)
        self.training_progress.setValue(0)
        self.training_percent_lbl.setText("0%")

        self.status_lbl.setText("Training failed")
        self.training_status_lbl.setText("Training status: Failed")
        self.training_summary_lbl.setText(err)
        self.training_current_action_lbl.setText("Current action: Training stopped due to error.")

        self._switch_tab(1)
        QMessageBox.critical(self, "Training Error", err)

    def close_page(self):
        if self.capture_in_progress:
            QMessageBox.warning(
                self,
                "Capture Running",
                "Capture is in progress. Please wait until it finishes."
            )
            return

        if self.training_in_progress:
            QMessageBox.warning(
                self,
                "Training Running",
                "Training is in progress. Please wait until it finishes."
            )
            return

        reply = QMessageBox.question(
            self,
            "Close",
            "Close New SKU capture and return to Dashboard?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            if callable(self.on_close):
                self.on_close()