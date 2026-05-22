import os
import re
import cv2
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List

from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal, QEvent, QSize  # type: ignore
from PyQt5.QtGui import QPixmap  # type: ignore
from PyQt5.QtWidgets import (  # type: ignore
    QWidget, QVBoxLayout, QHBoxLayout, QFrame, QLabel,
    QPushButton, QProgressBar, QMessageBox, QSizePolicy, QApplication,
    QGridLayout, QScrollArea, QDialog, QStackedWidget,
    QFormLayout, QLineEdit, QSpinBox,
    QTableWidget, QTableWidgetItem, QHeaderView,QComboBox
)

from src.COMMON.common import load_env
from src.COMMON.db import save_new_sku_image
from src.COMMON.recipe_service import RecipeService
from src.COMMON.model_validation_service import run_validation_for_sku
from src.training.central_vit_trainer import run_training_for_sku

try:
    from src.camera.new_sku_software_capture import capture_new_sku_images # type: ignore
except Exception:
    capture_new_sku_images = None


IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

TAB_SKU_SETUP = 0
TAB_AXIS_TEACHING = 1
TAB_CAPTURE = 2
TAB_TRAINING = 3
TAB_VALIDATION = 4
TAB_SAVE_RECIPE = 5


# =========================
# CAMERA / TRAINING CONFIG
# =========================
BASE_SRC_DIR = Path(__file__).resolve().parents[1]   # .../src
PROJECT_ROOT = BASE_SRC_DIR.parent
ENV_PATH = PROJECT_ROOT / ".env"

env_vars = load_env(str(ENV_PATH))

CAMERA_ROLE_ORDER = [
    ("sidewall1", "Side Wall 1", str(env_vars.get("CAM_SIDEWALL1_SERIAL", "")).strip()),
    ("sidewall2", "Side Wall 2", str(env_vars.get("CAM_SIDEWALL2_SERIAL", "")).strip()),
    ("innerwall", "Inner Side", str(env_vars.get("CAM_INNERWALL_SERIAL", "")).strip()),
    ("tread", "Tread", str(env_vars.get("CAM_TREAD_SERIAL", "")).strip()),
    ("bead", "Bead", str(env_vars.get("CAM_BEAD_SERIAL", "")).strip()),
]
CAMERA_ROLE_ORDER = [item for item in CAMERA_ROLE_ORDER if item[2]]

CAMERA_PIPELINE_MAP = {serial: role for role, title, serial in CAMERA_ROLE_ORDER}
CAMERA_SERIAL_ORDER = [serial for role, title, serial in CAMERA_ROLE_ORDER]
CAMERA_TITLE_MAP = {serial: title for role, title, serial in CAMERA_ROLE_ORDER}

TRAINING_DIR = BASE_SRC_DIR / "training"
VIT_TRAINING_ROOT = str(TRAINING_DIR / "VIT_Training")

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


def _to_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)

def _to_float_or_none(value: Any):
    try:
        text = str(value or "").strip()
        if text == "":
            return None
        return float(text)
    except Exception:
        return None

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

        def mkbtn(text: str) -> QPushButton:
            b = QPushButton(text)
            b.setFixedHeight(32)
            b.setCursor(Qt.PointingHandCursor)
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
        scaled = self._pixmap.scaled(w, h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
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
    PREVIEW_W = 210
    PREVIEW_H = 430

    def __init__(self, title: str = "", parent=None):
        super().__init__(parent)
        self._pm = None
        self._image_path = ""
        self._title = title

        self.setAlignment(Qt.AlignCenter)
        self.setCursor(Qt.PointingHandCursor)
        self.setFixedSize(self.PREVIEW_W, self.PREVIEW_H)
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.setStyleSheet("""
            QLabel {
                background: #faf9fc;
                border: 1px solid #e9e4f1;
                border-radius: 12px;
            }
        """)

    def sizeHint(self):
        return QSize(self.PREVIEW_W, self.PREVIEW_H)

    def minimumSizeHint(self):
        return QSize(self.PREVIEW_W, self.PREVIEW_H)

    def set_image_path(self, path: str):
        path = path or ""
        if path == self._image_path and self._pm is not None:
            return
        self._image_path = path
        if path and os.path.exists(path):
            pm = QPixmap(path)
            self._pm = pm if not pm.isNull() else None
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
            return
        scaled = self._pm.scaled(self.width(), self.height(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.setText("")
        self.setPixmap(scaled)


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
        parent=None,
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
            self.finished_signal.emit(summary or {})
        except Exception as e:
            self.error_signal.emit(str(e))

class CaptureWorker(QThread):
    status_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(dict)
    error_signal = pyqtSignal(str)

    def __init__(
        self,
        sku_name: str,
        media_path: str,
        images_per_camera: int,
        train_good_count: int = 10,
        multi_camera_manager=None,
        sku_meta=None,
        meta_collection: str = "New SKU",
        gridfs_bucket: str = "fs",
        parent=None,
    ):
        super().__init__(parent)
        self.sku_name = sku_name
        self.media_path = media_path
        self.images_per_camera = images_per_camera
        self.train_good_count = train_good_count
        self.multi_camera_manager = multi_camera_manager
        self.sku_meta = dict(sku_meta or {})
        self.meta_collection = meta_collection
        self.gridfs_bucket = gridfs_bucket

    def run(self):
        try:
            result = capture_new_sku_images(
                sku_name=self.sku_name,
                media_path=self.media_path,
                images_per_camera=self.images_per_camera,
                train_good_count=self.train_good_count,
                multi_camera_manager=self.multi_camera_manager,
                sku_meta=self.sku_meta,
                meta_collection=self.meta_collection,
                gridfs_bucket=self.gridfs_bucket,
                logger=self.status_signal.emit,
            )
            self.finished_signal.emit(result or {})
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
        plc_client=None,
        multi_camera_manager=None,
        parent=None,
    ):
        super().__init__(parent)

        self.media_path = media_path
        self.raw_dir = raw_dir
        self.save_root_dir = save_root_dir
        self.mydb = mydb
        self.meta_collection = meta_collection
        self.gridfs_bucket = gridfs_bucket
        self.sku_meta = dict(sku_meta or {})
        self.sku_meta.pop("machine_serial", None)  # removed by requirement
        self.on_close = on_close
        self.plc_client = plc_client
        self.multi_camera_manager = multi_camera_manager

        self.labels = ["SIDE WALL 1", "SIDE WALL 2", "INNER SIDE", "TREAD", "BEAD"]

        self.img_labels: List[AspectImageLabel] = []
        self.status_lbl: Optional[QLabel] = None
        self.capture_btn: Optional[QPushButton] = None
        self.training_btn: Optional[QPushButton] = None
        self.refresh_btn: Optional[QPushButton] = None
        self.close_btn: Optional[QPushButton] = None

        self.capture_in_progress = False
        self.training_in_progress = False
        self.latest_preview_paths: Dict[str, str] = {}
        self.training_worker: Optional[TrainingWorker] = None
        self.capture_worker: Optional[CaptureWorker] = None
        self.recipe_service = RecipeService(
            media_path=self.media_path,
            plc_client=self.plc_client,
        )
        self.recipe_doc: Dict[str, Any] = {}
        self.saved_recipe_doc: Optional[Dict[str, Any]] = None
        self.saved_recipe_result: Optional[Dict[str, Any]] = None
        self.load_machine_btn: Optional[QPushButton] = None

        self.latest_training_summary: Dict[str, Any] = {}
        self.latest_validation_result: Dict[str, Any] = {}

        self.tab_buttons: List[QPushButton] = []
        self.wizard_widgets: Dict[str, Any] = {}

        self.stack: Optional[QStackedWidget] = None
        self.wizard_page: Optional[QWidget] = None
        self.axis_teaching_page: Optional[QWidget] = None
        self.capture_page: Optional[QWidget] = None
        self.training_page: Optional[QWidget] = None
        self.validation_page: Optional[QWidget] = None
        self.recipe_page: Optional[QWidget] = None
        self.axis_entry_mode = "capture"
        self.axis_entry_mode_combo = None
        self.apply_manual_axis_btn = None
        self.axis_table: Optional[QTableWidget] = None
        self.validation_status_lbl: Optional[QLabel] = None
        self.validation_metrics_lbl: Optional[QLabel] = None
        self.recipe_summary_lbl: Optional[QLabel] = None

        self.training_progress: Optional[QProgressBar] = None
        self.training_status_lbl: Optional[QLabel] = None
        self.training_summary_lbl: Optional[QLabel] = None
        self.training_current_action_lbl: Optional[QLabel] = None
        self.training_percent_lbl: Optional[QLabel] = None

        self.camera_result_labels: Dict[str, QLabel] = {}
        self.camera_status_boxes: Dict[str, QFrame] = {}
        self.serial_status_state: Dict[str, str] = {}

        self.serial_to_title = dict(CAMERA_TITLE_MAP)
        self.camera_serial_order = list(CAMERA_SERIAL_ORDER)
        self.enabled_training_serials: List[str] = []
        self.serial_stage_progress: Dict[str, float] = {}
        self.active_training_serial = None
        self.current_gpu_training_serial = None

        self._build_ui()

        QTimer.singleShot(100, self.load_raw_images_for_preview)
        self.preview_timer = QTimer(self)
        self.preview_timer.timeout.connect(self.refresh_preview_only)
        self.preview_timer.start(1500)
        QTimer.singleShot(0, self.refresh_preview_only)


    def _on_capture_status(self, message: str):
        if self.status_lbl is not None:
            self.status_lbl.setText(str(message))


    def _on_capture_finished(self, result: dict):
        self.latest_preview_paths = result or {}
        self._update_preview_from_latest()

        sku_name = _safe_name(self._get_sku_name())

        if self.status_lbl is not None:
            self.status_lbl.setText(
                f"Capture completed. Saved in media/new_sku_images/{sku_name}/<camera_serial>/"
            )

        QMessageBox.information(
            self,
            "Capture Complete",
            f"Images saved in:\nmedia/new_sku_images/{sku_name}/<camera_serial>/",
        )

        self.capture_in_progress = False
        self._set_controls_enabled(True)

        if self.preview_timer:
            self.preview_timer.start(1500)

        if self.capture_worker is not None:
            self.capture_worker.deleteLater()
            self.capture_worker = None


    def _on_capture_error(self, message: str):
        QMessageBox.critical(self, "Capture Error", str(message))

        if self.status_lbl is not None:
            self.status_lbl.setText(f"Capture failed: {message}")

        self.capture_in_progress = False
        self._set_controls_enabled(True)

        if self.preview_timer:
            self.preview_timer.start(1500)

        if self.capture_worker is not None:
            self.capture_worker.deleteLater()
            self.capture_worker = None
    def set_plc_client(self, plc_client):
        self.plc_client = plc_client

        if hasattr(self, "recipe_service") and self.recipe_service is not None:
            if hasattr(self.recipe_service, "set_plc_client"):
                self.recipe_service.set_plc_client(plc_client)

    def set_multi_camera_manager(self, multi_camera_manager):
        self.multi_camera_manager = multi_camera_manager
    # ======================================================================
    # THEME HELPERS
    # ======================================================================
    def _page_stylesheet(self) -> str:
        return """
            QWidget {
                background: #f6f4f9;
                color: #2f2a36;
                font: 10pt 'Segoe UI';
            }
            QStackedWidget { background: transparent; }
            QFrame#PageCard {
                background: #ffffff;
                border: 1px solid #e6deef;
                border-radius: 18px;
            }
            QFrame#InnerCard {
                background: #fbf9fd;
                border: 1px solid #eee6f6;
                border-radius: 14px;
            }
            QFrame#ActionBar {
                background: #faf8fd;
                border: 1px solid #eee7f6;
                border-radius: 14px;
            }
            QFrame#StatusCard {
                background: #fbfafe;
                border: 1px solid #eee7f6;
                border-radius: 14px;
            }
            QLabel#PageTitle {
                font: 800 20px 'Segoe UI';
                color: #571c86;
                background: transparent;
                border: none;
            }
            QLabel#PageSubTitle {
                font: 500 11px 'Segoe UI';
                color: #7b7288;
                background: transparent;
                border: none;
            }
            QLabel#SectionTitle {
                font: 750 13px 'Segoe UI';
                color: #571c86;
                background: transparent;
                border: none;
            }
            QLabel#HintText {
                font: 500 10px 'Segoe UI';
                color: #8e86a0;
                background: transparent;
                border: none;
            }
            QLabel#InfoBox {
                background: #fbf9fd;
                border: 1px solid #ebe3f4;
                border-radius: 14px;
                padding: 16px;
                font: 500 11px 'Segoe UI';
                color: #4e4758;
            }
            QLabel#StatusPill {
                background: #f4eefb;
                color: #571c86;
                border: 1px solid #dfd2ef;
                border-radius: 12px;
                padding: 10px 14px;
                font: 700 11px 'Segoe UI';
            }
            QLineEdit, QSpinBox {
                background: #ffffff;
                border: 1px solid #d9d0e6;
                border-radius: 10px;
                min-height: 34px;
                padding: 0 12px;
                color: #2f2a36;
            }
            QLineEdit:focus, QSpinBox:focus {
                border: 2px solid #6a2ca0;
            }
            QTableWidget {
                background: #ffffff;
                border: 1px solid #dfd6ea;
                border-radius: 12px;
                gridline-color: #ece5f4;
                alternate-background-color: #faf8fd;
                selection-background-color: #eee4f8;
                selection-color: #2f2a36;
            }
            QHeaderView::section {
                background: #f3edf9;
                color: #571c86;
                padding: 8px;
                border: none;
                border-bottom: 1px solid #ddd3ea;
                font: 700 11px 'Segoe UI';
            }
        """

    def _make_button(self, text: str, variant: str = "secondary") -> QPushButton:
        btn = QPushButton(text)
        btn.setCursor(Qt.PointingHandCursor)
        btn.setFixedHeight(38)
        if variant == "primary":
            bg, hover, fg, border = "#571c86", "#6b2aa3", "#ffffff", "none"
        elif variant == "success":
            bg, hover, fg, border = "#1f9d55", "#18854a", "#ffffff", "none"
        elif variant == "danger":
            bg, hover, fg, border = "#d93f3f", "#bf3535", "#ffffff", "none"
        else:
            bg, hover, fg, border = "#ffffff", "#faf7fd", "#571c86", "1px solid #d7cae7"

        btn.setStyleSheet(f"""
            QPushButton {{
                background: {bg};
                color: {fg};
                border: {border};
                border-radius: 19px;
                padding: 0 18px;
                font: 700 11px 'Segoe UI';
            }}
            QPushButton:hover {{ background: {hover}; }}
            QPushButton:pressed {{ background: #49176f; color: #ffffff; }}
            QPushButton:disabled {{
                background: #c8b8dc;
                color: #f4f0f8;
                border: none;
            }}
        """)
        return btn

    def _section_header(self, title: str, subtitle: str) -> QVBoxLayout:
        header = QVBoxLayout()
        header.setSpacing(4)
        title_lbl = QLabel(title)
        title_lbl.setObjectName("PageTitle")
        header.addWidget(title_lbl)
        sub_lbl = QLabel(subtitle)
        sub_lbl.setObjectName("PageSubTitle")
        sub_lbl.setWordWrap(True)
        header.addWidget(sub_lbl)
        return header

    # ======================================================================
    # COMMON STATE
    # ======================================================================
    def set_sku_meta(self, sku_meta: dict):
        self.sku_meta = dict(sku_meta or {})
        self.sku_meta.pop("machine_serial", None)
        self._apply_sku_meta_to_form()

    def _apply_sku_meta_to_form(self):
        if not self.wizard_widgets:
            return
        text_keys = [
            "sku_name",
            "tyre_name",
            "tyre_size",
            "tyre_outer_diameter",
            "tyre_rpm",
            "barcode",
            "barcode_pattern",
            "operator",
        ]
        for key in text_keys:
            widget = self.wizard_widgets.get(key)
            if widget is not None:
                widget.setText(str(self.sku_meta.get(key, "") or ""))
        for key, default in [
            ("recipe_number", 1),
            ("inspection_zones", 5),
            ("image_count_per_zone", 20),
            ("train_good_count", 10),
        ]:
            widget = self.wizard_widgets.get(key)
            if widget is not None:
                widget.setValue(_to_int(self.sku_meta.get(key), default))

    def _get_sku_name(self) -> str:
        for key in ("sku_name", "sku", "name", "pattern_name", "tyre_name"):
            value = self.sku_meta.get(key)
            if value:
                return str(value).strip()

        base_dir = os.path.join(self.media_path, "new_sku_images")
        if os.path.isdir(base_dir):
            folders = [name for name in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, name))]
            if len(folders) == 1:
                return folders[0]
        return "unknown_sku"

    def _preview_serial_order(self):
        if any(serial in self.latest_preview_paths for serial in self.camera_serial_order):
            return self.camera_serial_order
        return [str(i + 1) for i in range(len(self.labels))]

    def _ordered_preview_paths(self):
        paths = []
        for idx, serial in enumerate(self.camera_serial_order):
            raw_key = str(idx + 1)
            path = self.latest_preview_paths.get(serial) or self.latest_preview_paths.get(raw_key) or ""
            paths.append(path)
        while len(paths) < len(self.labels):
            paths.append("")
        return paths[:len(self.labels)]

    def load_raw_images_for_preview(self):
        if self.capture_in_progress or self.training_in_progress:
            return
        self.latest_preview_paths = {}
        preview_keys = self._preview_serial_order()
        if os.path.exists(self.raw_dir):
            image_files = [f for f in os.listdir(self.raw_dir) if f.lower().endswith(IMAGE_EXTS)]
            image_files.sort()
            for idx, key in enumerate(preview_keys):
                if idx < len(image_files):
                    image_path = os.path.join(self.raw_dir, image_files[idx])
                    if os.path.exists(image_path):
                        self.latest_preview_paths[key] = image_path
            if not self.latest_preview_paths:
                for file in image_files:
                    name_without_ext = os.path.splitext(file)[0]
                    if name_without_ext in preview_keys:
                        self.latest_preview_paths[name_without_ext] = os.path.join(self.raw_dir, file)
        self._update_preview_from_latest()
        if self.status_lbl is not None:
            if self.latest_preview_paths:
                self.status_lbl.setText(f"Loaded {len(self.latest_preview_paths)} images from raw folder")
            else:
                self.status_lbl.setText("No images found in raw folder")

    # ======================================================================
    # MAIN PAGE UI
    # ======================================================================
    def _tab_button_style(self, active: bool) -> str:
        if active:
            return """
                QPushButton {
                    background: transparent;
                    color: #571c86;
                    border: none;
                    border-bottom: 2px solid #571c86;
                    font: 700 11px 'Segoe UI';
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
            QPushButton:hover { color: #571c86; }
        """

    def _switch_tab(self, idx: int):
        if self.stack is None:
            return
        self.stack.setCurrentIndex(idx)
        for i, btn in enumerate(self.tab_buttons):
            btn.setStyleSheet(self._tab_button_style(i == idx))

    def _build_ui(self):
        self.setStyleSheet(self._page_stylesheet())

        root = QVBoxLayout(self)
        root.setContentsMargins(18, 10, 18, 12)
        root.setSpacing(12)

        nav_frame = QFrame()
        nav_frame.setFixedHeight(38)
        nav_frame.setStyleSheet("QFrame { background: transparent; border: none; }")
        nav_l = QHBoxLayout(nav_frame)
        nav_l.setContentsMargins(0, 0, 0, 0)
        nav_l.setSpacing(0)

        self.tab_buttons = []
        tab_names = ["SKU Setup", "Axis Teaching", "Capture", "Training", "Validation", "Save Recipe"]
        for idx, name in enumerate(tab_names):
            btn = QPushButton(name)
            btn.setCursor(Qt.PointingHandCursor)
            btn.setFixedHeight(34)
            btn.clicked.connect(lambda checked=False, i=idx: self._switch_tab(i))
            nav_l.addWidget(btn)
            self.tab_buttons.append(btn)

        nav_l.addStretch(1)
        version_lbl = QLabel("v1.0")
        version_lbl.setStyleSheet("font: 500 9px 'Segoe UI'; color: #b9b0c7; padding: 0 6px;")
        nav_l.addWidget(version_lbl)
        root.addWidget(nav_frame)

        self.stack = QStackedWidget()
        self.stack.setMinimumSize(0, 0)
        self.stack.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.wizard_page = QWidget()
        self.axis_teaching_page = QWidget()
        self.capture_page = QWidget()
        self.training_page = QWidget()
        self.validation_page = QWidget()
        self.recipe_page = QWidget()

        self._build_wizard_page()
        self._build_axis_teaching_page()
        self._build_capture_page()
        self._build_training_page()
        self._build_validation_page()
        self._build_recipe_page()

        self.stack.addWidget(self.wizard_page)
        self.stack.addWidget(self.axis_teaching_page)
        self.stack.addWidget(self.capture_page)
        self.stack.addWidget(self.training_page)
        self.stack.addWidget(self.validation_page)
        self.stack.addWidget(self.recipe_page)

        root.addWidget(self.stack, 1)
        self._switch_tab(TAB_SKU_SETUP)

    # ======================================================================
    # F-015 SKU SETUP
    # ======================================================================
    def _build_wizard_page(self):
        root = QVBoxLayout(self.wizard_page)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(12)

        card = QFrame()
        card.setObjectName("PageCard")
        lay = QVBoxLayout(card)
        lay.setContentsMargins(24, 20, 24, 20)
        lay.setSpacing(18)

        lay.addLayout(self._section_header(
            "New SKU Creation Wizard",
            "Create a new tyre SKU by entering tyre, barcode, operator and image capture configuration. Machine serial is intentionally removed.",
        ))

        form_card = QFrame()
        form_card.setObjectName("InnerCard")
        form_l = QVBoxLayout(form_card)
        form_l.setContentsMargins(18, 18, 18, 18)
        form_l.setSpacing(14)

        section = QLabel("SKU Details")
        section.setObjectName("SectionTitle")
        form_l.addWidget(section)

        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignRight)
        form.setFormAlignment(Qt.AlignTop)
        form.setHorizontalSpacing(20)
        form.setVerticalSpacing(14)

        sku_edit = QLineEdit()
        sku_edit.setPlaceholderText("Example: SKU_001")

        tyre_name_edit = QLineEdit()
        tyre_name_edit.setPlaceholderText("Example: Apollo Amazer 4G")

        size_edit = QLineEdit()
        size_edit.setPlaceholderText("Example: 195/65 R15")

        barcode_edit = QLineEdit()
        barcode_edit.setPlaceholderText("Enter actual barcode value")

        tyre_outer_diameter_edit = QLineEdit()
        tyre_outer_diameter_edit.setPlaceholderText("Example: 600")

        tyre_rpm_edit = QLineEdit()
        tyre_rpm_edit.setPlaceholderText("Example: 2.0")

        barcode_pattern_edit = QLineEdit()
        barcode_pattern_edit.setPlaceholderText("Example: APOLLO-* or regex pattern")

        operator_edit = QLineEdit()
        operator_edit.setPlaceholderText("Enter operator name")

        recipe_number_spin = QSpinBox()
        recipe_number_spin.setMinimum(1)
        recipe_number_spin.setMaximum(9999)
        recipe_number_spin.setValue(_to_int(self.sku_meta.get("recipe_number", 1), 1))

        zones_spin = QSpinBox()
        zones_spin.setMinimum(1)
        zones_spin.setMaximum(5)

        img_count_spin = QSpinBox()
        img_count_spin.setMinimum(2)
        img_count_spin.setMaximum(100)

        train_good_spin = QSpinBox()
        train_good_spin.setMinimum(1)
        train_good_spin.setMaximum(100)

        self.wizard_widgets = {
            "sku_name": sku_edit,
            "recipe_number": recipe_number_spin,
            "tyre_name": tyre_name_edit,
            "tyre_size": size_edit,
            "tyre_outer_diameter": tyre_outer_diameter_edit,
            "tyre_rpm": tyre_rpm_edit,
            "barcode": barcode_edit,
            "barcode_pattern": barcode_pattern_edit,
            "operator": operator_edit,
            "inspection_zones": zones_spin,
            "image_count_per_zone": img_count_spin,
            "train_good_count": train_good_spin,
        }
        self._apply_sku_meta_to_form()

        # Defaults if no meta was supplied
        if not self.sku_meta:
            zones_spin.setValue(_to_int(env_vars.get("NEW_SKU_DEFAULT_ZONE_COUNT", 5), 5))
            img_count_spin.setValue(_to_int(env_vars.get("NEW_SKU_DEFAULT_IMAGE_COUNT_PER_ZONE", 20), 20))
            train_good_spin.setValue(_to_int(env_vars.get("NEW_SKU_DEFAULT_TRAIN_GOOD_COUNT", 10), 10))

        form.addRow("SKU Name", sku_edit)
        form.addRow("Recipe Number", recipe_number_spin)
        form.addRow("Tyre Name", tyre_name_edit)
        form.addRow("Tyre Size", size_edit)
        form.addRow("Tyre Outer Diameter", tyre_outer_diameter_edit)
        form.addRow("Tyre RPM", tyre_rpm_edit)
        form.addRow("Barcode", barcode_edit)
        form.addRow("Barcode Pattern", barcode_pattern_edit)
        form.addRow("Operator", operator_edit)
        form.addRow("Inspection Zones", zones_spin)
        form.addRow("Images per Zone", img_count_spin)
        form.addRow("Train Good Count", train_good_spin)

        form_l.addLayout(form)
        lay.addWidget(form_card)

        hint = QLabel("Note: Machine serial field is removed. Recipe will store camera and laser axis positions separately.")
        hint.setObjectName("HintText")
        lay.addWidget(hint)

        btn_row = QHBoxLayout()
        btn_row.addStretch(1)

        next_btn = self._make_button("Next: Axis Teaching", "secondary")
        next_btn.clicked.connect(lambda: self._switch_tab(TAB_AXIS_TEACHING))

        save_setup_btn = self._make_button("Save SKU Setup", "primary")
        save_setup_btn.clicked.connect(self._save_sku_setup)

        btn_row.addWidget(next_btn)
        btn_row.addWidget(save_setup_btn)

        lay.addLayout(btn_row)

        root.addWidget(card)
        root.addStretch(1)

    def _save_sku_setup(self):
        sku_name = self.wizard_widgets["sku_name"].text().strip()
        recipe_number = int(self.wizard_widgets["recipe_number"].value())
        tyre_name = self.wizard_widgets["tyre_name"].text().strip()
        tyre_size = self.wizard_widgets["tyre_size"].text().strip()
        tyre_outer_diameter_raw = self.wizard_widgets["tyre_outer_diameter"].text().strip()
        tyre_rpm_raw = self.wizard_widgets["tyre_rpm"].text().strip()
        barcode = self.wizard_widgets["barcode"].text().strip()
        barcode_pattern = self.wizard_widgets["barcode_pattern"].text().strip()
        operator = self.wizard_widgets["operator"].text().strip()
        inspection_zones = int(self.wizard_widgets["inspection_zones"].value())
        image_count_per_zone = int(self.wizard_widgets["image_count_per_zone"].value())
        train_good_count = int(self.wizard_widgets["train_good_count"].value())

        if not sku_name:
            QMessageBox.warning(self, "SKU Setup", "SKU name is required.")
            return
        if train_good_count >= image_count_per_zone:
            QMessageBox.warning(self, "SKU Setup", "Train Good Count must be smaller than Images per Zone.")
            return
        tyre_outer_diameter = _to_float_or_none(tyre_outer_diameter_raw)
        tyre_rpm = _to_float_or_none(tyre_rpm_raw)

        if tyre_outer_diameter_raw and tyre_outer_diameter is None:
            QMessageBox.warning(self, "SKU Setup", "Tyre Outer Diameter must be a valid number.")
            return

        if tyre_rpm_raw and tyre_rpm is None:
            QMessageBox.warning(self, "SKU Setup", "Tyre RPM must be a valid number.")
            return
        existing_recipe = self.recipe_service.find_recipe_by_number(recipe_number)

        if existing_recipe:
            existing_sku = existing_recipe.get("sku_name", "UNKNOWN")
            existing_version = existing_recipe.get("version", "-")

            QMessageBox.warning(
                self,
                "Duplicate Recipe Number",
                (
                    f"Recipe number {recipe_number} already exists.\n\n"
                    f"Existing SKU: {existing_sku}\n"
                    f"Version: {existing_version}\n\n"
                    "Please use a different recipe number."
                )
            )
            return
        tyre_name = tyre_name or sku_name
        barcode = barcode or barcode_pattern
        operator = operator or "operator"

        self.sku_meta.update({
            "sku_name": sku_name,
            "recipe_number": recipe_number,
            "plc_recipe_number": recipe_number,
            "tyre_name": tyre_name,
            "tyre_size": tyre_size,
            "tyre_outer_diameter": tyre_outer_diameter,
            "tyre_rpm": tyre_rpm,
            "barcode": barcode,
            "barcode_pattern": barcode_pattern,
            "operator": operator,
            "inspection_zones": inspection_zones,
            "image_count_per_zone": image_count_per_zone,
            "train_good_count": train_good_count,
        })
        self.sku_meta.pop("machine_serial", None)
        self.recipe_doc["sku_meta"] = dict(self.sku_meta)

        try:
            clean_sku_meta = dict(self.sku_meta)
            clean_sku_meta.pop("machine_serial", None)

            self.recipe_service.new_sku_col.update_one(
                {"type": "sku_setup", "sku_name": sku_name},
                {
                    "$set": {
                        "type": "sku_setup",
                        "sku_name": sku_name,
                        "sku_meta": clean_sku_meta,
                        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    },
                    "$setOnInsert": {
                        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    },
                },
                upsert=True,
            )
        except Exception as e:
            QMessageBox.warning(self, "DB Warning", f"SKU setup saved in page but DB update failed:\n{e}")

        if self.status_lbl is not None:
            self.status_lbl.setText(f"SKU setup saved successfully: {sku_name}")
        QMessageBox.information(self, "SKU Setup", f"SKU setup saved successfully for {sku_name}.")
        self._switch_tab(TAB_AXIS_TEACHING)

    # ======================================================================
    # F-016 / F-045 AXIS TEACHING
    # ======================================================================
    def _build_axis_teaching_page(self):
        root = QVBoxLayout(self.axis_teaching_page)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(12)

        card = QFrame()
        card.setObjectName("PageCard")
        lay = QVBoxLayout(card)
        lay.setContentsMargins(18, 16, 18, 16)
        lay.setSpacing(12)

        lay.addLayout(self._section_header(
            "Teaching Mode — Recipe Target Capture",
            "Create recipe target values from live servo positions or manual software entry. "
            "One physical servo axis can be used by camera and laser targets separately.",
        ))

        hint = QLabel(
            "Production mode uses src/COMMON/recipe_tag_map.py as the master recipe tag map. "
            "DB74 live servo positions are read only. Recipe targets are saved to MongoDB and written to DB53."
        )
        hint.setObjectName("HintText")
        lay.addWidget(hint)

        mode_row = QHBoxLayout()

        mode_lbl = QLabel("Axis Entry Mode:")
        mode_lbl.setObjectName("SectionTitle")
        mode_row.addWidget(mode_lbl)

        self.axis_entry_mode_combo = QComboBox()
        self.axis_entry_mode_combo.addItems([
            "Capture From Live PLC",
            "Manual Entry From Software",
        ])
        self.axis_entry_mode_combo.setFixedHeight(34)
        self.axis_entry_mode_combo.setMinimumWidth(240)
        self.axis_entry_mode_combo.currentIndexChanged.connect(self._on_axis_entry_mode_changed)
        mode_row.addWidget(self.axis_entry_mode_combo)

        self.apply_manual_axis_btn = self._make_button("Apply Manual Targets", "primary")
        self.apply_manual_axis_btn.clicked.connect(self._apply_manual_axis_targets_from_table)
        self.apply_manual_axis_btn.setEnabled(False)
        mode_row.addWidget(self.apply_manual_axis_btn)

        mode_row.addStretch(1)
        lay.addLayout(mode_row)

        self.axis_table = QTableWidget()
        self.axis_table.setColumnCount(11)
        self.axis_table.setHorizontalHeaderLabels([
            "Group",
            "Axis",
            "Position",
            "Target Key",
            "DB53 Address",
            "Physical Axis",
            "Axis Name",
            "Servo IP",
            "Current Axis Position",
            "Target Value",
            "Delta",
        ])
        self.axis_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.axis_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.axis_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.axis_table.setAlternatingRowColors(True)
        lay.addWidget(self.axis_table, 1)

        btn_row = QHBoxLayout()

        refresh_btn = self._make_button("Refresh Live Axis", "secondary")
        refresh_btn.clicked.connect(self._refresh_axis_table)

        # capture_all_btn = self._make_button("Capture All Live Targets", "primary")
        # capture_all_btn.clicked.connect(lambda: self._capture_axis_group("all"))
        capture_selected_btn = self._make_button("Capture Selected Target", "primary")
        capture_selected_btn.clicked.connect(self._capture_selected_axis_target)

        # capture_camera_btn = self._make_button("Capture Machine/Camera Targets", "primary")
        # capture_camera_btn.clicked.connect(lambda: self._capture_axis_group("camera"))

        # capture_laser_btn = self._make_button("Capture Laser Targets", "primary")
        # capture_laser_btn.clicked.connect(lambda: self._capture_axis_group("laser"))

        next_btn = self._make_button("Next: Capture Images", "secondary")
        next_btn.clicked.connect(lambda: self._switch_tab(TAB_CAPTURE))

        btn_row.addWidget(refresh_btn)
        btn_row.addWidget(capture_selected_btn)
        # btn_row.addWidget(capture_all_btn)
        # btn_row.addWidget(capture_camera_btn)
        # btn_row.addWidget(capture_laser_btn)
        btn_row.addStretch(1)
        btn_row.addWidget(next_btn)
        lay.addLayout(btn_row)

        root.addWidget(card)

        # QTimer.singleShot(200, self._refresh_axis_table)


    def _on_axis_entry_mode_changed(self):
        if self.axis_entry_mode_combo is None:
            return

        text = self.axis_entry_mode_combo.currentText().strip().lower()

        if "manual" in text:
            self.axis_entry_mode = "manual"
            if self.apply_manual_axis_btn is not None:
                self.apply_manual_axis_btn.setEnabled(True)

            if self.axis_table is not None:
                self.axis_table.setEditTriggers(
                    QTableWidget.DoubleClicked |
                    QTableWidget.EditKeyPressed |
                    QTableWidget.AnyKeyPressed
                )

            if self.status_lbl is not None:
                self.status_lbl.setText(
                    "Manual Entry Mode: edit Camera Target / Laser Target columns, then click Apply Manual Targets."
                )

        else:
            self.axis_entry_mode = "capture"
            if self.apply_manual_axis_btn is not None:
                self.apply_manual_axis_btn.setEnabled(False)

            if self.axis_table is not None:
                self.axis_table.setEditTriggers(QTableWidget.NoEditTriggers)

            if self.status_lbl is not None:
                self.status_lbl.setText(
                    "Capture Mode: move axis using PLC/HMI, refresh live axis, then capture targets."
                )

        self._refresh_axis_table()

    def _make_recipe_target_doc(self, cfg: Dict[str, Any], value, source: str) -> Dict[str, Any]:
        axis_id = int(cfg.get("axis_id", 0) or 0)
        axis_key = cfg.get("axis_key") or (f"axis_{axis_id:02d}" if axis_id > 0 else "")

        return {
            "target_key": cfg.get("target_key", ""),
            "legacy_key": cfg.get("legacy_key"),
            "target_index": cfg.get("target_index"),
            "group": str(cfg.get("group", "")).upper(),
            "position": cfg.get("position", ""),

            "axis_id": axis_id,
            "axis_key": axis_key,
            "axis_name": cfg.get("axis_name", ""),
            "axis_ip": cfg.get("axis_ip", ""),

            "target_name": cfg.get("target_name", ""),
            "value": None if value is None or value == "" else float(value),

            # PLC DB53 write address
            "write_db": cfg.get("write_db"),
            "write_byte": cfg.get("write_byte"),
            "type": cfg.get("type", "REAL"),

            # DB75 reference for Axis Status / debugging only
            "db75_db": cfg.get("db75_db"),
            "db75_byte": cfg.get("db75_byte"),
            "db75_type": cfg.get("db75_type", "REAL"),

            "captured_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "source": source,
        }


    def _sync_legacy_axis_targets_from_recipe_targets(self):
        """
        Keep old fields for backward compatibility:
            camera_axis_targets
            laser_axis_targets

        New production field:
            recipe_axis_targets

        Old fields are no longer the main production source.
        """
        recipe_targets = self.recipe_doc.get("recipe_axis_targets", {}) or {}

        camera_targets = {}
        laser_targets = {}

        for target_key, target in recipe_targets.items():
            group = str(target.get("group", "")).upper()

            legacy_item = {
                "target_key": target_key,
                "axis_id": target.get("axis_id"),
                "axis_key": target.get("axis_key"),
                "name": target.get("target_name") or target.get("axis_name"),
                "axis_name": target.get("axis_name"),
                "axis_ip": target.get("axis_ip"),
                "value": target.get("value"),
                "captured_at": target.get("captured_at"),
                "source": target.get("source"),
                "write_db": target.get("write_db"),
                "write_byte": target.get("write_byte"),
                "type": target.get("type", "REAL"),
            }

            if group in ("MACHINE", "CAMERA"):
                camera_targets[target_key] = legacy_item
            elif group == "LASER":
                laser_targets[target_key] = legacy_item

        self.recipe_doc["camera_axis_targets"] = camera_targets
        self.recipe_doc["laser_axis_targets"] = laser_targets

    def _refresh_axis_table(self):
        if self.axis_table is None:
            return

        try:
            positions = self.recipe_service.read_current_axis_positions()
            target_configs = self.recipe_service.get_recipe_target_configs()
        except Exception as e:
            self.axis_table.setRowCount(1)
            self.axis_table.setColumnCount(2)
            self.axis_table.setHorizontalHeaderLabels(["ERROR", "Message"])
            self.axis_table.setItem(0, 0, QTableWidgetItem("ERROR"))
            self.axis_table.setItem(0, 1, QTableWidgetItem(str(e)))
            return

        self.axis_table.setColumnCount(11)
        self.axis_table.setHorizontalHeaderLabels([
            "Group",
            "Axis",
            "Position",
            "Target Key",
            "DB53 Address",
            "Physical Axis",
            "Axis Name",
            "Servo IP",
            "Current Axis Position",
            "Target Value",
            "Delta",
        ])

        recipe_targets = self.recipe_doc.get("recipe_axis_targets", {}) or {}

        self.axis_table.setRowCount(len(target_configs))

        for row, cfg in enumerate(target_configs):
            target_key = cfg.get("target_key", "")
            group = str(cfg.get("group", "")).upper()

            axis_id = int(cfg.get("axis_id", row + 1) or row + 1)
            axis_key = cfg.get("axis_key") or f"axis_{axis_id:02d}"

            info = positions.get(axis_key, {}) or {}
            live_value = info.get("value")

            saved_target = recipe_targets.get(target_key, {}) or {}
            target_value = saved_target.get("value", "")

            delta = ""
            try:
                if live_value is not None and target_value != "":
                    delta = f"{float(live_value) - float(target_value):.3f}"
            except Exception:
                delta = ""

            db_no = cfg.get("write_db", "")
            write_byte = cfg.get("write_byte", "")
            db53_address = ""
            if db_no not in ("", None) and write_byte not in ("", None, -1):
                db53_address = f"DB{db_no}.DBD{write_byte}"

            values = [
                group,
                str(cfg.get("target_name", "")),
                str(cfg.get("position", "")),
                target_key,
                db53_address,
                axis_key,
                str(cfg.get("axis_name", "")),
                str(cfg.get("axis_ip", "")),
                "" if live_value is None else f"{float(live_value):.3f}",
                "" if target_value == "" or target_value is None else f"{float(target_value):.3f}",
                delta,
            ]

            for col, value in enumerate(values):
                item = QTableWidgetItem(str(value))
                item.setTextAlignment(Qt.AlignCenter)

                # In manual mode, allow editing only Target Value column.
                editable = False
                if self.axis_entry_mode == "manual" and col == 9:
                    editable = True

                if editable:
                    item.setFlags(item.flags() | Qt.ItemIsEditable)
                else:
                    item.setFlags(item.flags() & ~Qt.ItemIsEditable)

                self.axis_table.setItem(row, col, item)

    def _capture_axis_group(self, group: str):
        """
        Capture recipe target values from live PLC servo positions.

        group:
            all     -> capture all RECIPE_TARGET rows
            camera  -> capture MACHINE + CAMERA target rows
            laser   -> capture LASER target rows
        """
        try:
            positions = self.recipe_service.read_current_axis_positions()
            target_configs = self.recipe_service.get_recipe_target_configs()
        except Exception as e:
            QMessageBox.critical(self, "Axis Capture Error", str(e))
            return

        wanted_group = str(group or "all").strip().lower()

        existing = dict(self.recipe_doc.get("recipe_axis_targets", {}) or {})
        captured_count = 0

        for cfg in target_configs:
            cfg_group = str(cfg.get("group", "")).upper()

            if wanted_group == "camera":
                if cfg_group not in ("MACHINE", "CAMERA"):
                    continue
            elif wanted_group == "laser":
                if cfg_group != "LASER":
                    continue
            elif wanted_group == "all":
                pass
            else:
                continue

            axis_id = int(cfg.get("axis_id", 0) or 0)
            axis_key = cfg.get("axis_key") or f"axis_{axis_id:02d}"

            info = positions.get(axis_key)
            if not info:
                continue

            live_value = info.get("value")
            if live_value is None:
                continue

            target_key = cfg.get("target_key", "")
            if not target_key:
                continue

            existing[target_key] = self._make_recipe_target_doc(
                cfg=cfg,
                value=live_value,
                source="PLC_LIVE_CAPTURE",
            )
            captured_count += 1

        self.recipe_doc["recipe_axis_targets"] = existing
        self._sync_legacy_axis_targets_from_recipe_targets()

        self._refresh_axis_table()

        title = {
            "all": "All Recipe Targets",
            "camera": "Machine/Camera Recipe Targets",
            "laser": "Laser Recipe Targets",
        }.get(wanted_group, "Recipe Targets")

        QMessageBox.information(
            self,
            title,
            f"{captured_count} target values captured successfully."
        )
    
    def _capture_selected_axis_target(self):
        """
        Capture only the selected recipe target row from current live PLC position.

        This is the correct method for HOME / WORK1 / WORK2 / WORK3 teaching,
        because one physical axis has only one live position at a time.
        """
        if self.axis_table is None:
            return

        selected_rows = self.axis_table.selectionModel().selectedRows()
        if not selected_rows:
            QMessageBox.warning(
                self,
                "Capture Selected Target",
                "Please select one recipe target row first."
            )
            return

        row = selected_rows[0].row()

        target_key_item = self.axis_table.item(row, 3)
        if target_key_item is None:
            QMessageBox.warning(
                self,
                "Capture Selected Target",
                "Selected row does not have a target key."
            )
            return

        target_key = target_key_item.text().strip()
        if not target_key:
            QMessageBox.warning(
                self,
                "Capture Selected Target",
                "Selected row target key is empty."
            )
            return

        try:
            positions = self.recipe_service.read_current_axis_positions()
            target_cfg_map = self.recipe_service.get_recipe_target_config_map()
        except Exception as e:
            QMessageBox.critical(self, "Axis Capture Error", str(e))
            return

        cfg = target_cfg_map.get(target_key)
        if not cfg:
            QMessageBox.warning(
                self,
                "Capture Selected Target",
                f"Target config not found for: {target_key}"
            )
            return

        axis_id = int(cfg.get("axis_id", 0) or 0)
        axis_key = cfg.get("axis_key") or f"axis_{axis_id:02d}"

        info = positions.get(axis_key)
        if not info:
            QMessageBox.warning(
                self,
                "Capture Selected Target",
                f"Live position not found for {axis_key}."
            )
            return

        live_value = info.get("value")
        if live_value is None:
            QMessageBox.warning(
                self,
                "Capture Selected Target",
                f"Live value is empty for {axis_key}."
            )
            return

        existing = dict(self.recipe_doc.get("recipe_axis_targets", {}) or {})
        existing[target_key] = self._make_recipe_target_doc(
            cfg=cfg,
            value=live_value,
            source="PLC_SELECTED_ROW_CAPTURE",
        )

        self.recipe_doc["recipe_axis_targets"] = existing
        self._sync_legacy_axis_targets_from_recipe_targets()
        self._refresh_axis_table()

        QMessageBox.information(
            self,
            "Capture Selected Target",
            f"Captured {target_key} = {float(live_value):.3f}"
        )
    def _apply_manual_axis_targets_from_table(self, silent=False):
        """
        Apply manually typed target values from the Axis Teaching table.

        Only column 7 = Target Value is editable in manual mode.
        """
        if self.axis_table is None:
            return False

        target_cfg_map = self.recipe_service.get_recipe_target_config_map()

        recipe_targets = dict(self.recipe_doc.get("recipe_axis_targets", {}) or {})

        for row in range(self.axis_table.rowCount()):
            group_item = self.axis_table.item(row, 0)
            target_key_item = self.axis_table.item(row, 3)
            target_value_item = self.axis_table.item(row, 9)

            if target_key_item is None:
                continue

            target_key = target_key_item.text().strip()
            if not target_key:
                continue

            raw_value = target_value_item.text().strip() if target_value_item else ""

            # Blank value means not entered yet.
            if raw_value == "":
                continue

            try:
                value = float(raw_value)
            except Exception:
                if not silent:
                    QMessageBox.warning(
                        self,
                        "Manual Axis Entry",
                        f"Invalid target value for {target_key}: {raw_value}"
                    )
                return False

            cfg = target_cfg_map.get(target_key)
            if not cfg:
                if not silent:
                    QMessageBox.warning(
                        self,
                        "Manual Axis Entry",
                        f"Target config not found for: {target_key}"
                    )
                return False

            recipe_targets[target_key] = self._make_recipe_target_doc(
                cfg=cfg,
                value=value,
                source="MANUAL_ENTRY",
            )

        if recipe_targets:
            self.recipe_doc["recipe_axis_targets"] = recipe_targets
            self._sync_legacy_axis_targets_from_recipe_targets()

        self._refresh_axis_table()

        if not silent:
            machine_camera_count = sum(
                1 for v in recipe_targets.values()
                if str(v.get("group", "")).upper() in ("MACHINE", "CAMERA")
            )
            laser_count = sum(
                1 for v in recipe_targets.values()
                if str(v.get("group", "")).upper() == "LASER"
            )

            QMessageBox.information(
                self,
                "Manual Recipe Targets Applied",
                f"Total targets: {len(recipe_targets)}\n"
                f"Machine/Camera targets: {machine_camera_count}\n"
                f"Laser targets: {laser_count}"
            )

        return True

    # ======================================================================
    # F-017 IMAGE CAPTURE
    # ======================================================================
    def _build_capture_page(self):
        root = QVBoxLayout(self.capture_page)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(12)

        main_card = QFrame()
        main_card.setObjectName("PageCard")
        main_l = QVBoxLayout(main_card)
        main_l.setContentsMargins(18, 16, 18, 16)
        main_l.setSpacing(14)

        header_row = QHBoxLayout()
        header_left = QVBoxLayout()
        title_lbl = QLabel("New SKU Image Capture")
        title_lbl.setObjectName("PageTitle")
        header_left.addWidget(title_lbl)
        subtitle_lbl = QLabel("Capture and verify all tyre views before starting training.")
        subtitle_lbl.setObjectName("PageSubTitle")
        header_left.addWidget(subtitle_lbl)
        header_row.addLayout(header_left)
        header_row.addStretch(1)
        badge_lbl = QLabel(f"{len(self.labels)} Cameras")
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

        for i, label_name in enumerate(self.labels):
            card = QFrame()
            card.setObjectName("InnerCard")
            card.setFixedSize(250, 545)
            card.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
            card_l = QVBoxLayout(card)
            card_l.setContentsMargins(12, 12, 12, 12)
            card_l.setSpacing(10)

            top_row = QHBoxLayout()
            title = QLabel(label_name.title())
            title.setObjectName("SectionTitle")
            top_row.addWidget(title)
            top_row.addStretch(1)
            click_lbl = QLabel("Click to zoom")
            click_lbl.setObjectName("HintText")
            top_row.addWidget(click_lbl)
            card_l.addLayout(top_row)

            image_shell = QFrame()
            image_shell.setStyleSheet("""
                QFrame {
                    background: #f7f4fb;
                    border: 1px solid #e9e1f1;
                    border-radius: 14px;
                }
            """)
            image_shell.setFixedSize(226, 454)
            image_shell_l = QVBoxLayout(image_shell)
            image_shell_l.setContentsMargins(8, 8, 8, 8)
            img = AspectImageLabel(title=label_name.title())
            image_shell_l.addWidget(img, 0, Qt.AlignCenter)
            card_l.addWidget(image_shell, 1)

            footer_lbl = QLabel("Latest preview")
            footer_lbl.setAlignment(Qt.AlignCenter)
            footer_lbl.setObjectName("HintText")
            card_l.addWidget(footer_lbl)

            self.img_labels.append(img)
            preview_grid.addWidget(card, 0, i, Qt.AlignTop | Qt.AlignHCenter)

        for col in range(len(self.labels)):
            preview_grid.setColumnStretch(col, 1)
        main_l.addLayout(preview_grid)

        action_bar = QFrame()
        action_bar.setObjectName("ActionBar")
        action_bar.setFixedHeight(62)
        action_l = QHBoxLayout(action_bar)
        action_l.setContentsMargins(14, 10, 14, 10)
        action_l.setSpacing(10)

        self.capture_btn = self._make_button("Start Capture", "primary")
        self.capture_btn.clicked.connect(self.confirm_and_start_capture)
        self.training_btn = self._make_button("Start Training", "secondary")
        self.training_btn.clicked.connect(self.confirm_and_start_training)
        self.refresh_btn = self._make_button("Refresh Preview", "secondary")
        self.refresh_btn.clicked.connect(self.refresh_preview_with_raw_load)
        self.close_btn = self._make_button("Close", "secondary")
        self.close_btn.clicked.connect(self.close_page)

        action_l.addWidget(self.capture_btn)
        action_l.addWidget(self.training_btn)
        action_l.addWidget(self.refresh_btn)
        action_l.addStretch(1)
        action_l.addWidget(self.close_btn)
        main_l.addWidget(action_bar)

        status_card = QFrame()
        status_card.setObjectName("StatusCard")
        status_card.setFixedHeight(66)
        status_l = QVBoxLayout(status_card)
        status_l.setContentsMargins(14, 10, 14, 10)
        status_title = QLabel("Status")
        status_title.setObjectName("SectionTitle")
        status_l.addWidget(status_title)
        self.status_lbl = QLabel("Ready")
        self.status_lbl.setObjectName("HintText")
        self.status_lbl.setWordWrap(True)
        status_l.addWidget(self.status_lbl)
        main_l.addWidget(status_card)

        root.addWidget(main_card, 1)

    def _set_controls_enabled(self, enabled: bool):
        for btn in [self.capture_btn, self.training_btn, self.refresh_btn, self.close_btn]:
            if btn is not None:
                btn.setEnabled(enabled)
        if self.tab_buttons:
            current_idx = self.stack.currentIndex() if self.stack else -1
            for idx, tab_btn in enumerate(self.tab_buttons):
                tab_btn.setEnabled(True if enabled else idx == current_idx)

    def refresh_preview_only(self):
        if self.capture_in_progress or self.training_in_progress:
            return
        if not self.latest_preview_paths:
            self.load_raw_images_for_preview()
        preview_paths = [self.latest_preview_paths.get(key, "") for key in self._preview_serial_order()]
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
        if self.status_lbl is not None:
            if self.latest_preview_paths:
                self.status_lbl.setText(f"Loaded {len(self.latest_preview_paths)} images from raw folder")
            else:
                self.status_lbl.setText("No images found in raw folder")

    def _update_preview_from_latest(self):
        preview_paths = self._ordered_preview_paths()
        while len(preview_paths) < len(self.labels):
            preview_paths.append("")
        for i in range(len(self.labels)):
            if i < len(self.img_labels):
                self.img_labels[i].set_image_path(preview_paths[i])

    def _get_capture_plan(self):
        total = _to_int(self.sku_meta.get("image_count_per_zone", env_vars.get("NEW_SKU_DEFAULT_IMAGE_COUNT_PER_ZONE", 20)), 20)
        good_count = _to_int(self.sku_meta.get("train_good_count", env_vars.get("NEW_SKU_DEFAULT_TRAIN_GOOD_COUNT", 10)), 10)
        expected = _to_int(self.sku_meta.get("inspection_zones", env_vars.get("NEW_SKU_DEFAULT_ZONE_COUNT", len(CAMERA_SERIAL_ORDER) or 5)), len(CAMERA_SERIAL_ORDER) or 5)
        if total < 2:
            total = 20
        if good_count < 1:
            good_count = 10
        if good_count >= total:
            good_count = max(1, total // 2)
        if expected < 1:
            expected = len(CAMERA_SERIAL_ORDER) or 5
        return total, good_count, expected

    def confirm_and_start_capture(self):
        if self.capture_in_progress or self.training_in_progress:
            return

        total, good_count, expected = self._get_capture_plan()
        sku_name = self._get_sku_name()

        msg = (
            f"Capture {total} images per camera for SKU: {sku_name}\n\n"
            f"Save path:\n"
            f"media/new_sku_images/{_safe_name(sku_name)}/<camera_serial>/\n\n"
            "After placing the tyre, click OK.\n"
            "Then software trigger will capture images from connected cameras."
        )

        reply = QMessageBox.question(
            self,
            "Start New SKU Capture",
            msg,
            QMessageBox.Ok | QMessageBox.Cancel,
            QMessageBox.Cancel,
        )

        if reply == QMessageBox.Ok:
            self.start_capture()


    def start_capture(self):
        if self.capture_in_progress or self.training_in_progress:
            return

        if capture_new_sku_images is None:
            QMessageBox.critical(
                self,
                "Capture Error",
                "capture_new_sku_images could not be imported.\n"
                "Check src/camera/new_sku_software_capture.py",
            )
            return

        if self.multi_camera_manager is None:
            QMessageBox.critical(
                self,
                "Camera Error",
                "No connected camera manager found.\n\n"
                "Please run Test Mode first and connect cameras."
            )
            return

        self.capture_in_progress = True
        self._set_controls_enabled(False)

        if self.preview_timer:
            self.preview_timer.stop()

        self._switch_tab(TAB_CAPTURE)

        images_per_camera, good_folder_count, expected_cameras = self._get_capture_plan()
        sku_name = _safe_name(self._get_sku_name())

        self.latest_preview_paths = {}

        if self.status_lbl is not None:
            self.status_lbl.setText(
                f"Starting software capture | SKU={sku_name} | "
                f"Images/camera={images_per_camera} | Train good={good_folder_count}"
            )

        self.capture_worker = CaptureWorker(
            sku_name=sku_name,
            media_path=self.media_path,
            images_per_camera=images_per_camera,
            train_good_count=good_folder_count,
            multi_camera_manager=self.multi_camera_manager,
            sku_meta=self.sku_meta,
            meta_collection=self.meta_collection,
            gridfs_bucket=self.gridfs_bucket,
            parent=self,
        )

        self.capture_worker.status_signal.connect(self._on_capture_status)
        self.capture_worker.finished_signal.connect(self._on_capture_finished)
        self.capture_worker.error_signal.connect(self._on_capture_error)

        self.capture_worker.start()

    # ======================================================================
    # F-018 TRAINING
    # ======================================================================
    def _build_training_page(self):
        root = QVBoxLayout(self.training_page)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(12)

        main_card = QFrame()
        main_card.setObjectName("PageCard")
        main_l = QVBoxLayout(main_card)
        main_l.setContentsMargins(18, 16, 18, 16)
        main_l.setSpacing(14)

        header_row = QHBoxLayout()
        header_left = QVBoxLayout()
        title_lbl = QLabel("Model Training")
        title_lbl.setObjectName("PageTitle")
        header_left.addWidget(title_lbl)
        subtitle_lbl = QLabel("Monitor dataset preparation, epoch progress, and final result for each camera pipeline.")
        subtitle_lbl.setObjectName("PageSubTitle")
        subtitle_lbl.setWordWrap(True)
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
        top_card.setObjectName("InnerCard")
        top_l = QVBoxLayout(top_card)
        top_l.setContentsMargins(16, 14, 16, 14)
        self.training_status_lbl = QLabel("Training status: Waiting")
        self.training_status_lbl.setObjectName("SectionTitle")
        top_l.addWidget(self.training_status_lbl)
        self.training_summary_lbl = QLabel("No training started yet.")
        self.training_summary_lbl.setObjectName("PageSubTitle")
        self.training_summary_lbl.setWordWrap(True)
        top_l.addWidget(self.training_summary_lbl)
        self.training_current_action_lbl = QLabel("Current action: Waiting")
        self.training_current_action_lbl.setObjectName("HintText")
        self.training_current_action_lbl.setWordWrap(True)
        top_l.addWidget(self.training_current_action_lbl)
        main_l.addWidget(top_card)

        progress_card = QFrame()
        progress_card.setObjectName("InnerCard")
        progress_l = QVBoxLayout(progress_card)
        progress_l.setContentsMargins(16, 14, 16, 14)
        prog_title = QLabel("Overall Progress")
        prog_title.setObjectName("SectionTitle")
        progress_l.addWidget(prog_title)
        prog_row = QHBoxLayout()
        self.training_progress = QProgressBar()
        self.training_progress.setRange(0, 100)
        self.training_progress.setValue(0)
        self.training_progress.setTextVisible(False)
        self.training_progress.setFixedHeight(12)
        self.training_progress.setStyleSheet("""
            QProgressBar { background:#eee9f5; border-radius:6px; border:none; }
            QProgressBar::chunk { background:#571c86; border-radius:6px; }
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
        main_l.addWidget(progress_card)

        grid = QGridLayout()
        grid.setHorizontalSpacing(16)
        grid.setVerticalSpacing(16)
        self.camera_result_labels = {}
        self.camera_status_boxes = {}

        serials = list(CAMERA_SERIAL_ORDER)
        if not serials:
            empty = QLabel("No camera serials configured in .env. Add CAM_SIDEWALL1_SERIAL, CAM_SIDEWALL2_SERIAL, CAM_INNERWALL_SERIAL, CAM_TREAD_SERIAL, CAM_BEAD_SERIAL.")
            empty.setObjectName("InfoBox")
            main_l.addWidget(empty)
        else:
            for idx, serial in enumerate(serials):
                card = QFrame()
                card.setObjectName("InnerCard")
                card.setMinimumHeight(150)
                card.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
                cl = QVBoxLayout(card)
                cl.setContentsMargins(14, 14, 14, 14)
                top_row = QHBoxLayout()
                title = QLabel(self.serial_to_title.get(serial, serial))
                title.setObjectName("SectionTitle")
                top_row.addWidget(title)
                top_row.addStretch(1)
                pipe_name = CAMERA_PIPELINE_MAP.get(serial, "not configured")
                pipe_badge = QLabel(pipe_name)
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
                serial_lbl.setObjectName("HintText")
                cl.addWidget(serial_lbl)

                status_box = QFrame()
                status_box.setStyleSheet("QFrame { background:#f7f3fb; border:1px solid #e8def2; border-radius:12px; }")
                status_box_l = QVBoxLayout(status_box)
                status_box_l.setContentsMargins(12, 10, 12, 10)
                status_title = QLabel("Pipeline Status")
                status_title.setObjectName("SectionTitle")
                status_box_l.addWidget(status_title)
                result_lbl = QLabel("Status: Not started")
                result_lbl.setWordWrap(True)
                result_lbl.setMinimumHeight(48)
                result_lbl.setStyleSheet("font: 600 11px 'Segoe UI'; color:#6f6a7a;")
                status_box_l.addWidget(result_lbl)
                cl.addWidget(status_box, 1)

                self.camera_result_labels[serial] = result_lbl
                self.camera_status_boxes[serial] = status_box
                grid.addWidget(card, idx // 2, idx % 2)
            grid.setColumnStretch(0, 1)
            grid.setColumnStretch(1, 1)
            main_l.addLayout(grid, 1)

        root.addWidget(main_card, 1)

    def _status_theme(self, state: str):
        themes = {
            "waiting": ("#6f6a7a", "#f7f3fb", "#e8def2"),
            "prep": ("#b16e2c", "#fff7ee", "#efd8b3"),
            "queued": ("#1f6390", "#eef7ff", "#bfdcf2"),
            "training": ("#1f6390", "#eef4ff", "#c9daf8"),
            "done": ("#2a6e2f", "#eefaf0", "#cbe8cf"),
            "failed": ("#bd3b2b", "#fff1ef", "#f3c7c1"),
            "skipped": ("#9a92ae", "#f5f3f8", "#dfd9e8"),
            "unknown": ("#333333", "#f7f3fb", "#e8def2"),
        }
        return themes.get(state, themes["unknown"])

    def _set_camera_status(self, serial: str, text: str, color: str = "#333", state: str = "unknown"):
        lbl = self.camera_result_labels.get(serial)
        box = self.camera_status_boxes.get(serial)
        if lbl is not None:
            lbl.setText(text)
            lbl.setStyleSheet(f"font: 600 11px 'Segoe UI'; color: {color};")
        fg, bg, border = self._status_theme(state)
        if box is not None:
            box.setStyleSheet(f"QFrame {{ background: {bg}; border: 1px solid {border}; border-radius: 12px; }}")
        self.serial_status_state[serial] = state
        self._refresh_training_summary_counts()

    def _refresh_training_summary_counts(self):
        if not self.enabled_training_serials:
            return
        counts = {k: 0 for k in ["prep", "queued", "training", "done", "failed", "skipped", "waiting"]}
        for serial in self.enabled_training_serials:
            state = self.serial_status_state.get(serial, "waiting")
            counts[state if state in counts else "waiting"] += 1
        parts = []
        if counts["prep"]: parts.append(f"{counts['prep']} preparing")
        if counts["queued"]: parts.append(f"{counts['queued']} waiting for GPU")
        if counts["training"]: parts.append(f"{counts['training']} training")
        if counts["done"]: parts.append(f"{counts['done']} done")
        if counts["failed"]: parts.append(f"{counts['failed']} failed")
        if counts["skipped"]: parts.append(f"{counts['skipped']} skipped")
        if counts["waiting"]: parts.append(f"{counts['waiting']} waiting")
        if self.training_summary_lbl is not None and self.training_in_progress:
            self.training_summary_lbl.setText(", ".join(parts) if parts else "Training is in progress.")

    def _compact_training_msg(self, msg: str, max_len: int = 120) -> str:
        msg = (msg or "").strip().replace("\n", " ")
        return msg if len(msg) <= max_len else msg[:max_len - 3] + "..."

    def _reset_training_cards(self):
        self.serial_status_state = {}
        for serial in list(CAMERA_SERIAL_ORDER):
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
            self.training_progress.setValue(pct)
        if self.training_percent_lbl is not None:
            self.training_percent_lbl.setText(f"{pct}%")

    def _update_training_card_from_log(self, msg: str):
        compact_msg = self._compact_training_msg(msg)
        if self.training_current_action_lbl is not None:
            self.training_current_action_lbl.setText(f"Current action: {compact_msg}")

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
            self._set_camera_status(serial, f"Status: Prep failed\n{m.group(2)[:70]}", "#bd3b2b", "failed")
            self._set_serial_progress(serial, 1.0)
            return
        m = re.search(r"\[TRAIN\]\s+serial=(\d+)", msg)
        if m:
            serial = m.group(1)
            self.current_gpu_training_serial = serial
            self._set_camera_status(serial, "Status: Training started on GPU...", "#1f6390", "training")
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
            self._set_camera_status(serial, f"Status: Failed\n{m.group(2)[:70]}", "#bd3b2b", "failed")
            self._set_serial_progress(serial, 1.0)
            if self.current_gpu_training_serial == serial:
                self.current_gpu_training_serial = None
            return
        m = re.search(r"\[SKIP\]\s+serial=(\d+)\s+\|\s+(.*)", msg)
        if m:
            serial = m.group(1)
            self._set_camera_status(serial, f"Status: Skipped\n{m.group(2)[:70]}", "#9a92ae", "skipped")
            self._set_serial_progress(serial, 1.0)

    def confirm_and_start_training(self):
        if self.capture_in_progress or self.training_in_progress:
            return
        sku_name = self._get_sku_name()
        sku_folder = _safe_name(sku_name)
        sku_root = os.path.join(self.media_path, "new_sku_images", sku_folder)
        if not os.path.exists(sku_root):
            QMessageBox.warning(self, "Training", f"SKU folder not found:\n{sku_root}\n\nCapture first.")
            return
        reply = QMessageBox.question(
            self,
            "Start Training",
            "Start VIT training now?\n\nTraining will use train/good images from each camera serial folder and reference images from each serial root.",
            QMessageBox.Ok | QMessageBox.Cancel,
            QMessageBox.Cancel,
        )
        if reply == QMessageBox.Ok:
            self.start_training()

    def start_training(self):
        if self.capture_in_progress or self.training_in_progress:
            return
        self.training_in_progress = True
        self._set_controls_enabled(False)
        if self.preview_timer:
            self.preview_timer.stop()
        sku_name = self._get_sku_name()
        self._switch_tab(TAB_TRAINING)
        self._reset_training_cards()
        self._reset_training_progress()
        if self.training_status_lbl is not None:
            self.training_status_lbl.setText(f"Training status: Starting for SKU={sku_name}")
        if self.training_summary_lbl is not None:
            self.training_summary_lbl.setText("Dataset preparation and training are in progress.")
        if self.training_current_action_lbl is not None:
            self.training_current_action_lbl.setText("Current action: Waiting for training to start...")
        if self.status_lbl is not None:
            self.status_lbl.setText(f"Training running for SKU={sku_name} ...")

        self.training_worker = TrainingWorker(
            media_path=self.media_path,
            sku_name=sku_name,
            serial_pipeline_map=CAMERA_PIPELINE_MAP,
            vit_training_root=VIT_TRAINING_ROOT,
            yolo_r_path=YOLO_R_PATH,
            device="cuda",
            rebuild_dataset=True,
            parent=self,
        )
        self.training_worker.status_signal.connect(self._on_training_status)
        self.training_worker.finished_signal.connect(self._on_training_finished)
        self.training_worker.error_signal.connect(self._on_training_error)
        self.training_worker.start()

    def _on_training_status(self, msg: str):
        if self.training_status_lbl is not None:
            self.training_status_lbl.setText("Training status: Running")
        if self.training_summary_lbl is not None:
            self.training_summary_lbl.setText("Dataset preparation and training are in progress.")
        if self.status_lbl is not None:
            self.status_lbl.setText(self._compact_training_msg(msg, 90))
        self._update_training_card_from_log(msg)

    def _extract_model_path_from_training_summary(self, summary: dict) -> str:
        if not summary:
            return ""
        direct_keys = ["vit_model_path", "model_path", "checkpoint_path", "best_model_path", "final_model_path"]
        for key in direct_keys:
            value = summary.get(key)
            if value:
                return str(value)
        for item in summary.get("results", []) or []:
            for key in direct_keys:
                value = item.get(key)
                if value:
                    return str(value)
            run_dir = item.get("run_dir")
            if run_dir:
                for candidate in ["best.pth", "checkpoint_best.pth", "checkpoint_epoch_49.pth", "model.pth"]:
                    path = os.path.join(run_dir, candidate)
                    if os.path.exists(path):
                        return path
        return ""

    def _on_training_finished(self, summary: dict):
        self.training_in_progress = False
        self._set_controls_enabled(True)
        if self.preview_timer:
            self.preview_timer.start(1500)
        self.latest_training_summary = summary or {}
        self.recipe_doc["training_summary"] = dict(self.latest_training_summary)
        self.recipe_doc["vit_model_path"] = self._extract_model_path_from_training_summary(summary or {})

        if self.training_progress is not None:
            self.training_progress.setValue(100)
        if self.training_percent_lbl is not None:
            self.training_percent_lbl.setText("100%")
        if self.training_status_lbl is not None:
            self.training_status_lbl.setText("Training status: Completed")
        if self.training_summary_lbl is not None:
            self.training_summary_lbl.setText("Training completed. Review validation before saving the recipe.")
        if self.status_lbl is not None:
            self.status_lbl.setText("Training completed")

        for item in (summary or {}).get("results", []) or []:
            serial = str(item.get("serial", ""))
            if not serial:
                continue
            if item.get("status") in ("done", "success", "completed"):
                self._set_camera_status(serial, "Status: Completed", "#2a6e2f", "done")
            elif item.get("status") in ("skipped",):
                self._set_camera_status(serial, f"Status: Skipped\n{item.get('reason', '')}", "#9a92ae", "skipped")
            else:
                self._set_camera_status(serial, "Status: Completed", "#2a6e2f", "done")

        QMessageBox.information(self, "Training", "Training completed. Please run validation next.")
        self._switch_tab(TAB_VALIDATION)

    def _on_training_error(self, err: str):
        self.training_in_progress = False
        self._set_controls_enabled(True)
        if self.preview_timer:
            self.preview_timer.start(1500)
        if self.training_status_lbl is not None:
            self.training_status_lbl.setText("Training status: Failed")
        if self.status_lbl is not None:
            self.status_lbl.setText("Training failed")
        QMessageBox.critical(self, "Training Error", err)
        self._switch_tab(TAB_TRAINING)

    # ======================================================================
    # F-019 VALIDATION
    # ======================================================================
    def _build_validation_page(self):
        root = QVBoxLayout(self.validation_page)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(12)

        card = QFrame()
        card.setObjectName("PageCard")
        lay = QVBoxLayout(card)
        lay.setContentsMargins(24, 20, 24, 20)
        lay.setSpacing(16)
        lay.addLayout(self._section_header(
            "VIT Model Accuracy Validation",
            "Review validation metrics such as precision, recall, F1 score and confusion matrix before accepting the model.",
        ))

        self.validation_status_lbl = QLabel("Validation Status: Not run")
        self.validation_status_lbl.setObjectName("StatusPill")
        lay.addWidget(self.validation_status_lbl)

        self.validation_metrics_lbl = QLabel("No validation report available yet.")
        self.validation_metrics_lbl.setObjectName("InfoBox")
        self.validation_metrics_lbl.setWordWrap(True)
        self.validation_metrics_lbl.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.validation_metrics_lbl.setMinimumHeight(420)
        lay.addWidget(self.validation_metrics_lbl, 1)

        btn_row = QHBoxLayout()
        run_btn = self._make_button("Run Validation", "primary")
        run_btn.clicked.connect(self._run_validation)
        accept_btn = self._make_button("Accept Model", "success")
        accept_btn.clicked.connect(lambda: self._set_validation_acceptance(True))
        reject_btn = self._make_button("Reject Model", "danger")
        reject_btn.clicked.connect(lambda: self._set_validation_acceptance(False))
        next_btn = self._make_button("Next: Save Recipe", "secondary")
        next_btn.clicked.connect(lambda: self._switch_tab(TAB_SAVE_RECIPE))
        btn_row.addWidget(run_btn)
        btn_row.addWidget(accept_btn)
        btn_row.addWidget(reject_btn)
        btn_row.addStretch(1)
        btn_row.addWidget(next_btn)
        lay.addLayout(btn_row)

        root.addWidget(card)

    def _run_validation(self):
        sku_name = self._get_sku_name()
        result = run_validation_for_sku(
            media_path=self.media_path,
            sku_name=sku_name,
            training_summary=self.latest_training_summary,
        )
        self.latest_validation_result = result or {}
        self.recipe_doc["validation_result"] = dict(self.latest_validation_result)
        self._refresh_validation_ui()

    def _set_validation_acceptance(self, accepted: bool):
        if not self.latest_validation_result:
            QMessageBox.warning(self, "Validation", "Run validation first.")
            return
        self.latest_validation_result["accepted"] = accepted
        self.latest_validation_result["status"] = "ACCEPTED" if accepted else "REJECTED"
        self.recipe_doc["validation_result"] = dict(self.latest_validation_result)
        self._refresh_validation_ui()

    def _refresh_validation_ui(self):
        result = self.latest_validation_result or {}
        status = result.get("status", "UNKNOWN")
        accepted = bool(result.get("accepted", False))
        if self.validation_status_lbl is not None:
            self.validation_status_lbl.setText(f"Validation Status: {status} | Accepted: {'YES' if accepted else 'NO'}")
        text = (
            f"F1 Macro: {result.get('f1_macro')}\n\n"
            f"Precision:\n{result.get('precision', {})}\n\n"
            f"Recall:\n{result.get('recall', {})}\n\n"
            f"F1:\n{result.get('f1', {})}\n\n"
            f"Confusion Matrix:\n{result.get('confusion_matrix', [])}\n\n"
            f"Report:\n{result.get('report_path', '')}\n\n"
            f"Message:\n{result.get('message', '')}"
        )
        if self.validation_metrics_lbl is not None:
            self.validation_metrics_lbl.setText(text)

    # ======================================================================
    # F-020 / F-041 / F-042 SAVE RECIPE
    # ======================================================================
    def _build_recipe_page(self):
        root = QVBoxLayout(self.recipe_page)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(12)

        card = QFrame()
        card.setObjectName("PageCard")
        lay = QVBoxLayout(card)
        lay.setContentsMargins(24, 20, 24, 20)
        lay.setSpacing(16)
        lay.addLayout(self._section_header(
            "Save SKU Recipe & Link Model",
            "Save the complete SKU recipe including axis targets, camera/laser profile links, model path and validation result.",
        ))

        self.recipe_summary_lbl = QLabel("Recipe preview not generated yet.")
        self.recipe_summary_lbl.setObjectName("InfoBox")
        self.recipe_summary_lbl.setWordWrap(True)
        self.recipe_summary_lbl.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.recipe_summary_lbl.setMinimumHeight(420)
        lay.addWidget(self.recipe_summary_lbl, 1)

        btn_row = QHBoxLayout()

        preview_btn = self._make_button("Preview Recipe", "secondary")
        preview_btn.clicked.connect(self._preview_recipe)

        save_btn = self._make_button("Save Recipe", "primary")
        save_btn.clicked.connect(self._save_recipe_final)

        self.load_machine_btn = self._make_button("Load Recipe to Machine", "primary")
        self.load_machine_btn.clicked.connect(self._load_saved_recipe_to_machine)
        self.load_machine_btn.setEnabled(False)

        btn_row.addWidget(preview_btn)
        btn_row.addStretch(1)
        btn_row.addWidget(save_btn)
        btn_row.addWidget(self.load_machine_btn)

        lay.addLayout(btn_row)

        root.addWidget(card)

    def _collect_camera_config_links(self) -> dict:
        root = os.path.join(self.media_path, "camera_profiles")
        return {"profile_root": root, "exists": os.path.isdir(root)}

    def _collect_laser_config_links(self) -> dict:
        root = os.path.join(self.media_path, "laser_profiles")
        return {"profile_root": root, "exists": os.path.isdir(root)}

    def _build_final_recipe_doc(self) -> dict:
        sku_name = self._get_sku_name()
        if not sku_name or sku_name == "unknown_sku":
            raise ValueError("Complete SKU Setup before saving recipe.")

        # If operator typed values manually but forgot Apply Manual Targets,
        # capture the table values before saving.
        if getattr(self, "axis_entry_mode", "capture") == "manual":
            ok = self._apply_manual_axis_targets_from_table(silent=True)
            if not ok:
                raise ValueError("Manual recipe target entry has invalid values.")

        recipe_axis_targets = self.recipe_doc.get("recipe_axis_targets", {}) or {}
        target_configs = self.recipe_service.get_recipe_target_configs()

        if not recipe_axis_targets:
            raise ValueError(
                "Recipe target values are not captured.\n\n"
                "Go to Axis Teaching and either:\n"
                "1. Click Capture All Live Targets, or\n"
                "2. Enter Target Values manually and click Apply Manual Targets."
            )

        required_keys = [
            cfg.get("target_key")
            for cfg in target_configs
            if cfg.get("target_key")
        ]

        missing_keys = [
            key for key in required_keys
            if key not in recipe_axis_targets
            or recipe_axis_targets.get(key, {}).get("value") in (None, "")
        ]

        if missing_keys:
            preview = "\n".join(missing_keys[:12])
            extra = "" if len(missing_keys) <= 12 else f"\n... and {len(missing_keys) - 12} more"
            raise ValueError(
                "Some recipe target values are missing:\n\n"
                f"{preview}{extra}\n\n"
                "Fill/capture all target values before saving recipe."
            )

        self._sync_legacy_axis_targets_from_recipe_targets()

        camera_targets = self.recipe_doc.get("camera_axis_targets", {}) or {}
        laser_targets = self.recipe_doc.get("laser_axis_targets", {}) or {}

        sku_meta = dict(self.sku_meta)
        sku_meta.pop("machine_serial", None)

        recipe_number = int(self.sku_meta.get("recipe_number", 1) or 1)

        recipe_doc = self.recipe_service.build_recipe_doc(
            sku_meta=sku_meta,
            camera_axis_targets=camera_targets,
            laser_axis_targets=laser_targets,
            recipe_axis_targets=recipe_axis_targets,
            camera_config_links=self._collect_camera_config_links(),
            laser_config_links=self._collect_laser_config_links(),
            vit_model_path=self.recipe_doc.get("vit_model_path", ""),
            training_summary=self.latest_training_summary,
            validation_result=self.latest_validation_result,
            author=str(self.sku_meta.get("operator") or "operator"),
        )

        recipe_doc["recipe_number"] = recipe_number
        recipe_doc["plc_recipe_number"] = recipe_number

        return recipe_doc

    def _preview_recipe(self):
        try:
            recipe_doc = self._build_final_recipe_doc()
            recipe_number = int(recipe_doc.get("recipe_number", 0) or 0)
            existing_recipe = self.recipe_service.find_recipe_by_number(recipe_number)

            if existing_recipe:
                existing_sku = existing_recipe.get("sku_name", "UNKNOWN")
                existing_version = existing_recipe.get("version", "-")

                QMessageBox.warning(
                    self,
                    "Duplicate Recipe Number",
                    (
                        f"Recipe number {recipe_number} already exists.\n\n"
                        f"Existing SKU: {existing_sku}\n"
                        f"Version: {existing_version}\n\n"
                        "Recipe was not saved again. Please use a different recipe number."
                    )
                )
                return
            recipe_axis_targets = recipe_doc.get("recipe_axis_targets", {}) or {}

            machine_count = sum(
                1 for v in recipe_axis_targets.values()
                if str(v.get("group", "")).upper() == "MACHINE"
            )
            camera_count = sum(
                1 for v in recipe_axis_targets.values()
                if str(v.get("group", "")).upper() == "CAMERA"
            )
            laser_count = sum(
                1 for v in recipe_axis_targets.values()
                if str(v.get("group", "")).upper() == "LASER"
            )

            text = (
                f"SKU: {recipe_doc.get('sku_name')}\n"
                f"Recipe Number: {recipe_doc.get('recipe_number')}\n"
                f"Tyre Name: {recipe_doc.get('tyre_name')}\n"
                f"Tyre Size: {recipe_doc.get('tyre_size')}\n"
                f"Tyre Outer Diameter: {recipe_doc.get('tyre_outer_diameter')}\n"
                f"Tyre RPM: {recipe_doc.get('tyre_rpm')}\n"
                f"Barcode: {self.sku_meta.get('barcode', '')}\n"
                f"Barcode Pattern: {recipe_doc.get('barcode_pattern')}\n"
                f"Version: {recipe_doc.get('version')}\n"
                f"Operator/Author: {recipe_doc.get('author')}\n"
                f"Inspection Zones: {recipe_doc.get('inspection_zones')}\n"
                f"Image Count / Zone: {recipe_doc.get('image_count_per_zone')}\n\n"

                f"Production Recipe Targets: {len(recipe_axis_targets)}\n"
                f"Machine Targets: {machine_count}\n"
                f"Camera Targets: {camera_count}\n"
                f"Laser Targets: {laser_count}\n\n"

                f"Legacy Camera Axis Targets: {len(recipe_doc.get('camera_axis_targets', {}))}\n"
                f"Legacy Laser Axis Targets: {len(recipe_doc.get('laser_axis_targets', {}))}\n\n"

                f"VIT Model Path:\n{recipe_doc.get('vit_model_path')}\n\n"
                f"Validation F1 Macro: {recipe_doc.get('validation_score')}\n"
                f"Status: {recipe_doc.get('status')}"
            )

            if self.recipe_summary_lbl is not None:
                self.recipe_summary_lbl.setText(text)

        except Exception as e:
            QMessageBox.warning(self, "Recipe Preview", str(e))
    
    def _load_saved_recipe_to_machine(self):
        """
        Load the currently saved New SKU recipe to machine.

        This uses the same backend as Recipe Management:
            RecipeService.write_recipe_to_plc()

        It writes:
            - recipe name to DB53 string tag, if enabled
            - recipe_axis_targets to DB53
            - recipe number to DB75.DBW288
            - verifies DB53 read-back
            - verifies recipe number read-back
        """

        recipe = self.saved_recipe_doc

        if not recipe:
            QMessageBox.warning(
                self,
                "Load Recipe to Machine",
                "Please save the recipe first before loading it to machine."
            )
            return

        if not recipe.get("recipe_axis_targets"):
            QMessageBox.warning(
                self,
                "Load Recipe to Machine",
                (
                    "This recipe does not contain recipe_axis_targets.\n\n"
                    "Please complete Axis Teaching and save recipe first."
                )
            )
            return

        reply = QMessageBox.question(
            self,
            "Load Recipe to Machine",
            (
                "This will write the saved recipe target values to PLC DB53, "
                "write the recipe name if enabled, write the recipe number to DB75.DBW288, "
                "and verify PLC read-back.\n\n"
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
                plc_client=self.plc_client,
            )

            msg = self._format_plc_result_message(result)

            QMessageBox.information(
                self,
                "PLC Recipe Load",
                msg
            )

            if self.recipe_summary_lbl is not None:
                old_text = self.recipe_summary_lbl.text()
                self.recipe_summary_lbl.setText(
                    old_text + "\n\n--- Load Recipe to Machine ---\n" + msg
                )

        except Exception as e:
            QMessageBox.critical(
                self,
                "PLC Recipe Load Error",
                str(e)
            )
    
    def _format_plc_result_message(self, result: Dict[str, Any]) -> str:
        verify_result = result.get("verify_result", {}) or {}
        recipe_name_result = result.get("recipe_name_result", {}) or {}
        recipe_number_result = result.get("recipe_number_result", {}) or {}

        plc_enabled = bool(result.get("enabled", False))
        plc_written = bool(result.get("written", False))
        plc_verified = bool(result.get("verified", False))

        written_items = result.get("written_items", []) or []
        skipped_items = result.get("skipped_items", []) or []
        mismatches = (
            result.get("mismatches", [])
            or verify_result.get("mismatches", [])
            or []
        )

        if not plc_enabled:
            return (
                "PLC Write: Disabled\n"
                f"PLC Message: {result.get('message', '')}"
            )

        msg = (
            f"PLC Write: {'OK' if plc_written else 'NOT OK'}\n"
            f"PLC Verify: {'OK' if plc_verified else 'NOT OK / SKIPPED'}\n"
            f"Recipe Name Write: {'OK' if recipe_name_result.get('written') else 'NOT OK / SKIPPED'}\n"
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
                    f"DB={item.get('db')} | Byte={item.get('byte')}"
                )

            msg += "\n\nMismatches:\n" + "\n".join(mismatch_lines)

            if len(mismatches) > 8:
                msg += f"\n... and {len(mismatches) - 8} more."

        return msg
    def _save_recipe_final(self):
        try:
            recipe_doc = self._build_final_recipe_doc()

            result = self.recipe_service.save_recipe(
                recipe_doc,
                plc_client=self.plc_client,
                write_to_plc=None,
            )
            self.saved_recipe_doc = dict(recipe_doc)
            self.saved_recipe_doc["_id"] = result.get("inserted_id")
            self.saved_recipe_doc["version"] = result.get("version", recipe_doc.get("version"))

            self.saved_recipe_result = dict(result)

            if self.load_machine_btn is not None:
                self.load_machine_btn.setEnabled(True)
            plc_result = result.get("plc_result", {}) or {}
            verify_result = plc_result.get("verify_result", {}) or {}
            recipe_number_result = plc_result.get("recipe_number_result", {}) or {}
            plc_enabled = bool(plc_result.get("enabled", False))
            plc_written = bool(plc_result.get("written", False))
            plc_verified = bool(plc_result.get("verified", False))

            written_items = plc_result.get("written_items", []) or []
            skipped_items = plc_result.get("skipped_items", []) or []
            mismatches = plc_result.get("mismatches", []) or verify_result.get("mismatches", []) or []

            if not plc_enabled:
                plc_block = (
                    "PLC Write: Disabled\n"
                    f"PLC Message: {plc_result.get('message', '')}"
                )
            else:
                plc_block = (
                    f"PLC Write: {'OK' if plc_written else 'NOT OK'}\n"
                    f"PLC Verify: {'OK' if plc_verified else 'NOT OK / SKIPPED'}\n"
                    f"Recipe Number Write: {'OK' if recipe_number_result.get('written') else 'NOT OK / SKIPPED'}\n"
                    f"Recipe Number Verify: {'OK' if recipe_number_result.get('verified') else 'NOT OK / SKIPPED'}\n"
                    f"Targets Written: {len(written_items)}\n"
                    f"Targets Skipped: {len(skipped_items)}\n"
                    f"Verify Count: {verify_result.get('verified_count', 0)}\n"
                    f"Mismatch Count: {verify_result.get('mismatch_count', len(mismatches))}\n"
                    f"PLC Message: {plc_result.get('message', '')}"
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

                extra = ""
                if len(mismatches) > 8:
                    extra = f"\n... and {len(mismatches) - 8} more mismatches"

                mismatch_text = "\n\nPLC Mismatches:\n" + "\n".join(mismatch_lines) + extra
            else:
                mismatch_text = ""

            msg = (
                f"Recipe saved successfully.\n\n"
                f"SKU: {result.get('sku_name')}\n"
                f"Version: {result.get('version')}\n"
                f"Local Backup:\n{result.get('backup_path')}\n\n"
                f"{plc_block}"
                f"{mismatch_text}"
            )

            if self.recipe_summary_lbl is not None:
                self.recipe_summary_lbl.setText(msg)

            QMessageBox.information(self, "Recipe Saved", msg)

        except Exception as e:
            QMessageBox.critical(self, "Recipe Save Error", str(e))

    def close_page(self):
        if self.capture_in_progress or self.training_in_progress:
            QMessageBox.warning(self, "New SKU", "Please wait until capture/training is completed.")
            return
        if self.on_close:
            self.on_close()
