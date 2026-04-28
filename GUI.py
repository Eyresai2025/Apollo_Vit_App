# Import required libraries
import sys
import os
import signal
from datetime import datetime
from threading import Lock, Event
import numpy as np # type: ignore
import pandas as pd # type: ignore
from PIL import Image, ImageTk # type: ignore
import cv2 # type: ignore
import torch # type: ignore
import warnings
warnings.filterwarnings("ignore")
import threading
from PyQt5.QtWidgets import * # type: ignore
from PyQt5.QtCore import * # type: ignore
from PyQt5.QtGui import * # type: ignore
from PyQt5.QtGui import QPainter, QImageReader # type: ignore
from src.COMMON.db import get_db
from ultralytics import YOLO # type: ignore
from sahi import AutoDetectionModel # type: ignore
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import torchvision # type: ignore
import logging
import subprocess
import platform
from src.COMMON.common import load_env
from src.Pages.test_mode_page import TestModePage
from src.Pages.new_sku_page import NewSKUPage
from src.Pages.repeatability_page import RepeatabilityPage
from src.Pages.action_code_plan_page import ActionCodePlanPage
from src.Pages.dashboard import ApolloDashboardCardsWidget
from PyQt5.QtCore import Qt  # type: ignore
from src.Pages.annotation_tool import AnnotationTool  
from pathlib import Path
from snap7 import Client # type: ignore

####Local files imports
from src.camera.cam_connections import LineScanCamera
from src.models.Pipeline.R_Detection_align_crop import build_r_detector
from src.Main_cam import run_capture_folder_cycle, preload_live_runtimes, CAMERA_CAPTURE_ENABLED, start_continuous_cycle
from src.COMMON.db import get_db

os.environ['CUDA_MODULE_LOADING'] = 'LAZY'
os.environ["QT_DEVICE_PIXEL_RATIO"] = "0"
os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"
os.environ["QT_SCALE_FACTOR"] = "1"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def app_dir() -> Path:
    """Where bundled resources exist."""
    if getattr(sys, "frozen", False):
        if hasattr(sys, "_MEIPASS"):
            base = Path(sys._MEIPASS)
            return base
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent

BASE_DIR = app_dir()
MEDIA_PATH = str(BASE_DIR / "media")
ENV_PATH   = str(BASE_DIR / ".env")

try:
    os.chdir(str(BASE_DIR))
except Exception:
    pass

os.environ["CUPY_CUDA_PRELOAD"] = "0"
os.environ["CUPY_ACCELERATORS"] = ""

# Load environment variables
env_vars = load_env(ENV_PATH)  
deployment = env_vars.get('DEPLOYMENT')
plc_ip = env_vars.get('PLC_IP')
weight_path = env_vars.get('WEIGHT_FILE_Apollo')
weight_path_re_onnx = env_vars.get('WEIGHT_FILE_RE_ONNX')
weight_path_vit = env_vars.get('VIT_CHECKPOINT')
valid_username = env_vars.get("VALID_USERNAME")
valid_password = env_vars.get("VALID_PASSWORD")

# MongoDB Connections
mydb = get_db()
tyre_details_col = mydb["TYRE DETAILS"]
new_sku_col = mydb["New SKU"]
repeatability_col = mydb["Repeatability"]
accounts_col = mydb["Accounts"]

LOCAL_MULTI_SIDE_TEST_FOLDER = env_vars.get(
    "MULTI_CAPTURE_ROOT",
    r"C:\Users\Hi\OneDrive - radometech.com\Desktop\New folder\media\capture\DEMO_CYCLE_FOLDER"
)

MAIN_SEG_MODEL_PATH = os.path.join(MEDIA_PATH, "weights", weight_path) if weight_path else None
MAIN_R_DETECTOR_PATH = os.path.join(MEDIA_PATH, "weights", weight_path_re_onnx) if weight_path_re_onnx else None
MAIN_VIT_CHECKPOINT_PATH = os.path.join(MEDIA_PATH, "weights", weight_path_vit) if weight_path_vit else None

# ---------------- TORCH DEVICE + CPU FALLBACK ----------------
if torch.cuda.is_available():
    try:
        _ = torch.randn(1).to("cuda")
        device = torch.device("cuda")
        TORCH_GPU_OK = True
    except Exception as e:
        logger.warning(f"CUDA reported available but failed test: {e}")
        logger.info("Falling back to CPU.")
        device = torch.device("cpu")
        TORCH_GPU_OK = False
else:
    device = torch.device("cpu")
    TORCH_GPU_OK = False

logger.info(f"Using device: {device}")

MEDIA_ROOT_INIT_ERROR = False
MEDIA_PATH = str(BASE_DIR / "media")
RAW_IMAGE_DIR = os.path.join(MEDIA_PATH, "raw images")
STARTUP_IMAGE_PATHS = [
    os.path.join(RAW_IMAGE_DIR, "1.jpg"),
    os.path.join(RAW_IMAGE_DIR, "2.jpg"),
    os.path.join(RAW_IMAGE_DIR, "3.jpg"),
    os.path.join(RAW_IMAGE_DIR, "4.jpg"),
    os.path.join(RAW_IMAGE_DIR, "5.jpg"),
]

BAR_CODE_DIR = os.path.join(MEDIA_PATH, "barcode_images")
TEST_MODE_REPORTS = os.path.join(MEDIA_PATH, "TestMode Reports")

def get_available_sku_names(media_root):
    calibration_root = os.path.join(media_root, "calibration")
    if not os.path.isdir(calibration_root):
        return []
    sku_names = []
    for name in sorted(os.listdir(calibration_root)):
        full_path = os.path.join(calibration_root, name)
        artifacts_dir = os.path.join(full_path, "artifacts")
        if os.path.isdir(full_path) and name.upper().startswith("SKU") and os.path.isdir(artifacts_dir):
            sku_names.append(name)
    return sku_names

def _load_onnx_r_with_fallback(onnx_rel_path, label="R-detect ONNX", conf=0.3):
    if onnx_rel_path is None:
        logger.warning(f"{label} path is None from env. Skipping load.")
        return None, None
    full_path = os.path.join(MEDIA_PATH, f"weights/{onnx_rel_path}")
    if not os.path.exists(full_path):
        logger.error(f"{label} file not found at: {full_path}")
        return None, full_path
    try:
        target_device = "cuda" if TORCH_GPU_OK else "cpu"
        try:
            det_model = build_r_detector(full_path, conf=conf, device=target_device)
        except Exception as e:
            logger.warning(f"Loading {label} on {target_device} failed: {e}")
            logger.info(f"Falling back {label} to CPU.")
            det_model = build_r_detector(full_path, conf=conf, device="cpu")
        logger.info(f"Loaded {label} from {full_path}")
        return det_model, full_path
    except Exception as e:
        logger.error(f"Failed to load {label} from {full_path}: {e}")
        return None, full_path

# --- Load models with CPU-safe fallback ---
shared_r_detector_onnx, shared_r_detector_onnx_path = _load_onnx_r_with_fallback(
    weight_path_re_onnx, label="R-detect ONNX", conf=0.3,
)
logger.info("All Models loading step completed (with CPU/GPU fallback).")

# ---------------- PLC SETUP ----------------
plc_client = None

def connect_plc():
    global plc_client
    dep = str(deployment) == "True"
    if not dep:
        logger.info("PLC not connected because DEPLOYMENT is False")
        return None
    if not plc_ip:
        logger.info("PLC not connected because PLC_IP is missing in .env")
        return None
    try:
        plc_client = Client()
        logger.info(f"Connecting to PLC at {plc_ip}...")
        plc_client.connect(plc_ip, 0, 1)
        if plc_client.get_connected():
            logger.info("PLC CONNECTED")
            return plc_client
        else:
            logger.error("PLC connection failed")
            plc_client = None
            return None
    except Exception as e:
        logger.error(f"PLC connection failed: {e}")
        plc_client = None
        return None

def disconnect_plc():
    global plc_client
    try:
        if plc_client is not None and plc_client.get_connected():
            plc_client.disconnect()
            logger.info("PLC disconnected")
    except Exception as e:
        logger.error(f"PLC disconnect error: {e}")
    finally:
        plc_client = None

# Camera initialization
# REPLACE:
if deployment == "True":
    connect_plc()
    from src.camera.cam_connections import MultiCameraManager
    ...
    multi_cam = MultiCameraManager()
    multi_cam.connect_all()

# WITH:
if deployment == "True":
    connect_plc()
    from src.camera.HARDWARE_TRIGGER import MultiCameraManager
    logger.info("="*50)
    logger.info("Initializing Apollo Inspection System — 5 cameras")
    logger.info("="*50)
    try:
        multi_cam = MultiCameraManager()
        multi_cam.connect_all()
        logger.info(f"✅ {len(multi_cam.cameras)} cameras connected and configured")
    except Exception as e:
        logger.error(f"❌ Camera initialization failed: {e}")
        multi_cam = None
    logger.info("="*50)
else:
    logger.info("Cameras not initialized — running in local/demo mode")
    logger.info("PLC not connected because DEPLOYMENT is False")
    multi_cam = None

# ============================================================================
# THREAD MANAGER - Proper QThread Lifecycle Management
# ============================================================================

class ThreadManager:
    """Manages QThread lifecycle to prevent memory leaks and freezes"""
    
    def __init__(self, parent=None):
        self.parent = parent
        self.active_threads = {}
        self._lock = Lock()
    
    def start_thread(self, name, worker, on_finished=None, on_error=None):
        """Start a thread safely with proper cleanup"""
        with self._lock:
            # Stop existing thread with same name if running
            if name in self.active_threads:
                old_thread = self.active_threads[name]
                if old_thread.isRunning():
                    logger.debug(f"Stopping existing thread '{name}'")
                    old_thread.quit()
                    if not old_thread.wait(3000):
                        logger.warning(f"Thread '{name}' didn't stop, terminating")
                        old_thread.terminate()
                        old_thread.wait()
                old_thread.deleteLater()
            
            # Create new thread
            thread = QThread(self.parent)
            worker.moveToThread(thread)
            
            # Connect with explicit QueuedConnection
            thread.started.connect(worker.run, Qt.QueuedConnection)
            
            if on_finished:
                worker.finished.connect(on_finished, Qt.QueuedConnection)
            if on_error:
                worker.error.connect(on_error, Qt.QueuedConnection)
            
            worker.finished.connect(thread.quit, Qt.QueuedConnection)
            worker.error.connect(thread.quit, Qt.QueuedConnection)
            
            # Cleanup on finish
            def cleanup():
                with self._lock:
                    self.active_threads.pop(name, None)
            thread.finished.connect(cleanup)
            thread.finished.connect(thread.deleteLater)
            
            self.active_threads[name] = thread
            thread.start()
            return True
    
    def stop_all(self, timeout=5000):
        """Stop all running threads"""
        with self._lock:
            for name, thread in list(self.active_threads.items()):
                if thread.isRunning():
                    thread.quit()
                    if not thread.wait(timeout):
                        thread.terminate()
                        thread.wait()
            self.active_threads.clear()

# ============================================================================
# IMAGE CACHE - Prevents Repeated Image Loading
# ============================================================================

class ImageCache:
    """Thread-safe image cache with size limit"""
    
    def __init__(self, max_size=50):
        self._cache = {}
        self._lock = Lock()
        self._max_size = max_size
    
    def get(self, key, loader_func):
        """Get cached image or load it"""
        with self._lock:
            if key in self._cache:
                return self._cache[key]
        
        image = loader_func()
        
        if image is not None and not (hasattr(image, 'isNull') and image.isNull()):
            with self._lock:
                if len(self._cache) >= self._max_size:
                    self._cache.clear()
                self._cache[key] = image
        
        return image
    
    def clear(self):
        """Clear the cache"""
        with self._lock:
            self._cache.clear()

# Global image cache instance
image_cache = ImageCache(max_size=50)

# ============================================================================
# WORKER CLASSES
# ============================================================================

class RuntimePreloadWorker(QObject):
    finished = pyqtSignal(str)
    error = pyqtSignal(str)
    
    def __init__(self, media_root, sku_name, device, seg_model_a_path,
                 seg_model_b_path, vit_checkpoint_path, r_detector_path):
        super().__init__()
        self.media_root = media_root
        self.sku_name = sku_name
        self.device = device
        self.seg_model_a_path = seg_model_a_path
        self.seg_model_b_path = seg_model_b_path
        self.vit_checkpoint_path = vit_checkpoint_path
        self.r_detector_path = r_detector_path
        self._stop_event = Event()
    
    @pyqtSlot()
    def run(self):
        try:
            if self._stop_event.is_set():
                return
            preload_live_runtimes(
                capture_root=self.media_root,
                media_root=self.media_root,
                sku_name=self.sku_name,
                device=self.device,
                seg_model_a_path=self.seg_model_a_path,
                seg_model_b_path=self.seg_model_b_path,
                vit_checkpoint_path=self.vit_checkpoint_path,
                r_detector_path=self.r_detector_path,
                sides_to_run=["all"],
            )
            self.finished.emit(f"Runtime preload completed | SKU={self.sku_name}")
        except Exception as e:
            self.error.emit(str(e))
    
    def stop(self):
        self._stop_event.set()

class LiveInspectionWorker(QObject):
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)
    
    def __init__(self, media_root, sku_name="SKU_001", tyre_name="195_65_R15",
                 device="cuda", seg_model_a_path=None, seg_model_b_path=None,
                 vit_checkpoint_path=None, r_detector_path=None,
                 multi_camera_manager=None, demo_capture_root=None):
        super().__init__()
        self.media_root = media_root
        self.sku_name = sku_name
        self.tyre_name = tyre_name
        self.device = device
        self.seg_model_a_path = seg_model_a_path
        self.seg_model_b_path = seg_model_b_path
        self.vit_checkpoint_path = vit_checkpoint_path
        self.r_detector_path = r_detector_path
        self.multi_camera_manager = multi_camera_manager
        self.demo_capture_root = demo_capture_root
        self._stop_event = Event()
    
    @pyqtSlot()
    def run(self):
        try:
            if self._stop_event.is_set():
                return
            result = run_capture_folder_cycle(
                media_root=self.media_root,
                sku_name=self.sku_name,
                tyre_name=self.tyre_name,
                device=self.device,
                seg_model_a_path=self.seg_model_a_path,
                seg_model_b_path=self.seg_model_b_path,
                vit_checkpoint_path=self.vit_checkpoint_path,
                r_detector_path=self.r_detector_path,
                multi_camera_manager=self.multi_camera_manager,
                demo_capture_root=self.demo_capture_root,
            )
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))
    
    def stop(self):
        self._stop_event.set()

class LatestCycleImagesWorker(QObject):
    finished = pyqtSignal(object)
    error = pyqtSignal(str)
    
    def __init__(self, media_root, panel_size=(260, 700), fallback_paths=None):
        super().__init__()
        self.media_root = media_root
        self.panel_w, self.panel_h = panel_size
        self.fallback_paths = fallback_paths or {}
        self._stop_event = Event()
    
    @pyqtSlot()
    def run(self):
        try:
            if self._stop_event.is_set():
                return
            payload = self._collect_latest_cycle_images()
            self.finished.emit(payload)
        except Exception as e:
            self.error.emit(str(e))
    
    def _collect_latest_cycle_images(self):
        payload = {"cycle_dir": None, "images": {}}
        cycle_dir = self._get_latest_cycle_dir()
        payload["cycle_dir"] = cycle_dir
        
        side_folders = {
            "sidewall1": "sidewall1",
            "sidewall2": "sidewall2",
            "innerwall": "innerwall",
            "tread": "tread",
            "bead": "bead",
        }
        
        for side_key, folder_name in side_folders.items():
            img_path = None
            qimage = None
            if cycle_dir:
                img_path = self._find_latest_final_image(cycle_dir, folder_name)
            if not img_path:
                img_path = self.fallback_paths.get(side_key)
            if img_path and os.path.exists(img_path):
                qimage = self._load_scaled_qimage(img_path)
            payload["images"][side_key] = {"path": img_path, "qimage": qimage}
        
        return payload
    
    def _get_latest_cycle_dir(self):
        output_root = os.path.join(self.media_root, "output")
        if not os.path.isdir(output_root):
            return None
        cycle_dirs = [os.path.join(output_root, d) for d in os.listdir(output_root)
                     if os.path.isdir(os.path.join(output_root, d))]
        if not cycle_dirs:
            return None
        cycle_dirs.sort(key=os.path.getmtime, reverse=True)
        return cycle_dirs[0]
    
    def _find_latest_final_image(self, cycle_dir, side_folder):
        side_final_root = os.path.join(cycle_dir, side_folder, "final")
        if not os.path.isdir(side_final_root):
            return None
        valid_exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
        candidates = []
        for root, _, files in os.walk(side_final_root):
            for f in files:
                lf = f.lower()
                full = os.path.join(root, f)
                if lf == "final_stitched.png":
                    candidates.append((0, os.path.getmtime(full), full))
                elif lf.endswith(valid_exts):
                    candidates.append((1, os.path.getmtime(full), full))
        if not candidates:
            return None
        candidates.sort(key=lambda x: (x[0], -x[1]))
        return candidates[0][2]
    
    def _load_scaled_qimage(self, img_path):
        """Load scaled image with caching"""
        cache_key = f"qimage_{img_path}_{self.panel_w}_{self.panel_h}"
        
        def loader():
            reader = QImageReader(img_path)
            reader.setAutoTransform(True)
            original_size = reader.size()
            if original_size.isValid() and original_size.width() > 0 and original_size.height() > 0:
                scaled_size = original_size.scaled(self.panel_w, self.panel_h, Qt.KeepAspectRatio)
                reader.setScaledSize(scaled_size)
            image = reader.read()
            return image if not image.isNull() else None
        
        return image_cache.get(cache_key, loader)
    
    def stop(self):
        self._stop_event.set()

# ============================================================================
# IMAGE VIEWER
# ============================================================================

class ImageViewer(QDialog):
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
                QPushButton:hover {
                    background:#6b2aa3;
                }
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

# ============================================================================
# MAIN WINDOW
# ============================================================================

class MainWindow(QMainWindow):
    # Throttle constants
    REFRESH_MIN_INTERVAL = 1.0  # Minimum seconds between image refreshes
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle('EyresAi QC+')
        self.back_btn = None
        screen = QGuiApplication.primaryScreen().availableGeometry()
        self.screen_w = screen.width()
        self.screen_h = screen.height()
        self.ui_scale = min(self.screen_w / 1920.0, self.screen_h / 1080.0)
        
        self.resize(self.screen_w, self.screen_h)
        self.setWindowIcon(QIcon(os.path.join(MEDIA_PATH, "img/smartQC-.ico")))
        
        # Thread management
        self.thread_manager = ThreadManager(parent=self)
        
        # Refresh throttling
        self._last_refresh_time = 0
        self._refresh_lock = Lock()
        
        # Global variables
        self.uploaded_image_path = None
        self.image_labels = {}
        self.img_labels = {}
        self.content_stack = None
        self.action_plan_page = None
        self.test_mode_page = None
        self.new_sku_page = None
        self.available_skus = []
        self.pending_preload_sku = None
        self.current_preloaded_sku = None
        self.inspection = None
        self.multi_cam = multi_cam
        self.continuous_worker = None
        self.is_continuous_running = False
        
        self.side_order = [
            ("sidewall1", "Side Wall 1"),
            ("sidewall2", "Side Wall 2"),
            ("innerwall", "Inner Side"),
            ("tread", "Tread"),
            ("bead", "Bead"),
        ]
        
        self.image_labels_by_side = {}
        self.current_panel_image_paths = {}
        self.latest_loaded_cycle_dir = None
        self.image_refresh_busy = False
        
        # UI responsiveness tracker
        self._last_ui_update = time.time()
        
        # Setup UI
        self.setup_ui()
        self.load_startup_images()
        
        self.available_skus = get_available_sku_names(MEDIA_PATH)
        self.selected_live_sku = ""
        self.selected_live_tyre_name = ""
        self.pending_live_start = False
        self.pending_live_sku = None
        self.pending_live_tyre_name = None
        
        # Initial delayed refresh
        QTimer.singleShot(1200, self.refresh_cycle_images_async)
        
        # Start timers
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_datetime)
        self.update_timer.start(1000)
        
        self.update_label_timer = QTimer()
        self.update_label_timer.timeout.connect(self.update_label_async)
        self.update_label_timer.start(5000)
        
        self.update_images_timer = QTimer(self)
        self.update_images_timer.timeout.connect(self.refresh_cycle_images_async)
        self.update_images_timer.start(3000)
        
        # UI freeze monitor (for debugging)
        self._freeze_monitor = QTimer(self)
        self._freeze_monitor.timeout.connect(self._check_ui_responsiveness)
        self._freeze_monitor.start(3000)
        
        # ---------- BOTTOM MARQUEE COPYRIGHT BAR ----------
        status_bar = QStatusBar()
        status_bar.setStyleSheet("background-color: white;")
        status_bar.setSizeGripEnabled(False)
        self.setStatusBar(status_bar)
        
        self.copy_full_text = (
            "Copyright © Radome Technologies and Services Pvt Ltd | "
            "All Rights Reserved | Our privacy policy | www.radometechnologies.com | "
            "Version: v1.0"
        )
        self.copy_padded_text = " " * 40 + self.copy_full_text + " " * 40
        self.copy_index = 0
        
        self.copyright_label = QLabel()
        self.copyright_label.setStyleSheet("font: bold 12px 'Arial'; color: black;")
        self.copyright_label.setAlignment(Qt.AlignCenter)
        status_bar.addWidget(self.copyright_label, 1)
        
        self.copy_timer = QTimer(self)
        self.copy_timer.timeout.connect(self.update_marquee_text)
        self.copy_timer.start(150)
    
    # ========================================================================
    # UI FREEZE DETECTION
    # ========================================================================
    
    def _check_ui_responsiveness(self):
        """Monitor if UI thread is responsive"""
        current_time = time.time()
        time_since_update = current_time - self._last_ui_update
        if time_since_update > 5.0:
            logger.warning(f"⚠️ UI may be frozen! Last update: {time_since_update:.1f}s ago")
        self._last_ui_update = current_time
    
    def _mark_ui_active(self):
        """Mark that UI thread is alive"""
        self._last_ui_update = time.time()
    
    # ========================================================================
    # ASYNC LABEL UPDATE (Non-blocking DB query)
    # ========================================================================
    
    def update_label_async(self):
        """Update label count without blocking UI"""
        def do_query():
            try:
                currentdate = datetime.now().strftime("%d-%m-%Y")
                cnt = tyre_details_col.count_documents({"inspectionDate": currentdate})
                return cnt
            except Exception as e:
                logger.error(f"Error updating label: {e}")
                return 0
        
        # Run query in thread pool and update UI when done
        def update_ui():
            try:
                cnt = do_query()
                if self.label_count:
                    self.label_count.setText(str(cnt))
            except Exception as e:
                if self.label_count:
                    self.label_count.setText("0")
        
        QTimer.singleShot(0, update_ui)
    
    def update_label(self):
        """Sync version - deprecated, use update_label_async instead"""
        self.update_label_async()
    
    # ========================================================================
    # IMAGE LOADING
    # ========================================================================
    
    def set_label_image_safe(self, label, img_path, w, h, keep_aspect=True):
        try:
            if not img_path or not os.path.exists(img_path):
                label.clear()
                label.setText("🖼️")
                label.setStyleSheet("font: 48px; color: grey;")
                return False
            pixmap = QPixmap(img_path)
            if pixmap.isNull():
                logger.warning(f"Invalid/corrupt image skipped: {img_path}")
                label.clear()
                label.setText("🖼️")
                label.setStyleSheet("font: 48px; color: grey;")
                return False
            scaled = pixmap.scaled(
                w, h,
                Qt.KeepAspectRatio if keep_aspect else Qt.IgnoreAspectRatio,
                Qt.SmoothTransformation
            )
            label.setPixmap(scaled)
            return True
        except Exception as e:
            logger.warning(f"Failed to load image {img_path}: {e}")
            label.clear()
            label.setText("🖼️")
            label.setStyleSheet("font: 48px; color: grey;")
            return False
    
    def s(self, value):
        return max(1, int(value * self.ui_scale))
    
    def refresh_available_skus(self):
        self.available_skus = get_available_sku_names(MEDIA_PATH)
        return self.available_skus
    
    def update_live_info_cards(self, sku_name, tyre_name):
        self.selected_live_sku = sku_name
        self.selected_live_tyre_name = tyre_name
        if hasattr(self, "selected_sku_value_label"):
            self.selected_sku_value_label.setText(sku_name or "--")
        if hasattr(self, "selected_tyre_value_label"):
            self.selected_tyre_value_label.setText(tyre_name or "--")
    
    # ========================================================================
    # THROTTLED IMAGE REFRESH
    # ========================================================================
    
    def refresh_cycle_images_async(self):
        """Throttled async image refresh"""
        current_time = time.time()
        
        # Throttle check
        if current_time - self._last_refresh_time < self.REFRESH_MIN_INTERVAL:
            return
        
        # Lock to prevent concurrent refreshes
        if not self._refresh_lock.acquire(blocking=False):
            return
        
        try:
            self._last_refresh_time = current_time
            
            label_widths = [lbl.width() for lbl in self.image_labels_by_side.values() if lbl is not None]
            label_heights = [lbl.height() for lbl in self.image_labels_by_side.values() if lbl is not None]
            
            panel_w = max(label_widths) if label_widths else 260
            panel_h = max(label_heights) if label_heights else 700
            panel_w = max(panel_w, 220)
            panel_h = max(panel_h, 500)
            
            worker = LatestCycleImagesWorker(
                media_root=MEDIA_PATH,
                panel_size=(panel_w, panel_h),
                fallback_paths=self.startup_image_paths,
            )
            
            def on_finished(payload):
                try:
                    self.on_cycle_images_ready(payload)
                finally:
                    self._refresh_lock.release()
            
            def on_error(message):
                logger.error(f"Image refresh error: {message}")
                self._refresh_lock.release()
            
            self.thread_manager.start_thread("image_refresh", worker, on_finished, on_error)
            
        except Exception as e:
            logger.error(f"Refresh error: {e}")
            self._refresh_lock.release()
    
    def on_cycle_images_ready(self, payload):
        """Handle loaded images - batched UI updates"""
        self._mark_ui_active()
        
        try:
            cycle_dir = payload.get("cycle_dir")
            images = payload.get("images", {})
            
            # Check if anything changed
            changed = False
            for side_key, _title in self.side_order:
                data = images.get(side_key, {})
                new_path = data.get("path")
                old_path = self.current_panel_image_paths.get(side_key)
                if new_path != old_path:
                    changed = True
                    break
            
            if cycle_dir == self.latest_loaded_cycle_dir and not changed:
                return
            
            self.latest_loaded_cycle_dir = cycle_dir
            
            # Batch all updates
            for side_key, _title in self.side_order:
                label = self.image_labels_by_side.get(side_key)
                if label is None:
                    continue
                
                data = images.get(side_key, {})
                qimage = data.get("qimage")
                img_path = data.get("path")
                
                if qimage is not None and not qimage.isNull():
                    pixmap = QPixmap.fromImage(qimage)
                    label.setPixmap(pixmap)
                    label.setAlignment(Qt.AlignCenter)
                    self.current_panel_image_paths[side_key] = img_path
                    label.mousePressEvent = self._make_open_image_handler(side_key)
                else:
                    self.current_panel_image_paths[side_key] = None
                    label.clear()
                    label.setText("🖼️")
                    label.setStyleSheet("font: 48px; color: grey; background-color: white;")
            
            # Single repaint
            self.update()
            
        except Exception as e:
            logger.error(f"Error displaying images: {e}")
    
    # ========================================================================
    # LIVE INSPECTION WITH THREAD SAFETY    # ========================================================================
    
    def open_live_selection_dialog(self):
        self.refresh_available_skus()
        
        dialog = QDialog(self)
        dialog.setWindowTitle("Live Inspection")
        dialog.resize(self.s(500), self.s(380))
        dialog.setWindowIcon(QIcon(os.path.join(MEDIA_PATH, "img/smartQC-.ico")))
        dialog.setModal(True)
        
        dialog.setStyleSheet("""
            QDialog { background: #f8f9fa; }
            QFrame#Card {
                background: white;
                border-radius: 20px;
                border: 1px solid #e1e4e8;
            }
            QLabel#Title {
                font: 700 18px 'Segoe UI';
                color: #1a1a2e;
                letter-spacing: 0.5px;
            }
            QLabel#FieldLabel {
                font: 600 13px 'Segoe UI';
                color: #4a5568;
                margin-bottom: 4px;
            }
            QComboBox {
                min-height: 44px;
                background: white;
                border: 2px solid #e2e8f0;
                border-radius: 12px;
                padding: 0 12px;
                font: 500 13px 'Segoe UI';
                color: #2d3748;
                selection-background-color: transparent;
            }
            QComboBox:hover {
                border: 2px solid #571c86;
                background: #faf5ff;
            }
            QComboBox:focus {
                border: 2px solid #571c86;
                background: white;
            }
            QComboBox::drop-down {
                subcontrol-origin: padding;
                subcontrol-position: top right;
                width: 30px;
                border: none;
                background: transparent;
            }
            QComboBox::down-arrow {
                image: none;
                border-left: 5px solid transparent;
                border-right: 5px solid transparent;
                border-top: 6px solid #571c86;
                margin-right: 12px;
            }
            QComboBox QAbstractItemView {
                background: white;
                border: 2px solid #e2e8f0;
                border-radius: 12px;
                padding: 8px 0;
                margin-top: 4px;
                outline: 0px;
                selection-background-color: #f3e8ff;
                selection-color: #571c86;
            }
            QComboBox QAbstractItemView::item {
                min-height: 40px;
                padding: 8px 16px;
                margin: 2px 8px;
                border-radius: 8px;
                font: 500 13px 'Segoe UI';
                color: #2d3748;
            }
            QComboBox QAbstractItemView::item:hover {
                background: #f3e8ff;
                color: #571c86;
            }
            QComboBox QAbstractItemView::item:selected {
                background: #f3e8ff;
                color: #571c86;
            }
            QLineEdit {
                min-height: 44px;
                background: white;
                border: 2px solid #e2e8f0;
                border-radius: 12px;
                padding: 0 12px;
                font: 500 13px 'Segoe UI';
                color: #2d3748;
            }
            QLineEdit:hover {
                border: 2px solid #571c86;
                background: #faf5ff;
            }
            QLineEdit:focus {
                border: 2px solid #571c86;
                background: white;
            }
            QPushButton#CancelBtn {
                min-height: 44px;
                background: white;
                border: 2px solid #e2e8f0;
                border-radius: 12px;
                font: 600 13px 'Segoe UI';
                color: #4a5568;
                padding: 0 24px;
            }
            QPushButton#CancelBtn:hover {
                background: #f7fafc;
                border-color: #cbd5e0;
            }
            QPushButton#StartBtn {
                min-height: 44px;
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #571c86, stop:1 #6b2aa3);
                border: none;
                border-radius: 12px;
                font: 600 13px 'Segoe UI';
                color: white;
                padding: 0 32px;
            }
            QPushButton#StartBtn:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #6b2aa3, stop:1 #7b31be);
            }
            QScrollBar:vertical {
                border: none;
                background: #f1f5f9;
                width: 8px;
                border-radius: 4px;
                margin: 0px;
            }
            QScrollBar::handle:vertical {
                background: #cbd5e0;
                border-radius: 4px;
                min-height: 20px;
            }
        """)
        
        root = QVBoxLayout(dialog)
        root.setContentsMargins(20, 20, 20, 20)
        root.setSpacing(16)
        
        card = QFrame()
        card.setObjectName("Card")
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(28, 28, 28, 28)
        card_layout.setSpacing(20)
        
        title_label = QLabel("Start Live Inspection")
        title_label.setObjectName("Title")
        title_label.setAlignment(Qt.AlignCenter)
        card_layout.addWidget(title_label)
        
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setStyleSheet("background-color: #e2e8f0; max-height: 1px; margin: 4px 0;")
        card_layout.addWidget(separator)
        
        sku_label = QLabel("Select SKU")
        sku_label.setObjectName("FieldLabel")
        card_layout.addWidget(sku_label)
        
        sku_combo = QComboBox()
        sku_combo.setEditable(True)
        sku_combo.setInsertPolicy(QComboBox.NoInsert)
        sku_combo.lineEdit().setPlaceholderText("Search or select SKU...")
        sku_combo.setMinimumHeight(self.s(44))
        
        if self.available_skus:
            sku_combo.addItems(self.available_skus)
            completer = QCompleter(self.available_skus)
            completer.setCaseSensitivity(Qt.CaseInsensitive)
            completer.setFilterMode(Qt.MatchContains)
            sku_combo.setCompleter(completer)
            if self.selected_live_sku and self.selected_live_sku in self.available_skus:
                sku_combo.setCurrentText(self.selected_live_sku)
        
        card_layout.addWidget(sku_combo)
        
        tyre_label = QLabel("Tyre Number")
        tyre_label.setObjectName("FieldLabel")
        card_layout.addWidget(tyre_label)
        
        tyre_edit = QLineEdit()
        tyre_edit.setPlaceholderText("Enter tyre number / tyre name")
        tyre_edit.setText(self.selected_live_tyre_name or "195_65_R15")
        tyre_edit.setMinimumHeight(self.s(44))
        card_layout.addWidget(tyre_edit)
        
        card_layout.addStretch()
        
        info_badge = QLabel("ℹ️ Ensure tyre is properly positioned before starting")
        info_badge.setStyleSheet("""
            QLabel {
                font: 500 11px 'Segoe UI';
                color: #718096;
                background: #f7fafc;
                padding: 8px;
                border-radius: 8px;
            }
        """)
        info_badge.setAlignment(Qt.AlignCenter)
        card_layout.addWidget(info_badge)
        
        button_layout = QHBoxLayout()
        button_layout.setSpacing(16)
        
        cancel_btn = QPushButton("Cancel")
        cancel_btn.setObjectName("CancelBtn")
        cancel_btn.setCursor(Qt.PointingHandCursor)
        cancel_btn.clicked.connect(dialog.reject)
        
        start_btn = QPushButton("▶ Start Inspection")
        start_btn.setObjectName("StartBtn")
        start_btn.setCursor(Qt.PointingHandCursor)
        
        def proceed():
            sku_name = sku_combo.currentText().strip()
            tyre_name = tyre_edit.text().strip()
            
            if not sku_name or sku_name == "No SKU Available":
                QMessageBox.warning(dialog, "SKU Error", "Please select a valid SKU.")
                return
            
            if sku_name not in self.available_skus:
                reply = QMessageBox.question(
                    dialog, "New SKU Detected",
                    f"SKU '{sku_name}' not found in calibration.\n\nWould you like to create a new SKU?",
                    QMessageBox.Yes | QMessageBox.No
                )
                if reply == QMessageBox.Yes:
                    dialog.accept()
                    self.run_new_sku()
                return
            
            if not tyre_name:
                QMessageBox.warning(dialog, "Tyre Number", "Please enter tyre number.")
                return
            
            dialog.accept()
            self.update_live_info_cards(sku_name, tyre_name)
            self.begin_live_flow(sku_name, tyre_name)
        
        start_btn.clicked.connect(proceed)
        
        button_layout.addStretch()
        button_layout.addWidget(cancel_btn)
        button_layout.addWidget(start_btn)
        button_layout.addStretch()
        
        card_layout.addLayout(button_layout)
        root.addWidget(card)
        
        dialog.exec_()
    
    # REPLACE ENTIRE METHOD:
    def begin_live_flow(self, sku_name, tyre_name):
        """Start live inspection based on deployment mode"""
        self.update_live_info_cards(sku_name, tyre_name)
        
        if deployment == "True" and CAMERA_CAPTURE_ENABLED:
            # CAMERA MODE: Continuous PLC/Hardware monitored inspection
            self.start_continuous_inspection(sku_name, tyre_name)
        else:
            # DEMO MODE: Original single-cycle inspection
            if self.thread_manager.active_threads.get("inspection"):
                if self.thread_manager.active_threads["inspection"].isRunning():
                    QMessageBox.information(self, "Live Inspection", "Live inspection is already running.")
                    return
            
            if self.current_preloaded_sku == sku_name:
                self.start_live_inspection(sku_name=sku_name, tyre_name=tyre_name)
                return
            
            self.pending_live_start = True
            self.pending_live_sku = sku_name
            self.pending_live_tyre_name = tyre_name
            self.start_runtime_preload(sku_name=sku_name)

    def start_continuous_inspection(self, sku_name, tyre_name):
        """Start continuous PLC/Hardware monitored inspection"""
        if self.is_continuous_running:
            QMessageBox.information(self, "Live Inspection", "Continuous inspection already running.")
            return
        
        if self.multi_cam is None:
            QMessageBox.critical(self, "Camera Error", "Cameras not initialized.")
            return
        
        self.stop_continuous_inspection()
        
        self.continuous_worker = start_continuous_cycle(
            media_root=MEDIA_PATH,
            sku_name=sku_name,
            tyre_name=tyre_name,
            multi_camera_manager=self.multi_cam,
            plc_interface=plc_client,
            plc_trigger_tag="DB100.DBX0.0",
            min_capture_interval=2.0,
            seg_model_a_path=MAIN_SEG_MODEL_PATH,
            seg_model_b_path=MAIN_SEG_MODEL_PATH,
            vit_checkpoint_path=MAIN_VIT_CHECKPOINT_PATH,
            r_detector_path=MAIN_R_DETECTOR_PATH,
            device="cuda" if TORCH_GPU_OK else "cpu",
            auto_preload=True,
        )
        
        self.continuous_worker.status_update.connect(
            lambda msg: self.statusBar().showMessage(msg)
        )
        self.continuous_worker.processing_completed.connect(self._on_continuous_completed)
        self.continuous_worker.processing_error.connect(
            lambda err: logger.error(f"Continuous error: {err}")
        )
        
        self.thread_manager.start_thread(
            "continuous_cycle",
            self.continuous_worker,
            on_finished=lambda: setattr(self, 'is_continuous_running', False),
            on_error=lambda err: setattr(self, 'is_continuous_running', False)
        )
        
        self.is_continuous_running = True
        self.statusBar().showMessage(f"🔄 Continuous inspection | SKU={sku_name} | Monitoring trigger...")


    def stop_continuous_inspection(self):
        """Stop continuous inspection"""
        if self.continuous_worker:
            self.continuous_worker.stop()
            self.is_continuous_running = False


    def _on_continuous_completed(self, result):
        """Called when each AI pipeline cycle completes"""
        self._mark_ui_active()
        cycle_id = result.get('cycle_id', 'Unknown')
        final_label = result.get('final_label', 'Unknown')
        self.statusBar().showMessage(f"✅ {cycle_id} | Result: {final_label}")
        self.update_label_async()
        QTimer.singleShot(700, self.refresh_cycle_images_async)
    
    def start_runtime_preload(self, sku_name=None):
        sku_name = (sku_name or "").strip()
        if not sku_name:
            return
        
        # Check if already preloading
        preload_thread = self.thread_manager.active_threads.get("preload")
        if preload_thread and preload_thread.isRunning():
            return
        
        if self.current_preloaded_sku == sku_name:
            logger.info(f"[PRELOAD] Already warmed | SKU={sku_name}")
            if self.pending_live_start:
                self.pending_live_start = False
                sku_to_start = self.pending_live_sku
                tyre_to_start = self.pending_live_tyre_name
                self.pending_live_sku = None
                self.pending_live_tyre_name = None
                self.start_live_inspection(sku_name=sku_to_start, tyre_name=tyre_to_start)
            return
        
        self.pending_preload_sku = sku_name
        self.statusBar().showMessage(f"Preloading AI models for {sku_name}...")
        
        worker = RuntimePreloadWorker(
            media_root=MEDIA_PATH,
            sku_name=sku_name,
            device="cuda" if TORCH_GPU_OK else "cpu",
            seg_model_a_path=MAIN_SEG_MODEL_PATH,
            seg_model_b_path=MAIN_SEG_MODEL_PATH,
            vit_checkpoint_path=MAIN_VIT_CHECKPOINT_PATH,
            r_detector_path=MAIN_R_DETECTOR_PATH,
        )
        
        def on_finished(message):
            logger.info(f"[PRELOAD] {message}")
            if self.pending_preload_sku:
                self.current_preloaded_sku = self.pending_preload_sku
            self.statusBar().showMessage(f"Models loaded | SKU={self.current_preloaded_sku}")
            
            if self.pending_live_start:
                self.pending_live_start = False
                sku_name_to_start = self.pending_live_sku
                tyre_name_to_start = self.pending_live_tyre_name
                self.pending_live_sku = None
                self.pending_live_tyre_name = None
                self.start_live_inspection(sku_name=sku_name_to_start, tyre_name=tyre_name_to_start)
        
        def on_error(message):
            logger.error(f"[PRELOAD][ERROR] {message}")
            self.pending_live_start = False
            self.pending_live_sku = None
            self.pending_live_tyre_name = None
            self.statusBar().showMessage("Preload failed")
            QMessageBox.critical(self, "Preload Error", message)
        
        self.thread_manager.start_thread("preload", worker, on_finished, on_error)
    
    def start_live_inspection(self, sku_name=None, tyre_name=None):
        # Check if already running
        insp_thread = self.thread_manager.active_threads.get("inspection")
        if insp_thread and insp_thread.isRunning():
            QMessageBox.information(self, "Live Inspection", "Live inspection is already running.")
            return
        
        sku_name = (sku_name or self.selected_live_sku or "").strip()
        tyre_name = (tyre_name or self.selected_live_tyre_name or "195_65_R15").strip()
        
        if not sku_name:
            QMessageBox.critical(self, "SKU Error", "Please select a valid SKU.")
            return
        
        sku_calibration_dir = os.path.join(MEDIA_PATH, "calibration", sku_name)
        if not os.path.isdir(sku_calibration_dir):
            QMessageBox.critical(self, "Calibration Error", 
                               f"Selected SKU folder not found:\n{sku_calibration_dir}")
            return
        
        if not MAIN_SEG_MODEL_PATH or not os.path.isfile(MAIN_SEG_MODEL_PATH):
            QMessageBox.critical(self, "Model Error", f"Main model path invalid:\n{MAIN_SEG_MODEL_PATH}")
            return
        
        if not MAIN_VIT_CHECKPOINT_PATH or not os.path.isfile(MAIN_VIT_CHECKPOINT_PATH):
            QMessageBox.critical(self, "Model Error", f"VIT checkpoint path invalid:\n{MAIN_VIT_CHECKPOINT_PATH}")
            return
        
        if not MAIN_R_DETECTOR_PATH or not os.path.isfile(MAIN_R_DETECTOR_PATH):
            QMessageBox.critical(self, "Model Error", f"R detector path invalid:\n{MAIN_R_DETECTOR_PATH}")
            return
        
        if CAMERA_CAPTURE_ENABLED:
            if self.multi_cam is None:
                QMessageBox.critical(self, "Camera Error", 
                                   "Cameras not initialised. Check connections and restart.")
                return
            cam_mgr = self.multi_cam
            demo_capture_root = None
        else:
            cam_mgr = None
            demo_capture_root = LOCAL_MULTI_SIDE_TEST_FOLDER
            if not demo_capture_root or not os.path.isdir(demo_capture_root):
                QMessageBox.critical(self, "Path Error", 
                                   f"Demo capture folder not found:\n{demo_capture_root}")
                return
        
        self.statusBar().showMessage(f"Live Inspection Started | SKU={sku_name} | TYRE={tyre_name}")
        
        worker = LiveInspectionWorker(
            media_root=MEDIA_PATH,
            sku_name=sku_name,
            tyre_name=tyre_name,
            device="cuda" if TORCH_GPU_OK else "cpu",
            seg_model_a_path=MAIN_SEG_MODEL_PATH,
            seg_model_b_path=MAIN_SEG_MODEL_PATH,
            vit_checkpoint_path=MAIN_VIT_CHECKPOINT_PATH,
            r_detector_path=MAIN_R_DETECTOR_PATH,
            multi_camera_manager=cam_mgr,
            demo_capture_root=demo_capture_root,
        )
        
        def on_finished(result):
            self._mark_ui_active()
            try:
                sku = result.get('sku_name', 'Unknown') if isinstance(result, dict) else 'Unknown'
                tyre = result.get('tyre_name', 'Unknown') if isinstance(result, dict) else 'Unknown'
                self.statusBar().showMessage(f"Inspection finished | SKU={sku} | TYRE={tyre}")
            except Exception:
                self.statusBar().showMessage("Inspection completed")
            self.update_label_async()
            QTimer.singleShot(700, self.refresh_cycle_images_async)
        
        def on_error(message):
            self._mark_ui_active()
            self.statusBar().showMessage(f"Inspection failed: {message}")
            QMessageBox.critical(self, "Live Inspection Error", message)
        
        self.thread_manager.start_thread("inspection", worker, on_finished, on_error)
    
    # ========================================================================
    # REMAINING METHODS (unchanged but using thread_manager where applicable)
    # ========================================================================
    
    def on_preload_finished(self, message):
        logger.info(f"[PRELOAD] {message}")
        if self.pending_preload_sku:
            self.current_preloaded_sku = self.pending_preload_sku
        self.statusBar().showMessage(f"Models loaded and warmed | SKU={self.current_preloaded_sku}")
        if self.pending_live_start:
            self.pending_live_start = False
            sku_name = self.pending_live_sku
            tyre_name = self.pending_live_tyre_name
            self.pending_live_sku = None
            self.pending_live_tyre_name = None
            self.start_live_inspection(sku_name=sku_name, tyre_name=tyre_name)
    
    def on_preload_error(self, message):
        logger.error(f"[PRELOAD][ERROR] {message}")
        self.pending_live_start = False
        self.pending_live_sku = None
        self.pending_live_tyre_name = None
        self.statusBar().showMessage("Preload failed")
        QMessageBox.critical(self, "Preload Error", message)
    
    def on_live_inspection_finished(self, result):
        self.statusBar().showMessage(
            f"Inspection finished successfully | SKU={result.get('sku_name')} | TYRE={result.get('tyre_name')}"
        )
        self.update_label_async()
        QTimer.singleShot(700, self.refresh_cycle_images_async)
    
    def on_live_inspection_error(self, message):
        QMessageBox.critical(self, "Live Inspection Error", message)
    
    def open_new_sku_capture_page(self, sku_meta=None):
        if sku_meta is None:
            sku_meta = {
                "tyre_name": "", "barcode": "", "sku_name": "",
                "operator": "", "machine_serial": "",
            }
        if getattr(self, "new_sku_capture_page", None) is not None:
            self.new_sku_capture_page.set_sku_meta(sku_meta)
            self.content_stack.setCurrentWidget(self.new_sku_capture_page)
            if self.back_btn:
                self.back_btn.setVisible(True)
            return
        save_root = os.path.join(MEDIA_PATH, "NewSKU_Captures")
        self.new_sku_capture_page = NewSKUPage(
            media_path=MEDIA_PATH, raw_dir=RAW_IMAGE_DIR, save_root_dir=save_root,
            mydb=mydb, meta_collection="New SKU", gridfs_bucket="fs",
            sku_meta=sku_meta, on_close=self.handle_back_to_dashboard
        )
        self.content_stack.addWidget(self.new_sku_capture_page)
        self.content_stack.setCurrentWidget(self.new_sku_capture_page)
        if self.back_btn:
            self.back_btn.setVisible(True)
    
    def _go_dashboard_from_inner_pages(self):
        if self.content_stack:
            self.content_stack.setCurrentIndex(0)
        if self.back_btn:
            self.back_btn.setVisible(False)
    
    def setup_header_bar(self, parent_layout):
        header_frame = QFrame()
        header_frame.setStyleSheet("QFrame { background-color: white; border-radius: 8px; }")
        h = QHBoxLayout(header_frame)
        h.setContentsMargins(10, 5, 10, 5)
        h.setSpacing(8)
        
        cal_icon_path = os.path.join(MEDIA_PATH, "img", "calendar.png")
        clock_icon_path = os.path.join(MEDIA_PATH, "img", "clock.png")
        
        cal_pix = QPixmap(cal_icon_path)
        if not cal_pix.isNull():
            cal_pix = cal_pix.scaled(18, 18, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        clock_pix = QPixmap(clock_icon_path)
        if not clock_pix.isNull():
            clock_pix = clock_pix.scaled(18, 18, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        
        self.date_icon = QLabel()
        self.date_icon.setPixmap(cal_pix)
        self.date_label = QLabel()
        self.date_label.setStyleSheet("font: 13px 'Segoe UI'; color: black;")
        
        date_box = QWidget()
        date_layout = QHBoxLayout(date_box)
        date_layout.setContentsMargins(0, 0, 0, 0)
        date_layout.setSpacing(4)
        date_layout.addWidget(self.date_icon)
        date_layout.addWidget(self.date_label)
        
        self.time_icon = QLabel()
        self.time_icon.setPixmap(clock_pix)
        self.time_label = QLabel()
        self.time_label.setStyleSheet("font: 13px 'Segoe UI'; color: black;")
        
        time_box = QWidget()
        time_layout = QHBoxLayout(time_box)
        time_layout.setContentsMargins(0, 0, 0, 0)
        time_layout.setSpacing(4)
        time_layout.addWidget(self.time_icon)
        time_layout.addWidget(self.time_label)
        
        h.addWidget(date_box)
        h.addSpacing(10)
        h.addWidget(time_box)
        h.addStretch()
        
        self.back_btn = QToolButton()
        self.back_btn.setText("Back to Live")
        self.back_btn.setIcon(QIcon(os.path.join(MEDIA_PATH, "img/human-machine.png")))
        self.back_btn.setIconSize(QSize(self.s(18), self.s(18)))
        self.back_btn.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.back_btn.setStyleSheet("""
            QToolButton {
                font: 600 13px 'Segoe UI';
                color: #571c86;
                background-color: white;
                border: 1px solid #571c86;
                border-radius: 8px;
                padding: 5px 12px;
            }
            QToolButton:hover { background-color: #e6e6e6; }
        """)
        self.back_btn.clicked.connect(self.handle_back_to_dashboard)
        self.back_btn.setVisible(False)
        h.addWidget(self.back_btn)
        
        help_btn = QToolButton()
        help_btn.setText("Help")
        help_btn.setIcon(QIcon(os.path.join(MEDIA_PATH, "img/help.png")))
        help_btn.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        help_btn.setStyleSheet("""
            QToolButton {
                font: bold 14px 'Segoe UI';
                color: #571c86;
                background-color: white;
                border: 1px solid #571c86;
                border-radius: 8px;
                padding: 5px 10px;
            }
            QToolButton:hover { background-color: #e6e6e6; }
        """)
        help_btn.clicked.connect(self.open_help_doc)
        h.addWidget(help_btn)
        
        exit_btn = QToolButton()
        exit_btn.setText("Exit")
        exit_btn.setIcon(QIcon(os.path.join(MEDIA_PATH, "img/Logout1.png")))
        exit_btn.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        exit_btn.setStyleSheet("""
            QToolButton {
                font: bold 14px 'Segoe UI';
                color: #571c86;
                background-color: white;
                border: 1px solid #571c86;
                border-radius: 8px;
                padding: 5px 15px;
            }
            QToolButton:hover { background-color: #e6e6e6; }
        """)
        exit_btn.clicked.connect(self.stop_server)
        h.addWidget(exit_btn)
        
        parent_layout.addWidget(header_frame)
    
    def handle_back_to_dashboard(self):
        if self.content_stack is not None:
            self.content_stack.setCurrentIndex(0)
        if self.back_btn:
            self.back_btn.setVisible(False)
    
    def setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        root_layout = QHBoxLayout(central_widget)
        root_layout.setContentsMargins(10, 10, 10, 10)
        root_layout.setSpacing(15)
        self.setup_sidebar(root_layout)
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(10)
        self.setup_header_bar(right_layout)
        self.setup_main_content(right_layout)
        root_layout.addWidget(right_widget, 1)
    
    def setup_sidebar(self, main_layout):
        sidebar = QFrame()
        sidebar.setFixedWidth(self.s(210))
        sidebar.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        sidebar.setStyleSheet("QFrame { background-color: #571c86; border-radius: 30px; }")
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(8, 8, 8, 8)
        sidebar_layout.setSpacing(8)
        
        logo_label = QLabel()
        logo_pixmap = QPixmap(os.path.join(MEDIA_PATH, "img/Apollo_white-removebg-preview.png"))
        scaled_logo = logo_pixmap.scaled(200, 55, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        logo_label.setPixmap(scaled_logo)
        logo_label.setAlignment(Qt.AlignCenter)
        sidebar_layout.addWidget(logo_label)
        
        button_style = """
            QPushButton {
                font: bold 13px 'Segoe UI';
                color: #ffffff;
                background-color: rgba(255, 255, 255, 25);
                background-image: qlineargradient(
                    x1:0, y1:0, x2:0, y2:1,
                    stop:0   rgba(255, 255, 255, 120),
                    stop:0.45 rgba(255, 255, 255, 40),
                    stop:1   rgba(255, 255, 255, 10)
                );
                border-radius: 18px;
                border: 1px solid rgba(255, 255, 255, 140);
                padding: 4px 6px;
                min-height: 30px;
            }
            QPushButton:hover {
                background-color: rgba(255, 255, 255, 60);
                background-image: qlineargradient(
                    x1:0, y1:0, x2:0, y2:1,
                    stop:0   rgba(255, 255, 255, 200),
                    stop:0.4 rgba(255, 255, 255, 80),
                    stop:1   rgba(255, 255, 255, 30)
                );
                border: 1px solid rgba(255, 255, 255, 230);
            }
            QPushButton:pressed {
                background-color: rgba(255, 255, 255, 40);
                padding-top: 5px;
                padding-bottom: 3px;
            }
        """
        
        buttons = [
            ("Test Mode", "test_mode.png", self.open_test_popup),
            ("Live", "run_smart_qc.png", self.capture_image),
            ("Run New SKU", "run_new_sku.png", self.run_new_sku),
            ("Repeatability", "repeatability.png", self.open_repeatability_page),
            ("Action Code Plan", "action_code_plan.png", self.open_action_code_plan),
            ("Dashboard", "dashboard.png", self.open_dashboard),
            ("Annotation Tool", "annotation_tool.png", self.open_annotation_tool),
        ]
        
        def load_square_icon(icon_path: str, size: int = 18) -> QIcon:
            pm = QPixmap(icon_path)
            if pm.isNull():
                return QIcon()
            pm = pm.scaled(size, size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            canvas = QPixmap(size, size)
            canvas.fill(Qt.transparent)
            p = QPainter(canvas)
            x = (size - pm.width()) // 2
            y = (size - pm.height()) // 2
            p.drawPixmap(x, y, pm)
            p.end()
            return QIcon(canvas)
        
        for text, icon_name, slot in buttons:
            btn = QPushButton(text)
            btn.setLayoutDirection(Qt.LeftToRight)
            btn.setCursor(Qt.PointingHandCursor)
            btn.setFixedHeight(self.s(38))
            icon_path = os.path.join(MEDIA_PATH, "img", icon_name)
            if os.path.exists(icon_path):
                btn.setIcon(load_square_icon(icon_path, 18))
                btn.setIconSize(QSize(18, 18))
            btn.setStyleSheet(button_style + """
                QPushButton {
                    text-align: left;
                    padding-left: 14px;
                    padding-right: 10px;
                }
            """)
            btn.clicked.connect(slot)
            sidebar_layout.addWidget(btn)
        
        tyre_label = QLabel("TYRE INSPECTED")
        tyre_label.setStyleSheet("font: bold 12px 'Segoe UI'; color: #FFFFFF;")
        tyre_label.setAlignment(Qt.AlignCenter)
        sidebar_layout.addWidget(tyre_label)
        
        date_label = QLabel(datetime.today().strftime("%d/%m/%Y"))
        date_label.setStyleSheet("font: 11px 'Segoe UI'; color: #E8E0FF;")
        date_label.setAlignment(Qt.AlignCenter)
        sidebar_layout.addWidget(date_label)
        
        self.label_count = QLabel("0")
        self.label_count.setStyleSheet("""
            QLabel {
                font: bold 34px 'Segoe UI';
                color: #FFFFFF;
                background-color: #3A2FA3;
                border-radius: 16px;
                padding: 8px;
            }
        """)
        self.label_count.setAlignment(Qt.AlignCenter)
        self.label_count.setFixedHeight(self.s(70))
        sidebar_layout.addWidget(self.label_count)
        
        sidebar_layout.addStretch()
        
        bottom_logo = QLabel()
        bottom_pixmap = QPixmap(os.path.join(MEDIA_PATH, "img/Radome-removebg-preview.png"))
        scaled_bottom = bottom_pixmap.scaled(150, 60, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        bottom_logo.setPixmap(scaled_bottom)
        bottom_logo.setAlignment(Qt.AlignCenter)
        sidebar_layout.addWidget(bottom_logo)
        
        main_layout.addWidget(sidebar)
    
    def setup_main_content(self, main_layout):
        self.content_stack = QStackedWidget()
        
        dashboard_widget = QWidget()
        dashboard_widget.setStyleSheet("background-color: #f5f5f5;")
        content_layout = QHBoxLayout(dashboard_widget)
        content_layout.setSpacing(10)
        
        image_frame = QFrame()
        image_frame.setStyleSheet("background-color: #f5f5f5;")
        image_layout = QVBoxLayout(image_frame)
        
        center_frame = QFrame()
        center_frame.setStyleSheet("background-color: #f5f5f5;")
        center_layout = QVBoxLayout(center_frame)
        
        info_frame = QFrame()
        info_frame.setStyleSheet("background-color: #f5f5f5;")
        info_layout = QHBoxLayout(info_frame)
        info_layout.setSpacing(self.s(10))
        
        def _make_info_card(title):
            card = QFrame()
            card.setMinimumHeight(self.s(88))
            card.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            card.setStyleSheet("QFrame { background-color: white; border-radius: 10px; }")
            card_layout = QVBoxLayout(card)
            card_layout.setContentsMargins(10, 8, 10, 8)
            title_label = QLabel(title)
            title_label.setStyleSheet("font: bold 13px 'Arial';")
            title_label.setAlignment(Qt.AlignLeft)
            card_layout.addWidget(title_label)
            return card, card_layout
        
        sku_card, sku_layout = _make_info_card("Selected SKU")
        self.selected_sku_value_label = QLabel("--")
        self.selected_sku_value_label.setStyleSheet("""
            QLabel {
                font: bold 13px 'Arial';
                color: #0056D2;
                background-color: white;
                border: 1px solid #d0d0d0;
                border-radius: 8px;
                padding: 8px 10px;
            }
        """)
        self.selected_sku_value_label.setMinimumHeight(self.s(38))
        self.selected_sku_value_label.setAlignment(Qt.AlignVCenter | Qt.AlignLeft)
        sku_layout.addWidget(self.selected_sku_value_label)
        info_layout.addWidget(sku_card)
        
        tyre_card, tyre_layout = _make_info_card("Tyre Number")
        self.selected_tyre_value_label = QLabel("--")
        self.selected_tyre_value_label.setStyleSheet("""
            QLabel {
                font: bold 13px 'Arial';
                color: #0056D2;
                background-color: white;
                border: 1px solid #d0d0d0;
                border-radius: 8px;
                padding: 8px 10px;
            }
        """)
        self.selected_tyre_value_label.setMinimumHeight(self.s(38))
        self.selected_tyre_value_label.setAlignment(Qt.AlignVCenter | Qt.AlignLeft)
        tyre_layout.addWidget(self.selected_tyre_value_label)
        info_layout.addWidget(tyre_card)
        
        barcode_card, barcode_layout = _make_info_card("Bar Code Image")
        barcode_img = self.get_latest_image(BAR_CODE_DIR)
        
        img_label = QLabel()
        img_label.setAlignment(Qt.AlignCenter)
        img_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        img_label.setMinimumHeight(self.s(42))
        img_label.setMaximumHeight(self.s(42))
        img_label.setStyleSheet("background-color: white; border: none;")
        
        if barcode_img:
            pixmap = QPixmap(barcode_img)
            scaled_pixmap = pixmap.scaled(self.s(220), self.s(42), Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
            img_label.setPixmap(scaled_pixmap)
        else:
            img_label.setText("No Barcode")
            img_label.setStyleSheet(f"font: bold {self.s(12)}px 'Arial'; color: grey; background-color: white; border: none;")
        
        barcode_layout.addWidget(img_label)
        info_layout.addWidget(barcode_card)
        center_layout.addWidget(info_frame)
        
        self.images_row = QFrame()
        self.images_row.setStyleSheet("background-color: #f5f5f5;")
        self.images_layout = QHBoxLayout(self.images_row)
        
        self.startup_image_paths = {
            "sidewall1": STARTUP_IMAGE_PATHS[0],
            "sidewall2": STARTUP_IMAGE_PATHS[1],
            "innerwall": STARTUP_IMAGE_PATHS[2],
            "tread": STARTUP_IMAGE_PATHS[3],
            "bead": STARTUP_IMAGE_PATHS[4],
        }
        
        self.image_labels_dict = {}
        
        for index, (side_key, title_text) in enumerate(self.side_order):
            card = QFrame()
            card.setMinimumWidth(self.s(150))
            card.setMinimumHeight(self.s(400))
            card.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            card.setStyleSheet("QFrame { background-color: white; border-radius: 10px; }")
            card_layout = QVBoxLayout(card)
            card_layout.setContentsMargins(5, 5, 5, 5)
            card_layout.setSpacing(6)
            
            title_label = QLabel(title_text)
            title_label.setStyleSheet("font: bold 11px 'Arial'; color: #400080;")
            title_label.setAlignment(Qt.AlignCenter)
            card_layout.addWidget(title_label)
            
            img_label = QLabel()
            img_label.setAlignment(Qt.AlignCenter)
            img_label.setMinimumHeight(self.s(360))
            img_label.setStyleSheet("QLabel { background-color: white; border-radius: 8px; }")
            img_label.setText("🖼️")
            img_label.setCursor(Qt.PointingHandCursor)
            card_layout.addWidget(img_label, 1)
            
            self.images_layout.addWidget(card)
            self.image_labels_dict[index] = img_label
            self.image_labels_by_side[side_key] = img_label
            self.current_panel_image_paths[side_key] = None
        
        center_layout.addWidget(self.images_row)
        image_layout.addWidget(center_frame)
        content_layout.addWidget(image_frame, 4)
        
        defect_frame = QFrame()
        defect_frame.setFixedWidth(280)
        defect_frame.setStyleSheet("QFrame { background-color: white; border-radius: 20px; }")
        defect_layout = QVBoxLayout(defect_frame)
        defect_layout.setContentsMargins(10, 10, 10, 10)
        
        defect_title = QLabel("Defect Normal")
        defect_title.setStyleSheet("font: bold 15px 'Arial';")
        defect_title.setAlignment(Qt.AlignCenter)
        defect_layout.addWidget(defect_title)
        
        defects = [
            {"name": "Tread blister", "area": "1mm", "code": "-", "category": "OE"},
            {"name": "Tread lightness", "area": "3mm", "code": "-", "category": "Replacement"}
        ]
        
        for defect in defects:
            dcard = QFrame()
            dcard.setStyleSheet("QFrame { background-color: #F8F8F8; border-radius: 10px; }")
            dcard_layout = QVBoxLayout(dcard)
            dcard_layout.setContentsMargins(10, 6, 10, 6)
            
            name_label = QLabel(defect["name"])
            name_label.setStyleSheet("font: bold 13px 'Arial';")
            dcard_layout.addWidget(name_label)
            
            area_label = QLabel(f"Defect Area: {defect['area']}")
            area_label.setStyleSheet("font: 11px 'Arial';")
            dcard_layout.addWidget(area_label)
            
            code_label = QLabel(f"Action Code: {defect['code']}")
            code_label.setStyleSheet("font: 11px 'Arial';")
            dcard_layout.addWidget(code_label)
            
            category_label = QLabel(f"Category: {defect['category']}")
            category_label.setStyleSheet("font: 11px 'Arial';")
            dcard_layout.addWidget(category_label)
            
            defect_layout.addWidget(dcard)
        
        defect_layout.addStretch()
        content_layout.addWidget(defect_frame, 1)
        
        self.content_stack.addWidget(dashboard_widget)
        
        self.test_mode_page = TestModePage(
            reports_dir=TEST_MODE_REPORTS,
            expected_serials=["244802149","244802163","251102086","251401655","251300826","251500640"],
            on_close=lambda: self._go_dashboard_from_inner_pages(),
            media_path=MEDIA_PATH
        )
        self.content_stack.addWidget(self.test_mode_page)
        
        main_layout.addWidget(self.content_stack, 4)
    
    def darken_color(self, color):
        if color.startswith('#'):
            r = int(color[1:3], 16) * 0.8
            g = int(color[3:5], 16) * 0.8
            b = int(color[5:7], 16) * 0.8
            return f'#{int(r):02x}{int(g):02x}{int(b):02x}'
        return color
    
    def get_latest_image(self, directory):
        if not os.path.exists(directory):
            return None
        files = [os.path.join(directory, f) for f in os.listdir(directory) if f.lower().endswith((".jpg", ".png"))]
        return max(files, key=os.path.getctime) if files else None
    
    def update_datetime(self):
        self._mark_ui_active()
        now = datetime.now()
        self.date_label.setText(now.strftime("%d/%m/%Y"))
        self.time_label.setText(now.strftime("%I:%M %p"))
    
    def update_marquee_text(self):
        base_len = len(self.copy_full_text)
        view = self.copy_padded_text[self.copy_index:self.copy_index + base_len]
        self.copyright_label.setText(view)
        self.copy_index += 1
        if self.copy_index + base_len >= len(self.copy_padded_text):
            self.copy_index = 0
    
    def load_startup_images(self):
        for side_key, _title in self.side_order:
            img_label = self.image_labels_by_side.get(side_key)
            img_path = self.startup_image_paths.get(side_key)
            if side_key == "bead" and img_path is None:
                img_path = STARTUP_IMAGE_PATHS[4]
            if not img_label:
                continue
            if img_path and os.path.exists(img_path):
                ok = self.set_label_image_safe(img_label, img_path, max(self.s(220), 220), max(self.s(640), 640), keep_aspect=True)
                if ok:
                    self.current_panel_image_paths[side_key] = img_path
                    img_label.mousePressEvent = self._make_open_image_handler(side_key)
            else:
                img_label.setText("🖼️")
                img_label.setStyleSheet("font: 48px; color: grey; background-color: white;")
    
    def _make_open_image_handler(self, side_key):
        def handler(event):
            img_path = self.current_panel_image_paths.get(side_key)
            if img_path and os.path.exists(img_path):
                self.open_full_image(img_path)
        return handler
    
    def open_full_image(self, image_path):
        viewer = ImageViewer(image_path)
        viewer.exec_()
    
    def capture_image(self):
        self.open_live_selection_dialog()
    
    def run_new_sku(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("SKU Processing")
        dialog.resize(self.s(460), self.s(200))
        dialog.setWindowIcon(QIcon(os.path.join(MEDIA_PATH, "img/smartQC-.ico")))
        dialog.setStyleSheet("""
            QDialog { background: #f4f4f4; }
            QFrame#Card { background: white; border-radius: 16px; border: 1px solid #e6e6e6; }
            QLabel#Title { font: 900 14px 'Segoe UI'; color: #222; }
            QLabel#Msg { font: 700 12px 'Segoe UI'; color: #444; }
            QPushButton#Primary {
                min-height: 36px; border-radius: 12px; border: none;
                background: #571c86; color: white; font: 900 12px 'Segoe UI'; padding: 0 18px;
            }
            QPushButton#Primary:hover { background: #6b2aa3; }
            QPushButton#Ghost {
                min-height: 36px; border-radius: 12px; border: 1px solid #cfcfcf;
                background: #f7f7f7; color: #222; font: 800 12px 'Segoe UI'; padding: 0 18px;
            }
            QPushButton#Ghost:hover { background: #eeeeee; }
        """)
        
        root = QVBoxLayout(dialog)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(10)
        
        card = QFrame()
        card.setObjectName("Card")
        cl = QVBoxLayout(card)
        cl.setContentsMargins(18, 16, 18, 16)
        cl.setSpacing(12)
        
        title = QLabel("Instruction")
        title.setObjectName("Title")
        title.setAlignment(Qt.AlignCenter)
        cl.addWidget(title)
        
        msg = QLabel("Run at least 1 good tyre for 20 times.\nThen click OK to continue.")
        msg.setObjectName("Msg")
        msg.setAlignment(Qt.AlignCenter)
        msg.setWordWrap(True)
        cl.addWidget(msg)
        cl.addSpacing(6)
        
        btnrow = QHBoxLayout()
        btnrow.setSpacing(12)
        
        cancel_btn = QPushButton("Cancel")
        cancel_btn.setObjectName("Ghost")
        cancel_btn.setCursor(Qt.PointingHandCursor)
        cancel_btn.clicked.connect(dialog.reject)
        
        ok_btn = QPushButton("OK")
        ok_btn.setObjectName("Primary")
        ok_btn.setCursor(Qt.PointingHandCursor)
        ok_btn.clicked.connect(lambda: (dialog.accept(), self.open_new_sku_capture_page()))
        
        btnrow.addWidget(cancel_btn, 1)
        btnrow.addWidget(ok_btn, 1)
        cl.addLayout(btnrow)
        
        root.addWidget(card)
        dialog.exec_()
    
    def open_repeatability_page(self):
        if getattr(self, "repeatability_page", None) is None:
            save_root = os.path.join(MEDIA_PATH, "Repeatability_Captures")
            self.repeatability_page = RepeatabilityPage(
                media_path=MEDIA_PATH, raw_dir=RAW_IMAGE_DIR,
                save_root_dir=save_root, on_close=self.handle_back_to_dashboard
            )
            self.content_stack.addWidget(self.repeatability_page)
        self.repeatability_page.reset_page(refresh_preview=True)
        self.content_stack.setCurrentWidget(self.repeatability_page)
        if self.back_btn:
            self.back_btn.setVisible(True)
    
    def open_dashboard(self):
        try:
            if getattr(self, "dashboard_cards_page", None) is None:
                self.dashboard_cards_page = ApolloDashboardCardsWidget(parent=self)
                self.content_stack.addWidget(self.dashboard_cards_page)
            self.content_stack.setCurrentWidget(self.dashboard_cards_page)
            if self.back_btn:
                self.back_btn.setVisible(True)
        except Exception as e:
            QMessageBox.critical(self, "Dashboard Error", f"Failed to open dashboard:\n{e}")
    
    def open_test_popup(self):
        if self.content_stack and self.test_mode_page:
            self.content_stack.setCurrentWidget(self.test_mode_page)
        if self.back_btn:
            self.back_btn.setVisible(True)
    
    def get_latest_image_from_folder(self, folder_path):
        if not os.path.isdir(folder_path):
            return None
        valid_exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
        files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.lower().endswith(valid_exts)]
        if not files:
            return None
        files.sort(key=os.path.getmtime, reverse=True)
        return files[0]
    
    def open_annotation_tool(self):
        try:
            if getattr(self, "annotation_tool_page", None) is None:
                self.annotation_tool_page = AnnotationTool(media_path=MEDIA_PATH)
                self.annotation_tool_page.setWindowFlags(Qt.Widget)
                self.content_stack.addWidget(self.annotation_tool_page)
            self.content_stack.setCurrentWidget(self.annotation_tool_page)
            if self.back_btn:
                self.back_btn.setVisible(True)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to open annotation tool: {e}")
    
    def open_help_doc(self):
        help_file = os.path.join(MEDIA_PATH, "Guide", "Edge_App_GUI_Operating_Document.docx")
        if platform.system() == "Windows":
            os.startfile(help_file)
        elif platform.system() == "Darwin":
            subprocess.call(["open", help_file])
        else:
            subprocess.call(["xdg-open", help_file])
    
    def stop_server(self):
        reply = QMessageBox.question(self, 'Exit', 'Are you sure you want to exit?',
                                    QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.close()
    
    def open_action_code_plan(self):
        if getattr(self, "action_plan_page", None) is None:
            self.action_plan_page = ActionCodePlanPage()
            self.content_stack.addWidget(self.action_plan_page)
        self.content_stack.setCurrentWidget(self.action_plan_page)
        if self.back_btn:
            self.back_btn.setVisible(True)
    

    def closeEvent(self, event):
        logger.info("Application closing - cleaning up...")
        try:
            self.stop_continuous_inspection()  
            self.update_timer.stop()
            self.update_label_timer.stop()
            self.update_images_timer.stop()
            self.copy_timer.stop()
            self._freeze_monitor.stop()
            self.thread_manager.stop_all(timeout=3000)
        except Exception as e:
            logger.error(f"Cleanup error: {e}")
        finally:
            event.accept()

# ============================================================================
# CLEANUP AND MAIN
# ============================================================================
def cleanup_camera_resources():
    global multi_cam
    logger.info("Cleaning up resources...")
    try:
        if multi_cam is not None:
            multi_cam.close_all()
            multi_cam = None
            logger.info("Camera system closed successfully.")
    except Exception as e:
        logger.error(f"Error during camera cleanup: {e}")
    try:
        disconnect_plc()
    except Exception as e:
        logger.error(f"Error during PLC cleanup: {e}")
    logger.info("Application cleanup complete.")

def signal_handler(signum, frame):
    """Handle system signals for graceful shutdown"""
    logger.info(f"Received signal {signum}, shutting down...")
    QApplication.quit()

def main():
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    app.aboutToQuit.connect(cleanup_camera_resources)
    
    main_window = MainWindow()
    main_window.show()
    
    return_code = app.exec_()
    cleanup_camera_resources()
    sys.exit(return_code)

if __name__ == "__main__":
    main()