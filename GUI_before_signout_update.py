# Import required libraries
import sys
import os
import signal
from datetime import datetime
from threading import Lock
import pandas as pd # type: ignore
from PIL import Image, ImageTk # type: ignore
import torch # type: ignore
import warnings
warnings.filterwarnings("ignore")
import threading
from PyQt5.QtWidgets import (
    QApplication, QDialog, QFrame, QHBoxLayout, QLineEdit, QLabel,
    QMainWindow, QMessageBox, QPushButton, QSizePolicy, QStackedWidget,
    QStatusBar, QToolButton, QVBoxLayout, QWidget,
)
from PyQt5.QtCore import QSize, QTimer, Qt, pyqtSignal, QEvent
from PyQt5.QtGui import QGuiApplication, QIcon, QPainter, QPixmap
from src.COMMON.db import get_db
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import subprocess
import platform
from src.COMMON.config import get_config, get_config_manager
from src.COMMON.structured_logging import (
    configure_logging,
    get_logger,
    install_global_exception_hooks,
    shutdown_logging,
)
from src.COMMON.security import (
    ALL_PERMISSIONS,
    Permission,
    Role,
    SessionContext,
    UserPrincipal,
    get_security_service,
)
from src.Pages.login_window import LoginWindow
from src.Pages.user_management_page import UserManagementPage
from src.Pages.test_mode_page import TestModePage
from src.Pages.new_sku_page import NewSKUPage
from src.Pages.repeatability_page import RepeatabilityPage
from src.Pages.action_code_plan_page import ActionCodePlanPage
from src.Pages.dashboard import ApolloDashboardCardsWidget
from src.Pages.annotation_tool import AnnotationTool  
from pathlib import Path
from snap7 import Client # type: ignore

####Local files imports
from src.Main_cam import CAMERA_CAPTURE_ENABLED, start_continuous_cycle
from src.COMMON.system_check import show_startup_system_popup
from src.COMMON.full_hardware_check import is_hardware_ready, get_hardware_state
from src.COMMON.component_health_service import ComponentHealthService
from src.UI.component_health_ui import apply_component_health_to_gui
from src.Pages.axis_status_page import AxisStatusPage
from src.COMMON.live_inspection_state import (
    reset_live_progress,
    set_live_progress,
    get_live_progress,
)
from src.UI.live_progress_ui import (
    create_live_progress_widget,
    apply_live_progress_to_gui,
)
from src.COMMON.live_result_state import (
    reset_live_result,
    update_live_result_from_cycle_result,
    set_live_result_plc_output,
    set_live_result_failed,
)
from src.Pages.recipe_management_page import RecipeManagementPage
from src.COMMON.plc_result_sender import send_tyre_result_to_plc
from src.Pages.device_page import DevicePage
from src.UI.live_result_ui import (
    create_tyre_result_summary_widget,
    apply_tyre_result_to_gui,
)
from src.UI.gui_helpers import (
    get_available_sku_names,
    ThreadManager,
    RuntimePreloadWorker,
    LiveInspectionWorker,
    LatestCycleImagesWorker,
    ImageViewer,
)
from src.Pages.capture_settings_tab import CameraCaptureSettingsTab
from src.COMMON.live_sku_resolver import resolve_live_sku_from_plc
from src.COMMON.plc_gui_commands import PlcGuiCommandService
from src.Pages.roi_px_mm_tool import MainWindow as RoiMeasurementWindow, DARK_STYLE as ROI_DARK_STYLE
os.environ['CUDA_MODULE_LOADING'] = 'LAZY'
os.environ["QT_DEVICE_PIXEL_RATIO"] = "0"
os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"
os.environ["QT_SCALE_FACTOR"] = "1"

# Configure centralized queue-based structured logging before application startup.
ACTIVE_LOG_PATHS = configure_logging()
install_global_exception_hooks()
logger = get_logger(__name__, component="GUI")


def permission_required(permission: Permission):
    """Protect a MainWindow action even when called outside the sidebar."""
    def decorate(function):
        def wrapped(self, *args, **kwargs):
            # QPushButton.clicked emits a bool (checked). Protected page-opening
            # methods do not consume that signal argument. Drop it safely.
            if len(args) == 1 and isinstance(args[0], bool) and not kwargs:
                args = ()
            if not self._require_permission(permission, function.__name__):
                return None
            return function(self, *args, **kwargs)
        wrapped.__name__ = function.__name__
        wrapped.__doc__ = function.__doc__
        return wrapped
    return decorate

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

# Load all application settings through the central configuration service.
config_manager = get_config_manager(ENV_PATH)
app_config = config_manager.config
env_vars = config_manager.as_legacy_dict()  # temporary bridge for unmigrated GUI fields
deployment = app_config.deployment_mode
plc_ip = app_config.plc.ip
logger.info(
    "Apollo application configuration loaded",
    extra={
        "event_code": "APP_CONFIG_LOADED",
        "details": {
            "deployment": deployment,
            "version": app_config.application.version,
            "log_paths": ACTIVE_LOG_PATHS,
        },
    },
)

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

MAIN_SEG_MODEL_PATH = (
    str(app_config.models.segmentation_weight)
    if app_config.models.segmentation_weight else None
)
MAIN_R_DETECTOR_PATH = (
    str(app_config.models.r_detector_onnx)
    if app_config.models.r_detector_onnx else None
)
MAIN_VIT_CHECKPOINT_PATH = (
    str(app_config.models.vit_checkpoint)
    if app_config.models.vit_checkpoint else None
)

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



# Runtime objects are loaded only after Live SKU selection
shared_r_detector_onnx = None
shared_r_detector_onnx_path = None

logger.info("Startup completed without AI model loading.")

multi_cam = None
plc_client = None

logger.info("Startup completed without PLC/camera/laser initialization.")


# ============================================================================
# MAIN WINDOW
# ============================================================================

class MainWindow(QMainWindow):
    sign_out_requested = pyqtSignal()
    application_exit_requested = pyqtSignal()
    tyre_count_ready = pyqtSignal(int)
    # Throttle constants
    REFRESH_MIN_INTERVAL = 1.0  # Minimum seconds between image refreshes
    
    def __init__(self, session: SessionContext):
        super().__init__()
        self.security_service = get_security_service()
        self.session = session
        self._session_close_reason = "APPLICATION_EXIT"
        self.setWindowTitle('EyresAi QC+')
        self.back_btn = None
        screen = QGuiApplication.primaryScreen().availableGeometry()
        self.screen_w = screen.width()
        self.screen_h = screen.height()

        # Do not upscale UI above 1.0. This avoids oversized minimum layouts.
        self.ui_scale = min(self.screen_w / 1920.0, self.screen_h / 1080.0, 1.0)

        # Use available screen geometry, not full monitor geometry.
        self.setGeometry(screen)

        # Let Windows/Qt maximize within usable area.
        QTimer.singleShot(0, self.showMaximized)
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
        self.axis_status_page = None
        self.device_page = None
        self.recipe_management_page = None
        self.capture_settings_page = None
        self.user_management_page = None
        self.roi_measurement_window = None
        self._last_stack_widget = None
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
        self.plc_gui_command_service = PlcGuiCommandService(
            env_path=ENV_PATH,
            parent=self,
        )
        # Setup UI
        self.setup_ui()
        app_instance = QApplication.instance()
        if app_instance is not None:
            app_instance.installEventFilter(self)
        self.session_timer = QTimer(self)
        self.session_timer.timeout.connect(self._check_session_timeout)
        self.session_timer.start(15000)
        self.tyre_count_ready.connect(self._set_tyre_count_label, Qt.QueuedConnection)
        QTimer.singleShot(300, self.update_label_async)
        self.load_startup_images()
        QTimer.singleShot(
            800,
            lambda: show_startup_system_popup(
                self,
                MEDIA_PATH,
                ENV_PATH,
            )
        )
        self.available_skus = get_available_sku_names(MEDIA_PATH)
        self.selected_live_sku = ""
        self.selected_live_tyre_name = ""
        self.pending_live_start = False
        self.pending_live_sku = None
        self.pending_live_tyre_name = None

        # Lightweight Live Page component health service
        self.component_health_service = ComponentHealthService(
            media_path=MEDIA_PATH,
            env_path=ENV_PATH,
        )
        
        # Initial delayed refresh
        QTimer.singleShot(1200, self.refresh_cycle_images_async)
        QTimer.singleShot(3000, self.refresh_cycle_images_async)
        # Start timers
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_datetime)
        self.update_timer.start(1000)
        
        self.update_label_timer = QTimer()
        self.update_label_timer.timeout.connect(self.update_label_async)
        self.update_label_timer.start(5000)
        
        self.update_images_timer = QTimer(self)
        self.update_images_timer.timeout.connect(self.refresh_cycle_images_async)
        # self.update_images_timer.start(3000)
        
        # UI freeze monitor (for debugging)
        self._freeze_monitor = QTimer(self)
        self._freeze_monitor.timeout.connect(self._check_ui_responsiveness)
        self._freeze_monitor.start(3000)

        # Lightweight component health monitor
        # Safe during inspection: no reconnect, no camera configure, no AI loading
        self.health_timer = QTimer(self)
        self.health_timer.timeout.connect(self.refresh_component_health)
        self.health_timer.start(app_config.health.monitor_interval_ms)

        # Lightweight live inspection progress monitor
        # Reads only memory state, no hardware access
        self.live_progress_timer = QTimer(self)
        self.live_progress_timer.timeout.connect(
            lambda: apply_live_progress_to_gui(self)
        )
        self.live_progress_timer.start(500)

        reset_live_progress(total_images=len(self.side_order))
        QTimer.singleShot(1000, lambda: apply_live_progress_to_gui(self))
        reset_live_result()
        QTimer.singleShot(1000, lambda: apply_tyre_result_to_gui(self))

        QTimer.singleShot(1500, self.refresh_component_health)
        
        # ---------- BOTTOM MARQUEE COPYRIGHT BAR ----------
        status_bar = QStatusBar()
        status_bar.setStyleSheet("background-color: white;")
        status_bar.setSizeGripEnabled(False)
        self.setStatusBar(status_bar)
        self.plc_gui_command_service.started.connect(
            lambda name, address: self.statusBar().showMessage(
                f"{name}: sending pulse to {address}..."
            )
        )

        self.plc_gui_command_service.success.connect(
            lambda name, address: self.statusBar().showMessage(
                f"{name}: pulse completed at {address}"
            )
        )

        self.plc_gui_command_service.error.connect(
            lambda name, address, msg: QMessageBox.critical(
                self,
                "PLC Command Error",
                f"{name} failed at {address}\n\n{msg}",
            )
        )

        self.plc_gui_command_service.busy_changed.connect(
            lambda busy: (
                getattr(self, "auto_start_btn", None)
                and self.auto_start_btn.setEnabled(not busy),
                getattr(self, "servo_reset_btn", None)
                and self.servo_reset_btn.setEnabled(not busy),
            )
        )
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
            logger.warning(f"UI may be frozen! Last update: {time_since_update:.1f}s ago")
        self._last_ui_update = current_time
    
    def _mark_ui_active(self):
        """Mark that UI thread is alive"""
        self._last_ui_update = time.time()
    
    def refresh_component_health(self):
        """
        Lightweight Live Page health refresh.

        Safe during inspection:
        - no camera reconnect
        - no camera reconfigure
        - no laser reconnect
        - no AI loading
        """

        try:
            if not hasattr(self, "component_health_service"):
                return

            inspection_running = bool(
                getattr(self, "is_continuous_running", False)
                or (
                    self.thread_manager.active_threads.get("inspection")
                    and self.thread_manager.active_threads["inspection"].isRunning()
                )
            )

            health = self.component_health_service.get_health(
                inspection_running=inspection_running
            )

            apply_component_health_to_gui(self, health)

        except Exception as e:
            logger.debug(f"[HEALTH] refresh failed: {e}")
    # ========================================================================
    # ASYNC LABEL UPDATE (Non-blocking DB query)
    # ========================================================================

    def _set_tyre_count_label(self, cnt):
        try:
            if hasattr(self, "label_count") and self.label_count:
                self.label_count.setText(str(cnt))
        except Exception as e:
            logger.warning(f"Failed to update tyre count label: {e}")


    def update_label_async(self):
        """
        Count today's inspected tyres from MongoDB and update sidebar safely.
        """

        def worker():
            try:
                currentdate = datetime.now().strftime("%d-%m-%Y")

                cnt = tyre_details_col.count_documents(
                    {"inspectionDate": currentdate},
                    maxTimeMS=3000
                )

                # logger.info(f"[TYRE COUNT] date={currentdate} count={cnt}")

            except Exception as e:
                logger.warning(f"MongoDB count unavailable: {e}")
                cnt = 0

            self.tyre_count_ready.emit(int(cnt))

        threading.Thread(target=worker, daemon=True).start()

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
    
    def refresh_cycle_images_async(self, cycle_dir_override=None):
        """Throttled async image refresh"""
        current_time = time.time()

        # Throttle only normal refresh.
        # Do not throttle forced refresh after completed AI cycle.
        if cycle_dir_override is None:
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
                sku_name=self.selected_live_sku,
                cycle_dir_override=cycle_dir_override,
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
    
    @permission_required(Permission.ROI_MEASURE)
    def open_roi_measurement_tool(self):
        """
        Opens the ROI + 4-point px/mm measurement tool as a separate PyQt window.
        Uses the existing QApplication. Do NOT create another QApplication here.
        """
        try:
            if self.roi_measurement_window is None:
                self.roi_measurement_window = RoiMeasurementWindow()
                self.roi_measurement_window.setStyleSheet(ROI_DARK_STYLE)

                # Keep object alive after close/show again
                self.roi_measurement_window.setAttribute(Qt.WA_DeleteOnClose, False)

            self.roi_measurement_window.show()
            self.roi_measurement_window.raise_()
            self.roi_measurement_window.activateWindow()

        except Exception as e:
            QMessageBox.critical(
                self,
                "ROI Measurement Tool Error",
                f"Failed to open ROI measurement tool:\n\n{e}"
            )
    # ========================================================================
    # LIVE INSPECTION WITH THREAD SAFETY    
    # ========================================================================
    
    def open_live_selection_dialog(self):
        """
        Live flow:
        - Test Mode must already be completed.
        - Read active recipe number from PLC.
        - Map recipe number to SKU_XXX.
        - User enters only tyre number.
        - Start Live with resolved SKU.
        """

        if str(deployment) == "True" and not is_hardware_ready():
            QMessageBox.warning(
                self,
                "Test Mode Required",
                "Please open Test Mode and complete the Full Hardware Check before starting Live Inspection.",
            )
            return

        hardware_state = get_hardware_state()
        plc_client_from_test = hardware_state.get("plc_client")

        try:
            resolved = resolve_live_sku_from_plc(
                plc_client=plc_client_from_test,
                media_path=MEDIA_PATH,
                env_path=ENV_PATH,
            )
        except Exception as e:
            QMessageBox.critical(
                self,
                "Active Recipe Error",
                f"Could not resolve SKU from PLC active recipe.\n\n{e}",
            )
            return

        recipe_number = resolved["recipe_number"]
        sku_name = resolved["sku_name"]
        plc_tag = resolved["tag"]

        dialog = QDialog(self)
        dialog.setWindowTitle("Live Inspection")
        dialog.resize(self.s(520), self.s(330))
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
            }
            QLabel#FieldLabel {
                font: 600 13px 'Segoe UI';
                color: #4a5568;
            }
            QLabel#SkuBadge {
                background: #f3e8ff;
                color: #571c86;
                border-radius: 12px;
                padding: 12px;
                font: 800 13px 'Segoe UI';
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
            QPushButton#CancelBtn {
                min-height: 44px;
                background: white;
                border: 2px solid #e2e8f0;
                border-radius: 12px;
                font: 600 13px 'Segoe UI';
                color: #4a5568;
                padding: 0 24px;
            }
            QPushButton#StartBtn {
                min-height: 44px;
                background: #571c86;
                border: none;
                border-radius: 12px;
                font: 600 13px 'Segoe UI';
                color: white;
                padding: 0 32px;
            }
        """)

        root = QVBoxLayout(dialog)
        root.setContentsMargins(20, 20, 20, 20)

        card = QFrame()
        card.setObjectName("Card")
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(28, 28, 28, 28)
        card_layout.setSpacing(16)

        title_label = QLabel("Start Live Inspection")
        title_label.setObjectName("Title")
        title_label.setAlignment(Qt.AlignCenter)
        card_layout.addWidget(title_label)

        sku_badge = QLabel(
            f"PLC Active Recipe: {recipe_number}\n"
            f"PLC Tag: {plc_tag}\n"
            f"Resolved AI SKU: {sku_name}"
        )
        sku_badge.setObjectName("SkuBadge")
        sku_badge.setAlignment(Qt.AlignCenter)
        card_layout.addWidget(sku_badge)

        tyre_label = QLabel("Tyre Number")
        tyre_label.setObjectName("FieldLabel")
        card_layout.addWidget(tyre_label)

        tyre_edit = QLineEdit()
        tyre_edit.setPlaceholderText("Enter tyre number / tyre name")
        tyre_edit.setText(self.selected_live_tyre_name or "")
        tyre_edit.setMinimumHeight(self.s(44))
        card_layout.addWidget(tyre_edit)

        info_badge = QLabel("AI files will be loaded based on PLC active recipe.")
        info_badge.setAlignment(Qt.AlignCenter)
        info_badge.setStyleSheet("""
            QLabel {
                font: 500 11px 'Segoe UI';
                color: #718096;
                background: #f7fafc;
                padding: 8px;
                border-radius: 8px;
            }
        """)
        card_layout.addWidget(info_badge)

        button_layout = QHBoxLayout()

        cancel_btn = QPushButton("Cancel")
        cancel_btn.setObjectName("CancelBtn")
        cancel_btn.clicked.connect(dialog.reject)

        start_btn = QPushButton("▶ Load & Prepare")
        start_btn.setObjectName("StartBtn")

        def proceed():
            tyre_name = tyre_edit.text().strip()

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
    
    def validate_selected_sku_calibration(self, sku_name):
        sku_calibration_dir = os.path.join(
            MEDIA_PATH,
            "AI_Calibration_Files",
            sku_name,
        )

        if not os.path.isdir(sku_calibration_dir):
            QMessageBox.critical(
                self,
                "Calibration Error",
                f"Selected SKU folder not found:\n{sku_calibration_dir}",
            )
            return False

        # required_side_dirs = [
        #     "calibration_sidewall1",
        #     "calibration_sidewall2",
        #     "calibration_innerwall",
        #     "calibration_tread",
        #     "calibration_bead",
        # ]

        required_side_dirs = [
            "calibration_sidewall1",
            "calibration_sidewall2",
            "calibration_tread",
        ]

        missing = []

        for side_dir_name in required_side_dirs:
            artifacts_dir = os.path.join(
                sku_calibration_dir,
                side_dir_name,
                "artifacts",
            )

            if not os.path.isdir(artifacts_dir):
                missing.append(artifacts_dir)

        if missing:
            QMessageBox.critical(
                self,
                "Calibration Error",
                "Selected SKU calibration is incomplete.\n\nMissing:\n"
                + "\n".join(missing[:8]),
            )
            return False

        return True
    
    # REPLACE ENTIRE METHOD:
    def begin_live_flow(self, sku_name, tyre_name):
        """Start live inspection based on deployment mode"""
        self.update_live_info_cards(sku_name, tyre_name)

        if not self.validate_selected_sku_calibration(sku_name):
            return
        if str(deployment) == "True" and not is_hardware_ready():
            QMessageBox.warning(
                self,
                "Test Mode Required",
                "Please open Test Mode and complete the Full Hardware Check before starting Live Inspection."
            )
            return

        hardware_state = get_hardware_state()

        if hardware_state.get("multi_cam") is not None:
            self.multi_cam = hardware_state.get("multi_cam")

        globals()["multi_cam"] = hardware_state.get("multi_cam")
        globals()["plc_client"] = hardware_state.get("plc_client")

        if str(deployment) == "True" and CAMERA_CAPTURE_ENABLED:
            self.start_continuous_inspection(sku_name, tyre_name)
        else:
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
        
        active_sides = ["sidewall1", "sidewall2", "tread"]
        capture_sides = [
            "sidewall1",
            "sidewall2",
            "innerwall",
            "tread",
            # "bead",
        ]

        self.continuous_worker = start_continuous_cycle(
            media_root=MEDIA_PATH,
            sku_name=sku_name,
            tyre_name=tyre_name,
            multi_camera_manager=self.multi_cam,
            min_capture_interval=2.0,
            seg_model_a_path=MAIN_SEG_MODEL_PATH,
            seg_model_b_path=MAIN_SEG_MODEL_PATH,
            vit_checkpoint_path=MAIN_VIT_CHECKPOINT_PATH,
            r_detector_path=MAIN_R_DETECTOR_PATH,
            device="cuda" if TORCH_GPU_OK else "cpu",
            auto_preload=True,
            sides_to_run=active_sides,
            capture_sides=capture_sides,
        )
        def on_ready_for_inspection(message):
            QMessageBox.information(
                self,
                "Ready for Inspection",
                message,
            )

            if self.continuous_worker:
                self.continuous_worker.confirm_ready_to_start()

        self.continuous_worker.ready_for_inspection.connect(on_ready_for_inspection)
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
        reset_live_result()
        apply_tyre_result_to_gui(self)

        reset_live_progress(total_images=len(active_sides))
        set_live_progress(
            phase="WAITING",
            active_zone="-",
            images_captured=0,
            total_images=len(active_sides),
            message="Continuous inspection started. Waiting for trigger.",
        )
        apply_live_progress_to_gui(self)

        self.statusBar().showMessage(f"🔄 Continuous inspection | SKU={sku_name} | Monitoring trigger...")


    def stop_continuous_inspection(self):
        """Stop continuous inspection gracefully"""
        try:
            if self.continuous_worker:
                self.continuous_worker.stop()

            # Extra safety: directly stop camera manager also
            if self.multi_cam is not None:
                if hasattr(self.multi_cam, "_stop_event"):
                    self.multi_cam._stop_event.set()

                if hasattr(self.multi_cam, "stop_all_streams"):
                    threading.Thread(
                        target=self.multi_cam.stop_all_streams,
                        daemon=True,
                    ).start()

        except Exception as e:
            logger.warning(f"[EXIT] continuous inspection stop warning: {e}")

        self.is_continuous_running = False

        set_live_progress(
            phase="WAITING",
            active_zone="-",
            images_captured=0,
            total_images=len(self.side_order),
            message="Continuous inspection stopped",
        )
        apply_live_progress_to_gui(self)


    def _on_continuous_completed(self, result):
        """Called when each AI pipeline cycle completes"""
        self._mark_ui_active()

        cycle_id = result.get('cycle_id', 'Unknown')
        final_label = result.get('final_label', 'Unknown')

        set_live_progress(
            phase="COMPLETED",
            active_zone="All Zones",
            images_captured=len(self.side_order),
            total_images=len(self.side_order),
            message=f"Cycle completed: {final_label}",
        )
        apply_live_progress_to_gui(self)

        summary = update_live_result_from_cycle_result(
            result,
            total_zones=len(self.side_order),
        )

        plc_status = send_tyre_result_to_plc(
            summary.get("final_result", "WAITING"),
            env_path=ENV_PATH,
        )

        set_live_result_plc_output(plc_status.get("display", "Not Sent"))
        apply_tyre_result_to_gui(self)

        self.statusBar().showMessage(f"✅ {cycle_id} | Result: {final_label}")
        self.update_label_async()

        cycle_output_dir = (
            result.get("cycle_dir")
            or result.get("output_dir")
        )

        # Force GUI image panels to reload the exact completed cycle output.
        self.latest_loaded_cycle_dir = None
        self.current_panel_image_paths = {}
        self._last_refresh_time = 0

        cycle_output_dir = result.get("cycle_dir") or result.get("output_dir")

        self.latest_loaded_cycle_dir = None
        self.current_panel_image_paths = {}
        self._last_refresh_time = 0

        if cycle_output_dir and os.path.isdir(cycle_output_dir):
            QTimer.singleShot(
                300,
                lambda d=cycle_output_dir: self.refresh_cycle_images_async(
                    cycle_dir_override=d
                ),
            )
            QTimer.singleShot(
                1200,
                lambda d=cycle_output_dir: self.refresh_cycle_images_async(
                    cycle_dir_override=d
                ),
            )
        else:
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
            QMessageBox.information(
                self,
                "Live Inspection",
                "Live inspection is already running."
            )
            return

        sku_name = (sku_name or self.selected_live_sku or "").strip()
        tyre_name = (tyre_name or self.selected_live_tyre_name or "195_65_R15").strip()

        if not sku_name:
            QMessageBox.critical(
                self,
                "SKU Error",
                "Please select a valid SKU."
            )
            return

        # NOTE:
        # SKU calibration folder validation is already done in begin_live_flow()
        # using self.validate_selected_sku_calibration(sku_name).
        # So do not repeat media/AI_Calibration_Files/<SKU> validation here.

        if not MAIN_SEG_MODEL_PATH or not os.path.isfile(MAIN_SEG_MODEL_PATH):
            QMessageBox.critical(
                self,
                "Model Error",
                f"Main model path invalid:\n{MAIN_SEG_MODEL_PATH}"
            )
            return

        if not MAIN_VIT_CHECKPOINT_PATH or not os.path.isfile(MAIN_VIT_CHECKPOINT_PATH):
            QMessageBox.critical(
                self,
                "Model Error",
                f"VIT checkpoint path invalid:\n{MAIN_VIT_CHECKPOINT_PATH}"
            )
            return

        if not MAIN_R_DETECTOR_PATH or not os.path.isfile(MAIN_R_DETECTOR_PATH):
            QMessageBox.critical(
                self,
                "Model Error",
                f"R detector path invalid:\n{MAIN_R_DETECTOR_PATH}"
            )
            return

        if CAMERA_CAPTURE_ENABLED:
            if self.multi_cam is None:
                QMessageBox.critical(
                    self,
                    "Camera Error",
                    "Cameras not initialised. Check connections and restart."
                )
                return

            cam_mgr = self.multi_cam
            demo_capture_root = None

        else:
            cam_mgr = None
            demo_capture_root = LOCAL_MULTI_SIDE_TEST_FOLDER

            if not demo_capture_root or not os.path.isdir(demo_capture_root):
                QMessageBox.critical(
                    self,
                    "Path Error",
                    f"Demo capture folder not found:\n{demo_capture_root}"
                )
                return

        reset_live_progress(total_images=len(self.side_order))
        set_live_progress(
            phase="CAPTURING",
            active_zone="All Zones",
            images_captured=0,
            total_images=len(self.side_order),
            message="Live inspection started",
        )
        apply_live_progress_to_gui(self)

        reset_live_result()
        apply_tyre_result_to_gui(self)

        self.statusBar().showMessage(
            f"Live Inspection Started | SKU={sku_name} | TYRE={tyre_name}"
        )

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
                sku = result.get("sku_name", "Unknown") if isinstance(result, dict) else "Unknown"
                tyre = result.get("tyre_name", "Unknown") if isinstance(result, dict) else "Unknown"
                self.statusBar().showMessage(
                    f"Inspection finished | SKU={sku} | TYRE={tyre}"
                )
            except Exception:
                self.statusBar().showMessage("Inspection completed")

            set_live_progress(
                phase="COMPLETED",
                active_zone="All Zones",
                images_captured=len(self.side_order),
                total_images=len(self.side_order),
                message="Inspection completed",
            )
            apply_live_progress_to_gui(self)

            summary = update_live_result_from_cycle_result(
                result,
                total_zones=len(self.side_order),
            )

            plc_status = send_tyre_result_to_plc(
                summary.get("final_result", "WAITING"),
                env_path=ENV_PATH,
            )

            set_live_result_plc_output(plc_status.get("display", "Not Sent"))
            apply_tyre_result_to_gui(self)

            self.update_label_async()
            QTimer.singleShot(700, self.refresh_cycle_images_async)

        def on_error(message):
            self._mark_ui_active()

            self.statusBar().showMessage(f"Inspection failed: {message}")

            set_live_progress(
                phase="FAILED",
                active_zone="-",
                message=message,
            )
            apply_live_progress_to_gui(self)

            set_live_result_failed(message)
            apply_tyre_result_to_gui(self)

            QMessageBox.critical(
                self,
                "Live Inspection Error",
                message,
            )

        self.thread_manager.start_thread(
            "inspection",
            worker,
            on_finished,
            on_error,
        )
    
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
    
    @permission_required(Permission.SKU_MANAGE)
    def open_new_sku_capture_page(self, sku_meta=None):
        if sku_meta is None:
            sku_meta = {
                "tyre_name": "",
                "barcode": "",
                "sku_name": "",
                "operator": "",
            }

        # Get the already-connected PLC client from Test Mode hardware check
        hardware_state = get_hardware_state()
        shared_plc_client = hardware_state.get("plc_client")
        shared_multi_cam = hardware_state.get("multi_cam")

        # If New SKU page already exists, reuse same page
        if getattr(self, "new_sku_capture_page", None) is not None:
            self.new_sku_capture_page.set_sku_meta(sku_meta)

            if hasattr(self.new_sku_capture_page, "set_plc_client"):
                self.new_sku_capture_page.set_plc_client(shared_plc_client)

            if hasattr(self.new_sku_capture_page, "set_multi_camera_manager"):
                self.new_sku_capture_page.set_multi_camera_manager(shared_multi_cam)

            self.content_stack.setCurrentWidget(self.new_sku_capture_page)

            if self.back_btn:
                self.back_btn.setVisible(True)

            return

        save_root = os.path.join(MEDIA_PATH, "NewSKU_Captures")

        self.new_sku_capture_page = NewSKUPage(
            media_path=MEDIA_PATH,
            raw_dir=RAW_IMAGE_DIR,
            save_root_dir=save_root,
            mydb=mydb,
            meta_collection="New SKU",
            gridfs_bucket="fs",
            sku_meta=sku_meta,
            on_close=self.handle_back_to_dashboard,
            plc_client=shared_plc_client,   # important
            multi_camera_manager=shared_multi_cam,
        )

        self.content_stack.addWidget(self.new_sku_capture_page)
        self.content_stack.setCurrentWidget(self.new_sku_capture_page)

        if self.back_btn:
            self.back_btn.setVisible(True)
    
    def _go_dashboard_from_inner_pages(self):
        try:
            if getattr(self, "axis_status_page", None) is not None:
                if hasattr(self.axis_status_page, "stop_refresh"):
                    self.axis_status_page.stop_refresh()
        except Exception:
            pass

        if self.content_stack:
            self.content_stack.setCurrentIndex(0)

        if self.back_btn:
            self.back_btn.setVisible(False)

        try:
            self.refresh_component_health()
        except Exception:
            pass

        QTimer.singleShot(200, self.refresh_component_health)
    
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

        self.live_system_status_label = QLabel("System: NOT READY")
        self.live_system_status_label.setAlignment(Qt.AlignCenter)
        self.live_system_status_label.setStyleSheet("""
            QLabel {
                background: #eeeeee;
                color: #333333;
                border-radius: 10px;
                padding: 6px 12px;
                font: 800 12px 'Segoe UI';
            }
        """)
        h.addWidget(self.live_system_status_label)

        self.mode_indicator_label = QLabel("Mode: UNKNOWN")
        self.mode_indicator_label.setAlignment(Qt.AlignCenter)
        self.mode_indicator_label.setStyleSheet("""
            QLabel {
                background: #eeeeee;
                color: #333333;
                border-radius: 10px;
                padding: 6px 12px;
                font: 800 12px 'Segoe UI';
            }
        """)
        h.addWidget(self.mode_indicator_label)

        def make_header_command_button(text, bg, hover):
            b = QToolButton()
            b.setText(text)
            b.setCursor(Qt.PointingHandCursor)
            b.setToolButtonStyle(Qt.ToolButtonTextOnly)
            b.setStyleSheet(f"""
                QToolButton {{
                    font: 800 12px 'Segoe UI';
                    color: white;
                    background-color: {bg};
                    border: none;
                    border-radius: 8px;
                    padding: 6px 12px;
                }}
                QToolButton:hover {{
                    background-color: {hover};
                }}
                QToolButton:disabled {{
                    background-color: #999999;
                    color: #eeeeee;
                }}
            """)
            return b

        self.auto_start_btn = make_header_command_button(
            "Auto Start",
            "#198754",
            "#157347",
        )
        self.auto_start_btn.clicked.connect(
            self._guarded_slot(
                Permission.PLC_AUTO_START,
                self.plc_gui_command_service.pulse_auto_start,
                "PLC Auto Start",
            )
        )
        self.auto_start_btn.setVisible(self._has_permission(Permission.PLC_AUTO_START))
        h.addWidget(self.auto_start_btn)

        self.servo_reset_btn = make_header_command_button(
            "All Servo Reset",
            "#dc3545",
            "#bb2d3b",
        )
        self.servo_reset_btn.clicked.connect(
            self._guarded_slot(
                Permission.PLC_SERVO_RESET,
                self.plc_gui_command_service.pulse_all_servo_reset,
                "All Servo Reset",
            )
        )
        self.servo_reset_btn.setVisible(self._has_permission(Permission.PLC_SERVO_RESET))
        h.addWidget(self.servo_reset_btn)
                
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
        
        self.user_identity_label = QLabel(
            f"{self.session.user.full_name}  •  {self.session.user.role.value.replace('_', ' ').title()}"
        )
        self.user_identity_label.setStyleSheet("""
            QLabel {
                color:#334155; background:#F1F5F9; border:1px solid #CBD5E1;
                border-radius:8px; padding:6px 10px; font:700 11px 'Segoe UI';
            }
        """)
        h.addWidget(self.user_identity_label)

        logout_btn = QToolButton()
        logout_btn.setText("Sign Out")
        logout_btn.setToolButtonStyle(Qt.ToolButtonTextOnly)
        logout_btn.setStyleSheet("""
            QToolButton {
                font:700 12px 'Segoe UI'; color:#9F1239; background:#FFF1F2;
                border:1px solid #FECDD3; border-radius:8px; padding:6px 10px;
            }
            QToolButton:hover { background:#FFE4E6; }
        """)
        logout_btn.clicked.connect(self.logout_user)
        h.addWidget(logout_btn)

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

    def _on_content_stack_changed(self, index):
        """
        If user leaves Device page by clicking Back, Live, Dashboard, Axis Status, etc.,
        stop Device page camera streams and release camera handles.
        """

        try:
            new_widget = self.content_stack.widget(index)

            old_widget = getattr(self, "_last_stack_widget", None)

            if old_widget is not None:
                if old_widget is getattr(self, "device_page", None):
                    if hasattr(old_widget, "cleanup_device_page"):
                        old_widget.cleanup_device_page(destroy_devices=True)

            self._last_stack_widget = new_widget

        except Exception as e:
            logger.warning(f"[DEVICE PAGE] stack cleanup failed: {e}")
    
    def handle_back_to_dashboard(self):
        try:
            if getattr(self, "axis_status_page", None) is not None:
                if hasattr(self.axis_status_page, "stop_refresh"):
                    self.axis_status_page.stop_refresh()
        except Exception:
            pass

        try:
            if getattr(self, "device_page", None) is not None:
                if hasattr(self.device_page, "cleanup_device_page"):
                    self.device_page.cleanup_device_page(destroy_devices=True)
        except Exception as e:
            logger.warning(f"[DEVICE PAGE] back cleanup failed: {e}")

        if self.content_stack is not None:
            self.content_stack.setCurrentIndex(0)

        if self.back_btn:
            self.back_btn.setVisible(False)
    
    def _has_permission(self, permission: Permission | str) -> bool:
        value = permission.value if isinstance(permission, Permission) else str(permission)
        return self.session.user.has_permission(value)

    def _require_permission(self, permission: Permission | str, action: str) -> bool:
        if self._has_permission(permission):
            return True
        value = permission.value if isinstance(permission, Permission) else str(permission)
        self.security_service.record_permission_denied(self.session.user, value, action)
        QMessageBox.warning(
            self,
            "Access denied",
            f"Your role ({self.session.user.role.value.replace('_', ' ').title()}) "
            f"does not have permission for this function.\n\nRequired permission: {value}",
        )
        return False

    def _guarded_slot(self, permission: Permission, callback, action_name: str):
        def invoke(*_args, **_kwargs):
            if self._require_permission(permission, action_name):
                callback()
        return invoke

    def eventFilter(self, watched, event):
        if event.type() in (
            QEvent.MouseButtonPress,
            QEvent.KeyPress,
            QEvent.Wheel,
            QEvent.TouchBegin,
        ):
            self.session.touch()
        return super().eventFilter(watched, event)

    def _check_session_timeout(self):
        if not self.session.expired:
            return
        self._session_close_reason = "SESSION_TIMEOUT"
        logger.warning(
            "User session expired",
            extra={
                "event_code": "AUTH_SESSION_EXPIRED",
                "user_id": self.session.user.user_id,
                "status": "EXPIRED",
                "details": {"username": self.session.user.username},
            },
        )
        QMessageBox.warning(
            self,
            "Session expired",
            "The session expired because there was no user activity. The application will close. "
            "Start it again to sign in.",
        )
        self.close()

    def logout_user(self):
        reply = QMessageBox.question(
            self,
            "Sign out",
            f"Sign out {self.session.user.full_name} and close the application?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply == QMessageBox.Yes:
            self._session_close_reason = "USER_LOGOUT"
            self.close()

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
            ("Test Mode", "test_mode.png", self.open_test_popup, Permission.HARDWARE_TEST),
            ("Live", "run_smart_qc.png", self.capture_image, Permission.INSPECTION_RUN),
            ("Device", "cam.png", self.open_device_page, Permission.DEVICE_CONFIGURE),
            ("Capture", "cam.png", self.open_capture_settings_page, Permission.CAPTURE_CONFIGURE),
            ("Axis Status", "motor.png", self.open_axis_status_page, Permission.AXIS_VIEW),
            ("Run New SKU", "run_new_sku.png", self.run_new_sku, Permission.SKU_MANAGE),
            ("Recipe Management", "recipe.png", self.open_recipe_management_page, Permission.RECIPE_MANAGE),
            ("Repeatability", "repeatability.png", self.open_repeatability_page, Permission.REPEATABILITY_RUN),
            ("OSC Page", "action_code_plan.png", self.open_action_code_plan, Permission.OSC_MANAGE),
            ("Dashboard", "dashboard.png", self.open_dashboard, Permission.DASHBOARD_VIEW),
            ("Annotation Tool", "annotation_tool.png", self.open_annotation_tool, Permission.ANNOTATION_USE),
            ("ROI Measure", "cam.png", self.open_roi_measurement_tool, Permission.ROI_MEASURE),
            ("User Management", "human-machine.png", self.open_user_management_page, Permission.USER_MANAGE),
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
        
        self.sidebar_buttons = {}
        for text, icon_name, slot, permission in buttons:
            if not self._has_permission(permission):
                continue
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
            btn.clicked.connect(lambda _checked=False, callback=slot: callback())
            self.sidebar_buttons[permission.value] = btn
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
        self.content_stack.currentChanged.connect(self._on_content_stack_changed)
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
        
        barcode_card, barcode_layout = _make_info_card("Bar Code Num")
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
        # ------------------------------------------------------------------
        # LIVE INSPECTION PROGRESS BELOW IMAGE PANELS
        # ------------------------------------------------------------------
        self.live_progress_widget = create_live_progress_widget(self)
        center_layout.addWidget(self.live_progress_widget)
        image_layout.addWidget(center_frame)
        content_layout.addWidget(image_frame, 4)
        
        # ------------------------------------------------------------------
        # RIGHT PANEL: Component Health + Defect Info
        # ------------------------------------------------------------------
        right_panel = QFrame()
        right_panel.setFixedWidth(300)
        right_panel.setStyleSheet("QFrame { background-color: white; border-radius: 20px; }")
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(10, 10, 10, 10)
        right_layout.setSpacing(10)

        # ---------------- Component Health ----------------
        health_title = QLabel("Component Health")
        health_title.setStyleSheet("font: bold 15px 'Arial';")
        health_title.setAlignment(Qt.AlignCenter)
        right_layout.addWidget(health_title)

        self.health_labels = {}

        def make_health_row(key, title):
            row = QFrame()
            row.setStyleSheet("""
                QFrame {
                    background-color: #F8F8F8;
                    border-radius: 8px;
                }
            """)
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(8, 5, 8, 5)
            row_layout.setSpacing(6)

            name_lbl = QLabel(title)
            name_lbl.setStyleSheet("font: bold 11px 'Segoe UI'; color:#222;")
            row_layout.addWidget(name_lbl)

            row_layout.addStretch()

            status_lbl = QLabel("● Not checked")
            status_lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            status_lbl.setStyleSheet("font: bold 10px 'Segoe UI'; color:#666;")
            row_layout.addWidget(status_lbl)

            self.health_labels[key] = status_lbl
            right_layout.addWidget(row)

        make_health_row("plc", "PLC")
        make_health_row("cameras", "Cameras")
        make_health_row("laser", "Laser")
        make_health_row("gpu", "GPU")
        make_health_row("storage", "Storage")
        make_health_row("app_ok", "App OK")

        # separator
        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet("background-color:#e6e6e6;")
        right_layout.addWidget(sep)

        # ---------------- Tyre Result Summary / F-025 ----------------
        self.tyre_result_summary_widget = create_tyre_result_summary_widget(self)
        right_layout.addWidget(self.tyre_result_summary_widget)

        # separator
        sep_result = QFrame()
        sep_result.setFrameShape(QFrame.HLine)
        sep_result.setStyleSheet("background-color:#e6e6e6;")
        right_layout.addWidget(sep_result)

        # ---------------- Defect Info ----------------
        defect_title = QLabel("Defect Info")
        defect_title.setStyleSheet("font: bold 15px 'Arial';")
        defect_title.setAlignment(Qt.AlignCenter)
        right_layout.addWidget(defect_title)

        self.defect_info_container = QWidget()
        self.defect_info_layout = QVBoxLayout(self.defect_info_container)
        self.defect_info_layout.setContentsMargins(0, 0, 0, 0)
        self.defect_info_layout.setSpacing(8)

        # Initial/default defect cards
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

            self.defect_info_layout.addWidget(dcard)

        right_layout.addWidget(self.defect_info_container)
        right_layout.addStretch()

        content_layout.addWidget(right_panel, 1)
        
        self.content_stack.addWidget(dashboard_widget)
        
        self.test_mode_page = TestModePage(
            reports_dir=TEST_MODE_REPORTS,
            expected_serials=None,
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
    
    @permission_required(Permission.INSPECTION_RUN)
    def capture_image(self):
        self.open_live_selection_dialog()
    
    @permission_required(Permission.SKU_MANAGE)
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
    
    @permission_required(Permission.REPEATABILITY_RUN)
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
    
    @permission_required(Permission.DASHBOARD_VIEW)
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
    
    @permission_required(Permission.HARDWARE_TEST)
    def open_test_popup(self):
        if self.content_stack and self.test_mode_page:
            self.content_stack.setCurrentWidget(self.test_mode_page)
        if self.back_btn:
            self.back_btn.setVisible(True)

    @permission_required(Permission.CAPTURE_CONFIGURE)
    def open_capture_settings_page(self):
        try:
            # Stop Axis Status refresh if it is running
            try:
                if getattr(self, "axis_status_page", None) is not None:
                    if hasattr(self.axis_status_page, "stop_refresh"):
                        self.axis_status_page.stop_refresh()
            except Exception:
                pass

            # Create Capture Settings page only once
            if getattr(self, "capture_settings_page", None) is None:
                self.capture_settings_page = CameraCaptureSettingsTab(parent=self)
                self.content_stack.addWidget(self.capture_settings_page)

            # Show Capture Settings page
            self.content_stack.setCurrentWidget(self.capture_settings_page)

            if self.back_btn:
                self.back_btn.setVisible(True)

        except Exception as e:
            QMessageBox.critical(
                self,
                "Capture Settings Error",
                f"Failed to open Capture Settings page:\n{e}"
            )
    @permission_required(Permission.DEVICE_CONFIGURE)
    def open_device_page(self):
        try:
            # Stop axis status refresh if it was running
            try:
                if getattr(self, "axis_status_page", None) is not None:
                    if hasattr(self.axis_status_page, "stop_refresh"):
                        self.axis_status_page.stop_refresh()
            except Exception:
                pass

            # Create Device page only once
            if getattr(self, "device_page", None) is None:
                self.device_page = DevicePage(parent=self)
                self.content_stack.addWidget(self.device_page)

            # Show Device page
            self.content_stack.setCurrentWidget(self.device_page)

            if self.back_btn:
                self.back_btn.setVisible(True)

        except Exception as e:
            QMessageBox.critical(
                self,
                "Device Page Error",
                f"Failed to open Device page:\n{e}"
            )

    @permission_required(Permission.AXIS_VIEW)
    def open_axis_status_page(self):
        try:
            if getattr(self, "axis_status_page", None) is None:
                self.axis_status_page = AxisStatusPage(
                    media_path=MEDIA_PATH,
                    env_path=ENV_PATH,
                    on_close=self._go_dashboard_from_inner_pages,
                    parent=self,
                )
                self.content_stack.addWidget(self.axis_status_page)

            self.content_stack.setCurrentWidget(self.axis_status_page)

            if hasattr(self.axis_status_page, "start_refresh"):
                self.axis_status_page.start_refresh()

            if self.back_btn:
                self.back_btn.setVisible(True)

        except Exception as e:
            QMessageBox.critical(
                self,
                "Axis Status Error",
                f"Failed to open Axis Status page:\n{e}"
            )
    
    @permission_required(Permission.RECIPE_MANAGE)
    def open_recipe_management_page(self):
        try:
            # Stop Axis Status refresh if it is running
            try:
                if getattr(self, "axis_status_page", None) is not None:
                    if hasattr(self.axis_status_page, "stop_refresh"):
                        self.axis_status_page.stop_refresh()
            except Exception:
                pass

            if getattr(self, "recipe_management_page", None) is None:
                self.recipe_management_page = RecipeManagementPage(
                    media_path=MEDIA_PATH,
                    env_path=ENV_PATH,
                    on_close=self._go_dashboard_from_inner_pages,
                    on_edit_recipe=self.open_new_sku_capture_page,
                    parent=self,
                )
                self.content_stack.addWidget(self.recipe_management_page)

            if hasattr(self.recipe_management_page, "refresh_recipes"):
                self.recipe_management_page.refresh_recipes()

            self.content_stack.setCurrentWidget(self.recipe_management_page)

            if self.back_btn:
                self.back_btn.setVisible(True)

        except Exception as e:
            QMessageBox.critical(
                self,
                "Recipe Management Error",
                f"Failed to open Recipe Management page:\n{e}"
            )
    
    @permission_required(Permission.USER_MANAGE)
    def open_user_management_page(self, _checked=False):
        try:
            if self.user_management_page is None:
                self.user_management_page = UserManagementPage(
                    service=self.security_service,
                    session=self.session,
                    on_close=self._go_dashboard_from_inner_pages,
                    parent=self,
                )
                self.content_stack.addWidget(self.user_management_page)
            else:
                self.user_management_page.refresh_users()
            self.content_stack.setCurrentWidget(self.user_management_page)
            if self.back_btn:
                self.back_btn.setVisible(True)
        except Exception as exc:
            logger.exception(
                "Failed to open User Management page",
                extra={
                    "event_code": "USER_MANAGEMENT_OPEN_FAILED",
                    "error_code": "AUTH-UI-001",
                    "user_id": self.session.user.user_id,
                },
            )
            QMessageBox.critical(self, "User Management", f"Failed to open page:\n{exc}")

    def get_latest_image_from_folder(self, folder_path):
        if not os.path.isdir(folder_path):
            return None
        valid_exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
        files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.lower().endswith(valid_exts)]
        if not files:
            return None
        files.sort(key=os.path.getmtime, reverse=True)
        return files[0]
    
    @permission_required(Permission.ANNOTATION_USE)
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
    
    @permission_required(Permission.OSC_MANAGE)
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

            try:
                if getattr(self, "device_page", None) is not None:
                    if hasattr(self.device_page, "cleanup_device_page"):
                        self.device_page.cleanup_device_page(destroy_devices=True)
            except Exception as e:
                logger.warning(f"[DEVICE PAGE] close cleanup failed: {e}")

            self.update_timer.stop()
            self.update_label_timer.stop()
            self.update_images_timer.stop()
            self.copy_timer.stop()
            self._freeze_monitor.stop()
            if hasattr(self, "session_timer"):
                self.session_timer.stop()
            self.thread_manager.stop_all(timeout=3000)

        except Exception as e:
            logger.error(f"Cleanup error: {e}")
        finally:
            try:
                self.security_service.close_session(
                    self.session,
                    reason=self._session_close_reason,
                )
            except Exception:
                logger.exception(
                    "Failed to close security session",
                    extra={"event_code": "AUTH_SESSION_CLOSE_FAILED"},
                )
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

    security_service = get_security_service()
    if app_config.security.enabled:
        if security_service.user_count() == 0:
            QMessageBox.critical(
                None,
                "Apollo Security Setup Required",
                "No Apollo user accounts exist.\n\n"
                "Create the first administrator from the project folder using:\n"
                "python tools\\create_admin_user.py\n\n"
                "Then start GUI.py again.",
            )
            logger.error(
                "Application blocked because no security administrator exists",
                extra={
                    "event_code": "AUTH_BOOTSTRAP_REQUIRED",
                    "error_code": "AUTH-SETUP-001",
                    "status": "BLOCKED",
                },
            )
            shutdown_logging()
            return 2

        login_window = LoginWindow(media_path=MEDIA_PATH, service=security_service)
        if login_window.exec_() != QDialog.Accepted or login_window.logged_in_user is None:
            logger.info(
                "Application login cancelled",
                extra={"event_code": "AUTH_LOGIN_CANCELLED", "status": "CANCELLED"},
            )
            shutdown_logging()
            return 0
        session = security_service.create_session(login_window.logged_in_user)
    else:
        development_user = UserPrincipal(
            user_id=0,
            username="development",
            full_name="Development User",
            email="development@localhost",
            role=Role.ADMIN,
            permissions=frozenset(ALL_PERMISSIONS),
            must_change_password=False,
        )
        session = security_service.create_session(development_user)
        logger.warning(
            "Authentication bypassed because AUTH_ENABLED=False",
            extra={
                "event_code": "AUTH_DISABLED_BYPASS",
                "error_code": "AUTH-WARN-001",
                "status": "BYPASSED",
            },
        )

    main_window = MainWindow(session=session)
    main_window.show()
    
    return_code = app.exec_()
    cleanup_camera_resources()
    logger.info(
        "Apollo application stopped",
        extra={"event_code": "APPLICATION_STOPPED", "status": "STOPPED"},
    )
    shutdown_logging()
    sys.exit(return_code)

if __name__ == "__main__":
    main()