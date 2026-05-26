# src/UI/gui_helpers.py

import os
import logging
from threading import Lock, Event

from PyQt5.QtCore import QObject, QEvent, QThread, Qt, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QImageReader, QPixmap
from PyQt5.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
)

from src.Main_cam import run_capture_folder_cycle, preload_live_runtimes


logger = logging.getLogger(__name__)


# =========================================================
# SKU HELPER
# =========================================================

def get_available_sku_names(media_root):
    calibration_root = os.path.join(media_root, "AI_Calibration_Files")

    if not os.path.isdir(calibration_root):
        return []

    required_files_by_side = {
        "calibration_sidewall1": [
            "alignment_reference_polarized.png",
            "crop_anchor_reference.json",
            "embedding_bank.pt",
            "embedding_bank_meta.pt",
            "thresholds_by_rc.pt",
            "mahalanobis_stats.pt",
            "pca_artifact.pt",
            "reference_r.pt",
        ],
        "calibration_sidewall2": [
            "alignment_reference_polarized.png",
            "crop_anchor_reference.json",
            "embedding_bank.pt",
            "embedding_bank_meta.pt",
            "thresholds_by_rc.pt",
            "mahalanobis_stats.pt",
            "pca_artifact.pt",
            "reference_r.pt",
        ],
        "calibration_tread": [
            "embedding_bank.pt",
            "embedding_bank_meta.pt",
            "thresholds_by_rc.pt",
            "mahalanobis_stats.pt",
            "pca_artifact.pt",
            "tread_x_reference_crop.png",
            "tread_x_reference_signature.npy",
            "tread_x_reference_signature_meta.json",
        ],
    }

    sku_names = []

    for name in sorted(os.listdir(calibration_root)):
        sku_dir = os.path.join(calibration_root, name)

        if not os.path.isdir(sku_dir):
            continue

        if not name.upper().startswith("SKU"):
            continue

        ok = True

        for side_dir_name, required_files in required_files_by_side.items():
            artifacts_dir = os.path.join(sku_dir, side_dir_name, "artifacts")

            if not os.path.isdir(artifacts_dir):
                ok = False
                break

            for file_name in required_files:
                if not os.path.isfile(os.path.join(artifacts_dir, file_name)):
                    ok = False
                    break

            if not ok:
                break

        if ok:
            sku_names.append(name)

    return sku_names


# =========================================================
# THREAD MANAGER
# =========================================================

class ThreadManager:
    """Manages QThread lifecycle to prevent memory leaks and freezes."""

    def __init__(self, parent=None):
        self.parent = parent
        self.active_threads = {}
        self.active_workers = {}
        self._lock = Lock()

    def start_thread(self, name, worker, on_finished=None, on_error=None):
        with self._lock:
            if name in self.active_threads:
                old_thread = self.active_threads[name]

                if old_thread.isRunning():
                    logger.debug(f"Stopping existing thread '{name}'")
                    old_thread.quit()

                    if not old_thread.wait(3000):
                        logger.warning(f"Thread '{name}' did not stop. Terminating.")
                        old_thread.terminate()
                        old_thread.wait()

                old_thread.deleteLater()

            thread = QThread(self.parent)
            worker.moveToThread(thread)

            thread.started.connect(worker.run, Qt.QueuedConnection)

            if on_finished:
                worker.finished.connect(on_finished, Qt.QueuedConnection)

            if on_error:
                worker.error.connect(on_error, Qt.QueuedConnection)

            worker.finished.connect(thread.quit, Qt.QueuedConnection)
            worker.error.connect(thread.quit, Qt.QueuedConnection)

            def cleanup():
                with self._lock:
                    self.active_threads.pop(name, None)
                    self.active_workers.pop(name, None)

            thread.finished.connect(cleanup)
            thread.finished.connect(thread.deleteLater)

            self.active_threads[name] = thread
            self.active_workers[name] = worker

            thread.start()
            return True

    def stop_all(self, timeout=5000):
        with self._lock:
            for name, thread in list(self.active_threads.items()):
                if thread.isRunning():
                    thread.quit()

                    if not thread.wait(timeout):
                        thread.terminate()
                        thread.wait()

            self.active_threads.clear()
            self.active_workers.clear()


# =========================================================
# IMAGE CACHE
# =========================================================

class ImageCache:
    """Thread-safe image cache with size limit."""

    def __init__(self, max_size=50):
        self._cache = {}
        self._lock = Lock()
        self._max_size = max_size

    def get(self, key, loader_func):
        with self._lock:
            if key in self._cache:
                return self._cache[key]

        image = loader_func()

        if image is not None and not (hasattr(image, "isNull") and image.isNull()):
            with self._lock:
                if len(self._cache) >= self._max_size:
                    self._cache.clear()

                self._cache[key] = image

        return image

    def clear(self):
        with self._lock:
            self._cache.clear()


image_cache = ImageCache(max_size=50)


# =========================================================
# WORKERS
# =========================================================

class RuntimePreloadWorker(QObject):
    finished = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(
        self,
        media_root,
        sku_name,
        device,
        seg_model_a_path,
        seg_model_b_path,
        vit_checkpoint_path,
        r_detector_path,
    ):
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

    def __init__(
        self,
        media_root,
        sku_name="SKU_001",
        tyre_name="195_65_R15",
        device="cuda",
        seg_model_a_path=None,
        seg_model_b_path=None,
        vit_checkpoint_path=None,
        r_detector_path=None,
        multi_camera_manager=None,
        demo_capture_root=None,
    ):
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

    def __init__(
        self,
        media_root,
        panel_size=(260, 700),
        fallback_paths=None,
        sku_name=None,
        cycle_dir_override=None,
    ):
        super().__init__()
        self.media_root = media_root
        self.panel_w, self.panel_h = panel_size
        self.fallback_paths = fallback_paths or {}
        self.sku_name = sku_name
        self.cycle_dir_override = cycle_dir_override
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
                img_path = self._find_side_final_image(cycle_dir, folder_name)

                if not img_path:
                    img_path = self._find_latest_final_image(cycle_dir, folder_name)

            if not img_path:
                img_path = self.fallback_paths.get(side_key)

            if img_path and os.path.exists(img_path):
                qimage = self._load_scaled_qimage(img_path)

            payload["images"][side_key] = {
                "path": img_path,
                "qimage": qimage,
            }

        return payload

    def _find_side_final_image(self, cycle_dir, side_name):
        candidates = [
            os.path.join(cycle_dir, side_name, "final", "final_stitched.png"),
            os.path.join(cycle_dir, side_name, "final", "template_stitched.png"),
            os.path.join(cycle_dir, side_name, "final", "defect_overlay.png"),
            os.path.join(cycle_dir, side_name, "final", "final.png"),
            os.path.join(cycle_dir, side_name, "final_stitched.png"),
            os.path.join(cycle_dir, side_name, "template_stitched.png"),
        ]

        for path in candidates:
            if os.path.isfile(path):
                return path

        return None

    def _get_latest_cycle_dir(self):
        if self.cycle_dir_override and os.path.isdir(self.cycle_dir_override):
            return self.cycle_dir_override

        output_base = os.path.join(self.media_root, "Output")

        if not os.path.isdir(output_base):
            return None

        search_sku_roots = []

        if self.sku_name and str(self.sku_name).strip() not in ["", "--", "None"]:
            sku_root = os.path.join(output_base, self.sku_name)

            if os.path.isdir(sku_root):
                search_sku_roots.append(sku_root)

        if not search_sku_roots:
            for sku_name in os.listdir(output_base):
                sku_root = os.path.join(output_base, sku_name)

                if os.path.isdir(sku_root):
                    search_sku_roots.append(sku_root)

        cycle_candidates = []

        for sku_root in search_sku_roots:
            for date_name in os.listdir(sku_root):
                date_root = os.path.join(sku_root, date_name)

                if not os.path.isdir(date_root):
                    continue

                for cycle_name in os.listdir(date_root):
                    cycle_dir = os.path.join(date_root, cycle_name)

                    if not os.path.isdir(cycle_dir):
                        continue

                    if not cycle_name.startswith("Cycle_"):
                        continue

                    try:
                        cycle_num = int(cycle_name.replace("Cycle_", "").strip())
                    except Exception:
                        cycle_num = -1

                    cycle_candidates.append(
                        {
                            "cycle_dir": cycle_dir,
                            "cycle_num": cycle_num,
                            "mtime": os.path.getmtime(cycle_dir),
                        }
                    )

        if not cycle_candidates:
            return None

        cycle_candidates.sort(
            key=lambda x: (x["mtime"], x["cycle_num"]),
            reverse=True,
        )

        return cycle_candidates[0]["cycle_dir"]

    def _find_latest_final_image(self, cycle_dir, side_folder):
        side_final_root = os.path.join(cycle_dir, side_folder, "final")

        if not os.path.isdir(side_final_root):
            return None

        valid_exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
        candidates = []

        for root, _, files in os.walk(side_final_root):
            for file_name in files:
                lower_name = file_name.lower()
                full_path = os.path.join(root, file_name)

                if lower_name == "final_stitched.png":
                    candidates.append((0, os.path.getmtime(full_path), full_path))

                elif lower_name.endswith(valid_exts):
                    candidates.append((1, os.path.getmtime(full_path), full_path))

        if not candidates:
            return None

        candidates.sort(key=lambda x: (x[0], -x[1]))
        return candidates[0][2]

    def _load_scaled_qimage(self, img_path):
        cache_key = f"qimage_{img_path}_{self.panel_w}_{self.panel_h}"

        def loader():
            reader = QImageReader(img_path)
            reader.setAutoTransform(True)

            original_size = reader.size()

            if (
                original_size.isValid()
                and original_size.width() > 0
                and original_size.height() > 0
            ):
                scaled_size = original_size.scaled(
                    self.panel_w,
                    self.panel_h,
                    Qt.KeepAspectRatio,
                )
                reader.setScaledSize(scaled_size)

            image = reader.read()
            return image if not image.isNull() else None

        return image_cache.get(cache_key, loader)

    def stop(self):
        self._stop_event.set()


# =========================================================
# IMAGE VIEWER
# =========================================================

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
            button = QPushButton(text)
            button.setFixedHeight(32)
            button.setStyleSheet("""
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
            return button

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

        width = max(1, int(self._pixmap.width() * self.scale_factor))
        height = max(1, int(self._pixmap.height() * self.scale_factor))

        scaled = self._pixmap.scaled(
            width,
            height,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
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