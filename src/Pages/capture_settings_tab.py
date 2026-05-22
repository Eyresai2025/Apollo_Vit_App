# capture_settings_tab_4cam_profiles.py
# =========================================================
# PyQt5 CAMERA CAPTURE SETTINGS TAB
# Lucid Arena SDK + Multi Camera Stitching
# 4K + 2K mixed camera support
# Special 2K profile: serial 250500042
# =========================================================

import os
import time
import queue
import ctypes
import threading
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List

import cv2
import numpy as np

from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QFormLayout,
    QLabel, QLineEdit, QPushButton, QProgressBar, QComboBox,
    QSpinBox, QDoubleSpinBox, QFileDialog, QTextEdit, QGroupBox,
    QMessageBox, QScrollArea, QSizePolicy, QFrame
)
from arena_api.system import system
from arena_api.buffer import BufferFactory


# =========================================================
# SERIAL-WISE CAMERA PROFILE OVERRIDES
# =========================================================
# All cameras use the UI values by default.
# Only the serials listed here override selected parameters.
#
# IMPORTANT:
# 250500042 is TRI02KA-M / 2K camera.
# It does not have AcquisitionLineRate / AcquisitionLineRateEnable in your setup,
# so we skip line-rate setting only for this camera.
# =========================================================
CAMERA_SERIAL_OVERRIDES: Dict[str, Dict] = {
    "250500042": {
        "profile_name": "TRI02KA-M 2K",
        "width": 2048,
        "camera_height": 14000,
        "final_height": 42000,
        "set_line_rate": False,
        "line_rate": None,
        "pixel_format": "Mono16",
        "exposure_us": 120.0,
        "gain_db": 24.0,
    }
}


# =========================================================
# SETTINGS MODEL
# =========================================================
@dataclass
class CaptureSettings:
    save_dir: str

    mode: str
    num_cameras_to_use: int
    camera_serials: List[str]

    # Default/global settings for normal 4K cameras.
    # Serial overrides can replace these values per camera.
    width: int
    camera_height: int
    final_height: int
    line_rate: float
    pixel_format: str
    exposure_us: float
    gain_db: float

    trigger_selector: str
    trigger_source: str
    trigger_activation: str

    num_stream_buffers: int
    packet_size: int
    packet_delay: int

    save_queue_size: int
    png_compression: int

    num_full_images: int


# =========================================================
# CAPTURE WORKER THREAD
# =========================================================
class CameraCaptureWorker(QThread):
    log_signal = pyqtSignal(str)
    status_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(int)
    image_count_signal = pyqtSignal(int, int)
    finished_signal = pyqtSignal(str)
    error_signal = pyqtSignal(str)

    def __init__(self, settings: CaptureSettings):
        super().__init__()
        self.settings = settings

        self.running = True
        self.save_queue = queue.Queue(maxsize=settings.save_queue_size)

        self.progress_lock = threading.Lock()
        self.image_lock = threading.Lock()

        self.progress_done = 0
        self.progress_total = 1

        self.images_done = 0
        self.images_total = 1

        self.errors = []

    # -----------------------------------------------------
    def stop(self):
        self.running = False
        self.status_signal.emit("Stopping capture...")

    # -----------------------------------------------------
    def log(self, msg):
        self.log_signal.emit(str(msg))

    # -----------------------------------------------------
    def set_node(self, nodemap, name, value):
        try:
            node = nodemap.get_node(name)
            if node and node.is_writable:
                node.value = value
                self.log(f"[SET OK] {name}: {node.value}")
                return True
            else:
                self.log(f"[SKIP] {name}: not writable / not found")
                return False
        except Exception as e:
            self.log(f"[SET FAIL] {name} -> {value}: {e}")
            return False

    # -----------------------------------------------------
    def read_node_value(self, nodemap, name, default="-"):
        try:
            node = nodemap.get_node(name)
            if node and node.is_readable:
                return node.value
        except Exception:
            pass
        return default

    # -----------------------------------------------------
    def build_camera_profile(self, serial):
        """Return the final per-camera profile for this serial."""
        s = self.settings
        serial = str(serial)

        profile = {
            "profile_name": "DEFAULT 4K",
            "width": s.width,
            "camera_height": s.camera_height,
            "final_height": s.final_height,
            "set_line_rate": True,
            "line_rate": s.line_rate,
            "pixel_format": s.pixel_format,
            "exposure_us": s.exposure_us,
            "gain_db": s.gain_db,
        }

        if serial in CAMERA_SERIAL_OVERRIDES:
            profile.update(CAMERA_SERIAL_OVERRIDES[serial])

        return profile

    # -----------------------------------------------------
    def convert_buffer(self, buffer):
        copied = BufferFactory.copy(buffer)
        try:
            width = copied.width
            height = copied.height
            total_bytes = len(copied.data)

            c_arr = (ctypes.c_ubyte * total_bytes).from_address(
                ctypes.addressof(copied.pbytes)
            )

            np_arr = np.ctypeslib.as_array(c_arr)
            bytes_per_pixel = total_bytes // (width * height)

            if bytes_per_pixel == 2:
                image = np_arr.view(np.uint16).reshape(height, width)
            else:
                image = np_arr.reshape(height, width)

            return image.copy()

        finally:
            BufferFactory.destroy(copied)

    # -----------------------------------------------------
    def flush_camera_buffers(self, camera, camera_index, flush_count):
        flushed = 0

        for _ in range(flush_count):
            if not self.running:
                break

            try:
                buffer = camera.get_buffer(timeout=100)
                camera.requeue_buffer(buffer)
                flushed += 1
            except Exception:
                break

        self.log(f"[CAM {camera_index}] FLUSHED {flushed} OLD BUFFER(S)")

    # -----------------------------------------------------
    def save_worker(self):
        while self.running or not self.save_queue.empty():
            try:
                item = self.save_queue.get(timeout=1)
            except queue.Empty:
                continue

            if item is None:
                self.save_queue.task_done()
                break

            filename, image = item

            try:
                img_8bit = cv2.normalize(
                    image,
                    None,
                    0,
                    255,
                    cv2.NORM_MINMAX
                )

                img_8bit = img_8bit.astype(np.uint8)

                cv2.imwrite(
                    filename,
                    img_8bit,
                    [cv2.IMWRITE_PNG_COMPRESSION, self.settings.png_compression]
                )

                self.log(f"[SAVE OK] {filename}")

            except Exception as e:
                self.log(f"[SAVE ERROR] {filename}: {e}")

            finally:
                self.save_queue.task_done()

    # -----------------------------------------------------
    def configure_camera(self, camera, camera_index):
        s = self.settings
        nodemap = camera.nodemap
        serial = str(self.read_node_value(nodemap, "DeviceSerialNumber", f"CAM_{camera_index}"))
        profile = self.build_camera_profile(serial)

        self.log("")
        self.log(f"========== CONFIG CAMERA {camera_index} ==========")
        self.log(f"[CAM {camera_index}] SERIAL: {serial}")
        self.log(f"[CAM {camera_index}] PROFILE: {profile['profile_name']}")

        self.set_node(nodemap, "Width", profile["width"])
        self.set_node(nodemap, "Height", profile["camera_height"])
        self.set_node(nodemap, "PixelFormat", profile["pixel_format"])

        self.set_node(nodemap, "ExposureAutoLimitAuto", "Off")
        self.set_node(nodemap, "ExposureTime", profile["exposure_us"])

        self.set_node(nodemap, "Gain", profile["gain_db"])

        if profile.get("set_line_rate", True) and profile.get("line_rate") is not None:
            self.set_node(nodemap, "AcquisitionLineRateEnable", True)
            self.set_node(nodemap, "AcquisitionLineRate", profile["line_rate"])
        else:
            self.log(
                f"[CAM {camera_index}] Line-rate skipped for serial {serial} "
                f"({profile['profile_name']})"
            )

        self.set_node(nodemap, "AcquisitionMode", "Continuous")

        self.set_node(nodemap, "GevSCPSPacketSize", s.packet_size)
        self.set_node(nodemap, "GevSCPD", s.packet_delay)

        if s.mode == "FREE":
            self.log("[MODE] FREE MODE ENABLED")
            self.set_node(nodemap, "TriggerMode", "Off")

        elif s.mode == "AUTO":
            self.log("[MODE] AUTO MODE ENABLED")

            self.set_node(nodemap, "TriggerMode", "Off")
            self.set_node(nodemap, "TriggerSelector", s.trigger_selector)
            self.set_node(nodemap, "TriggerSource", s.trigger_source)
            self.set_node(nodemap, "TriggerActivation", s.trigger_activation)
            self.set_node(nodemap, "TriggerMode", "On")

        self.log("------ FINAL CAMERA SETTINGS ------")
        for node_name in [
            "DeviceSerialNumber",
            "Width",
            "Height",
            "PixelFormat",
            "ExposureTime",
            "Gain",
            "AcquisitionLineRate",
            "TriggerMode",
            "TriggerSelector",
            "TriggerSource",
            "TriggerActivation",
            "GevSCPSPacketSize",
            "GevSCPD"
        ]:
            value = self.read_node_value(nodemap, node_name)
            self.log(f"{node_name}: {value}")

    # -----------------------------------------------------
    def step_progress(self):
        with self.progress_lock:
            self.progress_done += 1
            percent = int((self.progress_done / max(1, self.progress_total)) * 100)
            percent = max(0, min(100, percent))
            self.progress_signal.emit(percent)

    # -----------------------------------------------------
    def step_image_count(self):
        with self.image_lock:
            self.images_done += 1
            self.image_count_signal.emit(self.images_done, self.images_total)

    # -----------------------------------------------------
    def camera_worker(self, camera, camera_index):
        s = self.settings

        try:
            nodemap = camera.nodemap
            serial = str(self.read_node_value(nodemap, "DeviceSerialNumber", f"CAM_{camera_index}"))
            profile = self.build_camera_profile(serial)

            width = int(profile["width"])
            camera_height = int(profile["camera_height"])
            final_height = int(profile["final_height"])

            self.log("")
            self.log(f"[CAM {camera_index}] SERIAL: {serial}")
            self.log(
                f"[CAM {camera_index}] RUNTIME: width={width}, "
                f"camera_height={camera_height}, final_height={final_height}"
            )

            serial_dir = os.path.join(s.save_dir, str(serial))
            os.makedirs(serial_dir, exist_ok=True)

            stream_started = False

            try:
                camera.start_stream(s.num_stream_buffers)
                stream_started = True

                self.log(f"[CAM {camera_index}] STREAM STARTED")

                if s.mode == "AUTO":
                    self.log(f"[CAM {camera_index}] WAITING FOR PLC TRIGGER...")
                else:
                    self.log(f"[CAM {camera_index}] FREE RUNNING...")

                for img_idx in range(s.num_full_images):
                    if not self.running:
                        break

                    self.flush_camera_buffers(
                        camera,
                        camera_index,
                        flush_count=s.num_stream_buffers
                    )
                    time.sleep(0.05)
                    self.status_signal.emit(
                        f"Camera {camera_index}: capturing image {img_idx + 1}/{s.num_full_images}"
                    )

                    self.log("")
                    self.log(
                        f"[CAM {camera_index}] START STITCH IMAGE "
                        f"{img_idx + 1}/{s.num_full_images}"
                    )

                    full_img = np.zeros(
                        (final_height, width),
                        dtype=np.uint16
                    )

                    current_row = 0
                    start_time = time.time()

                    while current_row < final_height and self.running:
                        try:
                            buffer = camera.get_buffer(timeout=1000)
                        except Exception:
                            self.log(f"[CAM {camera_index}] WAITING FOR TRIGGER...")
                            time.sleep(0.01)
                            continue

                        try:
                            frame = self.convert_buffer(buffer)
                            h, w = frame.shape

                            if w != width:
                                self.log(
                                    f"[CAM {camera_index}] WIDTH WARNING: "
                                    f"frame width={w}, expected={width}"
                                )

                            remaining = final_height - current_row
                            lines_to_copy = min(h, remaining)
                            cols_to_copy = min(w, width)

                            full_img[
                                current_row:current_row + lines_to_copy,
                                0:cols_to_copy
                            ] = frame[:lines_to_copy, :cols_to_copy]

                            current_row += lines_to_copy

                            self.log(
                                f"[CAM {camera_index}] "
                                f"{current_row}/{final_height}"
                            )

                            self.step_progress()

                        finally:
                            camera.requeue_buffer(buffer)

                    if not self.running:
                        break

                    end_time = time.time()

                    self.log(
                        f"[CAM {camera_index}] STITCH COMPLETE "
                        f"Time: {end_time - start_time:.2f} sec"
                    )

                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

                    filename = os.path.join(
                        serial_dir,
                        f"cam_{serial}_{timestamp}.png"
                    )

                    self.save_queue.put((filename, full_img))
                    self.step_image_count()

            finally:
                if stream_started:
                    try:
                        camera.stop_stream()
                        self.log(f"[CAM {camera_index}] STREAM STOPPED")
                    except Exception as e:
                        self.log(f"[CAM {camera_index}] STOP STREAM ERROR: {e}")

        except Exception as e:
            err = f"[CAM {camera_index}] ERROR: {e}"
            self.errors.append(err)
            self.log(err)
            self.running = False

    # -----------------------------------------------------
    def select_cameras(self, devices):
        """Select cameras either by exact serial list or first N detected cameras."""
        s = self.settings

        if s.camera_serials:
            serial_to_camera = {}
            detected_serials = []

            for cam in devices:
                serial = str(self.read_node_value(cam.nodemap, "DeviceSerialNumber", ""))
                detected_serials.append(serial)
                if serial:
                    serial_to_camera[serial] = cam

            missing = [serial for serial in s.camera_serials if serial not in serial_to_camera]
            if missing:
                raise RuntimeError(
                    "Requested camera serial(s) not found: "
                    + ", ".join(missing)
                    + "\nDetected serial(s): "
                    + ", ".join(detected_serials)
                )

            return [serial_to_camera[serial] for serial in s.camera_serials]

        use_count = min(s.num_cameras_to_use, len(devices))
        return devices[:use_count]

    # -----------------------------------------------------
    def run(self):
        devices = []

        try:
            s = self.settings

            os.makedirs(s.save_dir, exist_ok=True)

            self.progress_signal.emit(0)
            self.status_signal.emit("Searching cameras...")

            self.log("")
            self.log("Searching Cameras...")

            devices = system.create_device()

            if len(devices) == 0:
                raise RuntimeError("No cameras found")

            self.log(f"Detected Cameras: {len(devices)}")

            cameras = self.select_cameras(devices)
            use_count = len(cameras)

            if use_count == 0:
                raise RuntimeError("No camera selected")

            self.log(f"Using Cameras: {use_count}")

            # Progress is calculated per camera profile because 2K/4K profiles can differ.
            total_chunks_one_cycle = 0
            for cam in cameras:
                serial = str(self.read_node_value(cam.nodemap, "DeviceSerialNumber", ""))
                profile = self.build_camera_profile(serial)
                chunks = int(np.ceil(profile["final_height"] / profile["camera_height"]))
                total_chunks_one_cycle += chunks

            self.progress_total = s.num_full_images * total_chunks_one_cycle
            self.images_total = use_count * s.num_full_images

            self.image_count_signal.emit(0, self.images_total)

            self.status_signal.emit("Configuring cameras...")

            for idx, cam in enumerate(cameras):
                if not self.running:
                    break
                self.configure_camera(cam, idx)

            if not self.running:
                self.finished_signal.emit("Capture stopped before start")
                return

            saver_thread = threading.Thread(
                target=self.save_worker,
                daemon=True
            )
            saver_thread.start()

            camera_threads = []
            start_time = time.time()

            self.status_signal.emit("Capture started...")

            for idx, cam in enumerate(cameras):
                if not self.running:
                    break

                t = threading.Thread(
                    target=self.camera_worker,
                    args=(cam, idx),
                    daemon=True
                )
                t.start()
                camera_threads.append(t)

            for t in camera_threads:
                t.join()

            self.save_queue.join()
            self.save_queue.put(None)
            saver_thread.join(timeout=5)

            end_time = time.time()

            self.progress_signal.emit(100)

            if self.errors:
                raise RuntimeError("\n".join(self.errors))

            if self.running:
                self.finished_signal.emit(
                    f"Capture completed successfully. Total time: {end_time - start_time:.2f} sec"
                )
            else:
                self.finished_signal.emit("Capture stopped")

        except Exception as e:
            self.error_signal.emit(str(e))

        finally:
            try:
                system.destroy_device()
            except Exception:
                pass


# =========================================================
# UI TAB
# =========================================================
class CameraCaptureSettingsTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.worker = None

        self.build_ui()

    # -----------------------------------------------------
    def build_ui(self):
        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.setSpacing(0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        scroll_content = QWidget()
        scroll_content.setMinimumSize(0, 0)
        scroll_content.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        main_layout = QVBoxLayout(scroll_content)
        main_layout.setContentsMargins(18, 18, 18, 18)
        main_layout.setSpacing(12)

        scroll.setWidget(scroll_content)
        outer_layout.addWidget(scroll)

        title = QLabel("Camera Capture Settings")
        title.setObjectName("PageTitle")
        main_layout.addWidget(title)

        settings_box = QGroupBox("Capture Settings")
        settings_box.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        settings_layout = QGridLayout(settings_box)
        settings_layout.setSpacing(12)

        # SAVE DIR
        self.save_dir_edit = QLineEdit(
            r"C:\Users\PrajwalSridhar\Desktop\Apollo_share"
        )
        browse_btn = QPushButton("Browse")
        browse_btn.clicked.connect(self.browse_save_dir)

        save_dir_layout = QHBoxLayout()
        save_dir_layout.addWidget(self.save_dir_edit)
        save_dir_layout.addWidget(browse_btn)

        # MODE
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["FREE", "AUTO"])

        self.num_cameras_spin = self.make_spin(1, 16, 4)

        # Optional exact serial order. Leave blank to use first N detected cameras.
        self.camera_serials_edit = QLineEdit("")
        self.camera_serials_edit.setPlaceholderText(
            "Optional: 254901428,254901432,254901430,250500042"
        )

        # CAMERA SETTINGS - default values for normal 4K cameras.
        # Serial 250500042 overrides width=2048 and skips line-rate automatically.
        self.width_spin = self.make_spin(1, 100000, 4096)
        self.camera_height_spin = self.make_spin(1, 100000, 14000)
        self.final_height_spin = self.make_spin(1, 200000, 42000)

        self.line_rate_spin = self.make_double(1, 200000, 8169.178266, 6)
        self.pixel_format_combo = QComboBox()
        self.pixel_format_combo.addItems(["Mono16", "Mono8"])

        self.exposure_spin = self.make_double(1, 100000, 120.0, 3)
        self.gain_spin = self.make_double(0, 48, 24.0, 3)

        # TRIGGER SETTINGS
        self.trigger_selector_combo = QComboBox()
        self.trigger_selector_combo.addItems(["AcquisitionStart", "FrameStart"])
        self.trigger_selector_combo.setCurrentText("FrameStart")

        self.trigger_source_combo = QComboBox()
        self.trigger_source_combo.addItems(["Line0", "Line1", "Software"])
        self.trigger_source_combo.setCurrentText("Line0")

        self.trigger_activation_combo = QComboBox()
        self.trigger_activation_combo.addItems(["RisingEdge", "FallingEdge", "AnyEdge", "LevelHigh", "LevelLow"])
        self.trigger_activation_combo.setCurrentText("RisingEdge")

        # STREAM SETTINGS
        self.stream_buffers_spin = self.make_spin(1, 128, 8)
        self.packet_size_spin = self.make_spin(576, 9014, 9000)
        self.packet_delay_spin = self.make_spin(0, 100000, 1000)

        # SAVE SETTINGS
        # Keep this small for 4 cameras, because every full image is very large.
        self.save_queue_spin = self.make_spin(1, 10000, 8)
        self.png_compression_spin = self.make_spin(0, 9, 0)

        # CAPTURE COUNT
        self.num_images_spin = self.make_spin(1, 1000, 1)

        left_form = QFormLayout()
        left_form.setSpacing(10)
        left_form.addRow("Save Folder", save_dir_layout)
        left_form.addRow("Mode", self.mode_combo)
        left_form.addRow("Number of Cameras", self.num_cameras_spin)
        left_form.addRow("Camera Serials", self.camera_serials_edit)
        left_form.addRow("4K Width", self.width_spin)
        left_form.addRow("Camera Height / Patch Height", self.camera_height_spin)
        left_form.addRow("Final Stitch Height", self.final_height_spin)
        left_form.addRow("4K Line Rate", self.line_rate_spin)
        left_form.addRow("Pixel Format", self.pixel_format_combo)
        left_form.addRow("Exposure Time us", self.exposure_spin)
        left_form.addRow("Gain dB", self.gain_spin)

        right_form = QFormLayout()
        right_form.setSpacing(10)
        right_form.addRow("Trigger Selector", self.trigger_selector_combo)
        right_form.addRow("Trigger Source", self.trigger_source_combo)
        right_form.addRow("Trigger Activation", self.trigger_activation_combo)
        right_form.addRow("Stream Buffers", self.stream_buffers_spin)
        right_form.addRow("Packet Size", self.packet_size_spin)
        right_form.addRow("Packet Delay", self.packet_delay_spin)
        right_form.addRow("Save Queue Size", self.save_queue_spin)
        right_form.addRow("PNG Compression", self.png_compression_spin)
        right_form.addRow("Number of Full Images", self.num_images_spin)

        settings_layout.addLayout(left_form, 0, 0)
        settings_layout.addLayout(right_form, 0, 1)

        main_layout.addWidget(settings_box)

        note = QLabel(
            "Profile rule: serial 250500042 is auto-treated as 2K. "
            "It uses width=2048 and skips AcquisitionLineRate/AcquisitionLineRateEnable. "
            "Other cameras use the 4K UI defaults."
        )
        note.setWordWrap(True)
        note.setObjectName("InfoNote")
        main_layout.addWidget(note)

        # CONTROL BOX
        control_box = QGroupBox("Capture Control")
        control_box.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        control_layout = QVBoxLayout(control_box)
        btn_layout = QHBoxLayout()

        self.capture_btn = QPushButton("Start Capture")
        self.capture_btn.clicked.connect(self.start_capture)

        self.stop_btn = QPushButton("Stop")
        self.stop_btn.clicked.connect(self.stop_capture)
        self.stop_btn.setEnabled(False)

        btn_layout.addWidget(self.capture_btn)
        btn_layout.addWidget(self.stop_btn)
        btn_layout.addStretch()

        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setAlignment(Qt.AlignCenter)

        self.status_label = QLabel("Ready")
        self.image_count_label = QLabel("Images Captured: 0 / 0")

        control_layout.addLayout(btn_layout)
        control_layout.addWidget(self.progress_bar)
        control_layout.addWidget(self.status_label)
        control_layout.addWidget(self.image_count_label)

        main_layout.addWidget(control_box)

        # LOG BOX
        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setMinimumHeight(120)
        self.log_box.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        main_layout.addWidget(self.log_box)

        self.setStyleSheet("""
            QWidget {
                background: #f7f7f9;
                font-family: Arial;
                font-size: 13px;
            }

            QLabel#PageTitle {
                font-size: 22px;
                font-weight: bold;
                color: #5b168b;
            }

            QLabel#InfoNote {
                background: #fff7df;
                border: 1px solid #e8d28a;
                border-radius: 8px;
                padding: 8px 10px;
                color: #4b3b00;
            }

            QGroupBox {
                background: white;
                border: 1px solid #dedede;
                border-radius: 12px;
                margin-top: 12px;
                padding: 14px;
                font-weight: bold;
            }

            QGroupBox::title {
                subcontrol-origin: margin;
                left: 14px;
                padding: 0 6px;
                color: #5b168b;
            }

            QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {
                min-height: 30px;
                border: 1px solid #cfcfcf;
                border-radius: 6px;
                padding: 4px 8px;
                background: white;
            }

            QPushButton {
                min-height: 34px;
                border-radius: 8px;
                padding: 6px 16px;
                background: #6d2fa0;
                color: white;
                font-weight: bold;
            }

            QPushButton:hover {
                background: #7e3bb8;
            }

            QPushButton:disabled {
                background: #9a9a9a;
            }

            QProgressBar {
                height: 26px;
                border: 1px solid #cfcfcf;
                border-radius: 8px;
                text-align: center;
                background: white;
                font-weight: bold;
            }

            QProgressBar::chunk {
                border-radius: 8px;
                background: #6d2fa0;
            }

            QTextEdit {
                background: #111;
                color: #00ff7f;
                border-radius: 8px;
                padding: 8px;
                font-family: Consolas;
                font-size: 12px;
            }
        """)

    # -----------------------------------------------------
    def make_spin(self, min_val, max_val, default):
        spin = QSpinBox()
        spin.setRange(min_val, max_val)
        spin.setValue(default)
        return spin

    # -----------------------------------------------------
    def make_double(self, min_val, max_val, default, decimals):
        spin = QDoubleSpinBox()
        spin.setRange(min_val, max_val)
        spin.setDecimals(decimals)
        spin.setValue(default)
        spin.setSingleStep(1.0)
        return spin

    # -----------------------------------------------------
    def browse_save_dir(self):
        folder = QFileDialog.getExistingDirectory(
            self,
            "Select Save Folder",
            self.save_dir_edit.text()
        )

        if folder:
            self.save_dir_edit.setText(folder)

    # -----------------------------------------------------
    def parse_camera_serials(self):
        text = self.camera_serials_edit.text().strip()
        if not text:
            return []

        # Supports comma or newline separated serials.
        text = text.replace("\n", ",")
        serials = [item.strip() for item in text.split(",") if item.strip()]
        return serials

    # -----------------------------------------------------
    def get_settings_from_ui(self):
        return CaptureSettings(
            save_dir=self.save_dir_edit.text().strip(),

            mode=self.mode_combo.currentText(),
            num_cameras_to_use=self.num_cameras_spin.value(),
            camera_serials=self.parse_camera_serials(),

            width=self.width_spin.value(),
            camera_height=self.camera_height_spin.value(),
            final_height=self.final_height_spin.value(),
            line_rate=self.line_rate_spin.value(),
            pixel_format=self.pixel_format_combo.currentText(),
            exposure_us=self.exposure_spin.value(),
            gain_db=self.gain_spin.value(),

            trigger_selector=self.trigger_selector_combo.currentText(),
            trigger_source=self.trigger_source_combo.currentText(),
            trigger_activation=self.trigger_activation_combo.currentText(),

            num_stream_buffers=self.stream_buffers_spin.value(),
            packet_size=self.packet_size_spin.value(),
            packet_delay=self.packet_delay_spin.value(),

            save_queue_size=self.save_queue_spin.value(),
            png_compression=self.png_compression_spin.value(),

            num_full_images=self.num_images_spin.value()
        )

    # -----------------------------------------------------
    def start_capture(self):
        if self.worker is not None and self.worker.isRunning():
            QMessageBox.warning(self, "Capture Running", "Capture is already running.")
            return

        settings = self.get_settings_from_ui()

        if not settings.save_dir:
            QMessageBox.warning(self, "Missing Folder", "Please select save folder.")
            return

        self.log_box.clear()
        self.progress_bar.setValue(0)
        self.status_label.setText("Starting capture...")
        self.image_count_label.setText("Images Captured: 0 / 0")

        self.capture_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)

        self.worker = CameraCaptureWorker(settings)

        self.worker.log_signal.connect(self.append_log)
        self.worker.status_signal.connect(self.status_label.setText)
        self.worker.progress_signal.connect(self.progress_bar.setValue)
        self.worker.image_count_signal.connect(self.update_image_count)
        self.worker.finished_signal.connect(self.capture_finished)
        self.worker.error_signal.connect(self.capture_error)

        self.worker.start()

    # -----------------------------------------------------
    def stop_capture(self):
        if self.worker is not None and self.worker.isRunning():
            self.worker.stop()
            self.status_label.setText("Stopping capture...")

    # -----------------------------------------------------
    def append_log(self, msg):
        self.log_box.append(msg)
        self.log_box.verticalScrollBar().setValue(
            self.log_box.verticalScrollBar().maximum()
        )

    # -----------------------------------------------------
    def update_image_count(self, done, total):
        self.image_count_label.setText(f"Images Captured: {done} / {total}")

    # -----------------------------------------------------
    def capture_finished(self, msg):
        self.status_label.setText(msg)
        self.capture_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.append_log("")
        self.append_log(msg)
        QMessageBox.information(self, "Capture Finished", msg)

    # -----------------------------------------------------
    def capture_error(self, err):
        self.status_label.setText("Capture failed")
        self.capture_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.append_log("")
        self.append_log("[ERROR]")
        self.append_log(err)
        QMessageBox.critical(self, "Capture Error", err)
