from PyQt5.QtWidgets import (
    QScrollArea, QSizePolicy, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTabWidget, QTableWidget, QTableWidgetItem, QComboBox,
    QLineEdit, QFormLayout, QGroupBox, QMessageBox, QCheckBox,
    QAbstractItemView, QHeaderView,QFrame
)
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt

from src.device.camera_profile_manager import (
    CameraProfileManager,
    ZONE_NAMES,
    ZONE_KEYS,
    DEFAULT_CAMERA_SETTINGS
)
from src.device.arena_camera_manager import ArenaCameraManager
from src.workers.camera_live_preview_worker import CameraLivePreviewWorker
from src.workers.camera_capture_worker import CameraCaptureWorker
from src.device.laser_profile_manager import (
    LaserProfileManager,
    LASER_ZONE_NAMES,
    LASER_ZONE_KEYS,
    DEFAULT_LASER_SETTINGS
)
from src.device.teledyne_laser_manager import TeledyneLaserManager
from src.workers.laser_live_profile_worker import LaserLiveProfileWorker
from src.workers.laser_capture_worker import LaserCaptureWorker
from src.device.sku_device_profile_store import SKUDeviceProfileStore

class DevicePage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.profile_manager = CameraProfileManager()
        self.camera_manager = ArenaCameraManager()

        self.laser_profile_manager = LaserProfileManager()
        self.laser_manager = TeledyneLaserManager()
        self.sku_profile_store = SKUDeviceProfileStore("media")
        self.selected_serial = None
        self.camera_settings_by_serial = {}

        self.selected_laser_id = None
        self.laser_settings_by_id = {}

        self.live_worker = None
        self.capture_worker = None

        self.laser_live_worker = None
        self.laser_capture_worker = None

        self.init_ui()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------
    def init_ui(self):
        main_layout = QVBoxLayout(self)

        title = QLabel("Device Configuration")
        title.setStyleSheet("font-size: 20px; font-weight: bold;")
        main_layout.addWidget(title)

        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)

        self.camera_tab = QWidget()
        self.laser_tab = QWidget()

        self.tabs.addTab(self.camera_tab, "Camera")
        self.tabs.addTab(self.laser_tab, "Laser")

        self.build_camera_tab()
        self.build_laser_tab()

    def build_camera_tab(self):
        layout = QVBoxLayout(self.camera_tab)

        top_row = QHBoxLayout()

        self.sku_input = QLineEdit()
        self.sku_input.setPlaceholderText("Enter SKU name, example: 185_70_R14_AMZ4G")

        self.refresh_btn = QPushButton("Refresh Cameras")
        self.load_profile_btn = QPushButton("Load Profile")
        self.save_profile_btn = QPushButton("Save Profile")

        top_row.addWidget(QLabel("SKU:"))
        top_row.addWidget(self.sku_input)
        top_row.addWidget(self.refresh_btn)
        top_row.addWidget(self.load_profile_btn)
        top_row.addWidget(self.save_profile_btn)

        layout.addLayout(top_row)

        self.camera_table = QTableWidget()
        self.camera_table.setColumnCount(6)
        self.camera_table.setHorizontalHeaderLabels([
            "Camera Serial",
            "Model",
            "IP",
            "Connection Status",
            "Assigned Zone",
            "Enabled"
        ])
        self.camera_table.horizontalHeader().setStretchLastSection(True)
        self.camera_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.camera_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.camera_table.setEditTriggers(QAbstractItemView.NoEditTriggers)

        layout.addWidget(self.camera_table)

        bottom_row = QHBoxLayout()

        self.settings_box = self.create_settings_box()
        self.preview_box = self.create_preview_box()

        bottom_row.addWidget(self.settings_box, 1)
        bottom_row.addWidget(self.preview_box, 2)

        layout.addLayout(bottom_row)

        self.refresh_btn.clicked.connect(self.refresh_cameras)
        self.load_profile_btn.clicked.connect(self.load_profile)
        self.save_profile_btn.clicked.connect(self.save_profile)
        self.camera_table.cellClicked.connect(self.on_camera_selected)

    def build_laser_tab(self):
        layout = QVBoxLayout(self.laser_tab)

        # Top controls
        top_row = QHBoxLayout()

        self.laser_sku_input = QLineEdit()
        self.laser_sku_input.setPlaceholderText("Enter SKU name, example: 185_70_R14_AMZ4G")

        self.refresh_lasers_btn = QPushButton("Refresh Lasers")
        self.load_laser_profile_btn = QPushButton("Load Laser Profile")
        self.save_laser_profile_btn = QPushButton("Save Laser Profile")

        top_row.addWidget(QLabel("SKU:"))
        top_row.addWidget(self.laser_sku_input)
        top_row.addWidget(self.refresh_lasers_btn)
        top_row.addWidget(self.load_laser_profile_btn)
        top_row.addWidget(self.save_laser_profile_btn)

        layout.addLayout(top_row)

        # Laser table
        self.laser_table = QTableWidget()
        self.laser_table.setColumnCount(6)
        self.laser_table.setHorizontalHeaderLabels([
            "Laser ID",
            "Laser Name",
            "Model",
            "Status",
            "Assigned Zone",
            "Enabled"
        ])
        self.laser_table.horizontalHeader().setStretchLastSection(True)
        self.laser_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.laser_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.laser_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.laser_table.setMinimumHeight(90)
        self.laser_table.setMaximumHeight(150)
        layout.addWidget(self.laser_table)

        bottom_row = QHBoxLayout()

        self.laser_settings_box = self.create_laser_settings_box()
        self.laser_preview_box = self.create_laser_preview_box()

        # Make settings panel scrollable to avoid Qt window geometry issue
        laser_settings_scroll = self.make_scrollable_widget(
            self.laser_settings_box,
            min_width=520
        )

        bottom_row.addWidget(laser_settings_scroll, 1)
        bottom_row.addWidget(self.laser_preview_box, 2)

        layout.addLayout(bottom_row, 1)

        self.refresh_lasers_btn.clicked.connect(self.refresh_lasers)
        self.load_laser_profile_btn.clicked.connect(self.load_laser_profile)
        self.save_laser_profile_btn.clicked.connect(self.save_laser_profile)
        self.laser_table.cellClicked.connect(self.on_laser_selected)

    def create_settings_box(self):
        box = QGroupBox("Selected Camera Settings")
        layout = QFormLayout(box)

        self.selected_camera_label = QLabel("-")

        self.hardware_trigger_checkbox = QCheckBox("Use Hardware Trigger Settings for Production")
        self.hardware_trigger_checkbox.setChecked(True)
        self.hardware_trigger_checkbox.stateChanged.connect(self.on_mode_changed)

        self.mode_status_label = QLabel("Mode: Hardware Trigger Line0")
        self.mode_status_label.setStyleSheet("font-weight: bold; color: #8a2be2;")

        self.width_input = QLineEdit("4096")
        self.height_input = QLineEdit("6000")

        self.pixel_format_combo = QComboBox()
        self.pixel_format_combo.addItems(["Mono16", "Mono8"])

        self.exposure_input = QLineEdit("150.0")
        self.gain_input = QLineEdit("0.0")

        self.line_rate_input = QLineEdit("4096.0")

        self.acquisition_mode_combo = QComboBox()
        self.acquisition_mode_combo.addItems(["Continuous"])

        self.line_selector_combo = QComboBox()
        self.line_selector_combo.addItems(["Line0"])

        self.line_mode_combo = QComboBox()
        self.line_mode_combo.addItems(["Input"])

        self.line_source_combo = QComboBox()
        self.line_source_combo.addItems(["Off"])

        self.trigger_selector_combo = QComboBox()
        self.trigger_selector_combo.addItems(["AcquisitionStart"])

        self.trigger_source_combo = QComboBox()
        self.trigger_source_combo.addItems(["Line0"])

        self.trigger_activation_combo = QComboBox()
        self.trigger_activation_combo.addItems(["RisingEdge", "FallingEdge"])

        self.packet_size_input = QLineEdit("1500")

        self.live_line_count_label = QLabel("0 / 6000")

        self.apply_settings_btn = QPushButton("Apply Settings")
        self.start_preview_btn = QPushButton("Start Live Preview")
        self.stop_preview_btn = QPushButton("Stop Live Preview")
        self.capture_one_btn = QPushButton("Capture One Image")

        self.stop_preview_btn.setEnabled(False)

        layout.addRow("Selected Serial:", self.selected_camera_label)
        layout.addRow(self.hardware_trigger_checkbox)
        layout.addRow("Current Mode:", self.mode_status_label)

        layout.addRow("Width:", self.width_input)
        layout.addRow("Height:", self.height_input)
        layout.addRow("PixelFormat:", self.pixel_format_combo)

        layout.addRow("ExposureAuto:", QLabel("Off"))
        layout.addRow("ExposureTime:", self.exposure_input)
        layout.addRow("GainAuto:", QLabel("Off"))
        layout.addRow("Gain:", self.gain_input)

        layout.addRow("AcquisitionLineRateEnable:", QLabel("True"))
        layout.addRow("AcquisitionLineRate:", self.line_rate_input)

        layout.addRow("AcquisitionMode:", self.acquisition_mode_combo)

        layout.addRow("LineSelector:", self.line_selector_combo)
        layout.addRow("LineMode:", self.line_mode_combo)
        layout.addRow("LineSource:", self.line_source_combo)

        layout.addRow("TriggerSelector:", self.trigger_selector_combo)
        layout.addRow("TriggerSource:", self.trigger_source_combo)
        layout.addRow("TriggerActivation:", self.trigger_activation_combo)
        layout.addRow("TriggerMode:", QLabel("Auto: On for Hardware, Off for Preview"))

        layout.addRow("GevSCPSPacketSize:", self.packet_size_input)
        layout.addRow("Live Line Count:", self.live_line_count_label)

        layout.addRow(self.apply_settings_btn)
        layout.addRow(self.start_preview_btn)
        layout.addRow(self.stop_preview_btn)
        layout.addRow(self.capture_one_btn)

        self.apply_settings_btn.clicked.connect(self.apply_settings_to_selected)
        self.start_preview_btn.clicked.connect(self.start_live_preview)
        self.stop_preview_btn.clicked.connect(self.stop_live_preview)
        self.capture_one_btn.clicked.connect(self.capture_one_image)

        return box

    def create_preview_box(self):
        box = QGroupBox("Live Camera Preview")
        layout = QVBoxLayout(box)

        self.preview_label = QLabel("No Image")
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setMinimumHeight(520)
        self.preview_label.setStyleSheet("""
            QLabel {
                background-color: #111;
                color: white;
                border-radius: 8px;
                font-size: 16px;
            }
        """)

        self.capture_status_label = QLabel("Status: Waiting")
        self.capture_status_label.setStyleSheet("font-weight: bold;")

        self.preview_help_label = QLabel(
            "For image quality check: uncheck Hardware Trigger and click Start Live Preview.\n"
            "For production setting: check Hardware Trigger, click Apply Settings, then Save Profile."
        )
        self.preview_help_label.setStyleSheet("color: gray;")

        layout.addWidget(self.preview_label)
        layout.addWidget(self.capture_status_label)
        layout.addWidget(self.preview_help_label)

        return box

    # ------------------------------------------------------------------
    # Mode handling
    # ------------------------------------------------------------------
    def get_current_mode(self):
        if self.hardware_trigger_checkbox.isChecked():
            return "hardware"
        return "preview_free_run"

    def on_mode_changed(self):
        if self.hardware_trigger_checkbox.isChecked():
            self.mode_status_label.setText("Mode: Hardware Trigger Line0")
            self.capture_status_label.setText(
                "Status: Hardware mode selected. Camera will wait for Line0 trigger."
            )
        else:
            self.mode_status_label.setText("Mode: Software / Free-run Preview")
            self.capture_status_label.setText(
                "Status: Preview mode selected. No Line0 trigger required."
            )

    # ------------------------------------------------------------------
    # Camera discovery
    # ------------------------------------------------------------------
    def refresh_cameras(self):
        if self.live_worker:
            self.stop_live_preview()

        cameras = self.camera_manager.refresh_cameras()

        self.camera_table.setRowCount(0)

        for cam in cameras:
            row = self.camera_table.rowCount()
            self.camera_table.insertRow(row)

            self.camera_table.setItem(row, 0, QTableWidgetItem(cam.serial))
            self.camera_table.setItem(row, 1, QTableWidgetItem(cam.model))
            self.camera_table.setItem(row, 2, QTableWidgetItem(cam.ip))
            self.camera_table.setItem(row, 3, QTableWidgetItem(cam.status))

            zone_combo = QComboBox()
            zone_combo.addItems(["Unassigned"] + ZONE_NAMES)
            self.camera_table.setCellWidget(row, 4, zone_combo)

            enabled_checkbox = QCheckBox()
            enabled_checkbox.setChecked(True)
            enabled_checkbox.setStyleSheet("margin-left: 20px;")
            self.camera_table.setCellWidget(row, 5, enabled_checkbox)

            if cam.serial not in self.camera_settings_by_serial:
                self.camera_settings_by_serial[cam.serial] = DEFAULT_CAMERA_SETTINGS.copy()
                self.camera_settings_by_serial[cam.serial]["serial"] = cam.serial

        self.capture_status_label.setText(f"Status: Found {len(cameras)} camera(s)")

    def on_camera_selected(self, row, col):
        serial_item = self.camera_table.item(row, 0)

        if not serial_item:
            return

        if self.selected_serial:
            self.save_form_to_memory(self.selected_serial)

        serial = serial_item.text()
        self.selected_serial = serial

        if serial not in self.camera_settings_by_serial:
            self.camera_settings_by_serial[serial] = DEFAULT_CAMERA_SETTINGS.copy()
            self.camera_settings_by_serial[serial]["serial"] = serial

        self.load_memory_to_form(serial)

    # ------------------------------------------------------------------
    # Settings memory
    # ------------------------------------------------------------------
    def save_form_to_memory(self, serial):
        try:
            settings = {
                "serial": serial,
                "enabled": True,

                "use_hardware_trigger": self.hardware_trigger_checkbox.isChecked(),

                # Geometry
                "width": int(self.width_input.text()),
                "height": int(self.height_input.text()),
                "pixel_format": self.pixel_format_combo.currentText(),

                # Exposure / gain
                "exposure_auto": "Off",
                "exposure_time": float(self.exposure_input.text()),
                "gain_auto": "Off",
                "gain": float(self.gain_input.text()),

                # Line rate
                "acquisition_line_rate_enable": True,
                "acquisition_line_rate": float(self.line_rate_input.text()),

                # Acquisition
                "acquisition_mode": self.acquisition_mode_combo.currentText(),

                # Hardware trigger settings
                "line_selector": self.line_selector_combo.currentText(),
                "line_mode": self.line_mode_combo.currentText(),
                "line_source": self.line_source_combo.currentText(),
                "trigger_selector": self.trigger_selector_combo.currentText(),
                "trigger_source": self.trigger_source_combo.currentText(),
                "trigger_activation": self.trigger_activation_combo.currentText(),
                "trigger_mode": "On" if self.hardware_trigger_checkbox.isChecked() else "Off",

                # Network
                "packet_size": int(self.packet_size_input.text()),
            }

            self.camera_settings_by_serial[serial] = settings

        except Exception as e:
            QMessageBox.warning(self, "Invalid Settings", str(e))

    def load_memory_to_form(self, serial):
        settings = self.camera_settings_by_serial.get(
            serial,
            DEFAULT_CAMERA_SETTINGS.copy()
        )

        self.selected_camera_label.setText(serial)

        self.hardware_trigger_checkbox.setChecked(
            bool(settings.get("use_hardware_trigger", True))
        )

        self.width_input.setText(str(settings.get("width", 4096)))
        self.height_input.setText(str(settings.get("height", 6000)))
        self.pixel_format_combo.setCurrentText(settings.get("pixel_format", "Mono16"))

        self.exposure_input.setText(str(settings.get("exposure_time", 150.0)))
        self.gain_input.setText(str(settings.get("gain", 0.0)))

        self.line_rate_input.setText(str(settings.get("acquisition_line_rate", 4096.0)))

        self.acquisition_mode_combo.setCurrentText(
            settings.get("acquisition_mode", "Continuous")
        )

        self.line_selector_combo.setCurrentText(
            settings.get("line_selector", "Line0")
        )
        self.line_mode_combo.setCurrentText(
            settings.get("line_mode", "Input")
        )
        self.line_source_combo.setCurrentText(
            settings.get("line_source", "Off")
        )

        self.trigger_selector_combo.setCurrentText(
            settings.get("trigger_selector", "AcquisitionStart")
        )
        self.trigger_source_combo.setCurrentText(
            settings.get("trigger_source", "Line0")
        )
        self.trigger_activation_combo.setCurrentText(
            settings.get("trigger_activation", "RisingEdge")
        )

        self.packet_size_input.setText(str(settings.get("packet_size", 1500)))

        self.live_line_count_label.setText(
            f"0 / {settings.get('height', 6000)}"
        )

        self.on_mode_changed()

    def get_selected_settings(self):
        if not self.selected_serial:
            raise RuntimeError("No camera selected")

        self.save_form_to_memory(self.selected_serial)
        return self.camera_settings_by_serial[self.selected_serial]

    # ------------------------------------------------------------------
    # Apply settings
    # ------------------------------------------------------------------
    def apply_settings_to_selected(self):
        if not self.selected_serial:
            QMessageBox.warning(self, "No Camera", "Please select a camera first.")
            return

        settings = self.get_selected_settings()
        mode = self.get_current_mode()

        ok, msg = self.camera_manager.apply_settings(
            self.selected_serial,
            settings,
            mode=mode
        )

        if ok:
            self.capture_status_label.setText(f"Status: {msg}")
        else:
            QMessageBox.warning(self, "Apply Failed", msg)

    # ------------------------------------------------------------------
    # Live preview
    # ------------------------------------------------------------------
    def start_live_preview(self):
        if not self.selected_serial:
            QMessageBox.warning(self, "No Camera", "Please select a camera first.")
            return

        if self.live_worker:
            QMessageBox.warning(self, "Preview Running", "Live preview is already running.")
            return

        settings = self.get_selected_settings()
        mode = self.get_current_mode()

        self.start_preview_btn.setEnabled(False)
        self.stop_preview_btn.setEnabled(True)
        self.capture_one_btn.setEnabled(False)

        if mode == "preview_free_run":
            self.capture_status_label.setText(
                "Status: Starting software/free-run preview..."
            )
        else:
            self.capture_status_label.setText(
                "Status: Starting hardware preview. Waiting for Line0 trigger..."
            )

        self.live_worker = CameraLivePreviewWorker(
            self.camera_manager,
            self.selected_serial,
            settings,
            mode=mode
        )

        self.live_worker.frame_ready.connect(self.on_live_frame_ready)
        self.live_worker.status_signal.connect(self.on_live_status)
        self.live_worker.error_signal.connect(self.on_live_error)
        self.live_worker.finished.connect(self.on_live_worker_finished)

        self.live_worker.start()

    def on_live_worker_finished(self):
        self.live_worker = None

        self.start_preview_btn.setEnabled(True)
        self.stop_preview_btn.setEnabled(False)
        self.capture_one_btn.setEnabled(True)
        
    def stop_live_preview(self):
        if self.live_worker:
            self.live_worker.stop()
            self.live_worker.wait(3000)
            self.live_worker = None

        self.start_preview_btn.setEnabled(True)
        self.stop_preview_btn.setEnabled(False)
        self.capture_one_btn.setEnabled(True)

        self.capture_status_label.setText("Status: Live preview stopped")

    def on_live_frame_ready(self, qimg, line_count):
        pixmap = QPixmap.fromImage(qimg)

        w = max(self.preview_label.width(), 800)
        h = max(self.preview_label.height(), 500)

        scaled = pixmap.scaled(
            w,
            h,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )

        self.preview_label.setPixmap(scaled)

        settings = self.get_selected_settings()
        expected_height = settings.get("height", 6000)

        self.live_line_count_label.setText(
            f"{line_count} / {expected_height}"
        )

    def on_live_status(self, msg):
        self.capture_status_label.setText(f"Status: {msg}")

    def on_live_error(self, error_msg):
        self.start_preview_btn.setEnabled(True)
        self.stop_preview_btn.setEnabled(False)
        self.capture_one_btn.setEnabled(True)

        self.capture_status_label.setText("Status: Live preview error")
        QMessageBox.critical(self, "Live Preview Error", error_msg)

    # ------------------------------------------------------------------
    # Capture one image
    # ------------------------------------------------------------------
    def capture_one_image(self):
        if not self.selected_serial:
            QMessageBox.warning(self, "No Camera", "Please select a camera first.")
            return

        if self.live_worker:
            QMessageBox.warning(
                self,
                "Live Preview Running",
                "Stop live preview before Capture One Image."
            )
            return

        settings = self.get_selected_settings()
        mode = self.get_current_mode()

        self.capture_one_btn.setEnabled(False)
        self.start_preview_btn.setEnabled(False)

        if mode == "preview_free_run":
            self.capture_status_label.setText(
                "Status: Capturing one image in software/free-run mode..."
            )
        else:
            self.capture_status_label.setText(
                "Status: Capturing one image in hardware mode. Send Line0 trigger..."
            )

        self.capture_worker = CameraCaptureWorker(
            self.camera_manager,
            self.selected_serial,
            settings,
            mode=mode
        )

        self.capture_worker.capture_done.connect(self.on_capture_done)
        self.capture_worker.capture_failed.connect(self.on_capture_failed)
        self.capture_worker.start()

    def on_capture_done(self, image_path, line_count):
        self.capture_one_btn.setEnabled(True)
        self.start_preview_btn.setEnabled(True)

        settings = self.get_selected_settings()
        expected_height = settings.get("height", 6000)

        self.live_line_count_label.setText(f"{line_count} / {expected_height}")
        self.capture_status_label.setText(f"Status: Image saved: {image_path}")

        pixmap = QPixmap(image_path)

        if not pixmap.isNull():
            w = max(self.preview_label.width(), 800)
            h = max(self.preview_label.height(), 500)

            scaled = pixmap.scaled(
                w,
                h,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            self.preview_label.setPixmap(scaled)
        else:
            self.preview_label.setText("Image saved but preview failed")

    def on_capture_failed(self, error_msg):
        self.capture_one_btn.setEnabled(True)
        self.start_preview_btn.setEnabled(True)

        self.capture_status_label.setText("Status: Capture failed")
        QMessageBox.critical(self, "Capture Failed", error_msg)

    # ------------------------------------------------------------------
    # Save / load profile
    # ------------------------------------------------------------------
    def save_profile(self):
        sku = self.sku_input.text().strip()

        if not sku:
            QMessageBox.warning(self, "Missing SKU", "Please enter SKU name.")
            return

        if self.selected_serial:
            self.save_form_to_memory(self.selected_serial)

        profile = self.profile_manager.default_profile(sku)

        saved_count = 0
        unassigned_serials = []

        for row in range(self.camera_table.rowCount()):
            serial_item = self.camera_table.item(row, 0)

            if not serial_item:
                continue

            serial = serial_item.text()

            zone_combo = self.camera_table.cellWidget(row, 4)
            enabled_checkbox = self.camera_table.cellWidget(row, 5)

            zone_name = zone_combo.currentText()
            enabled = enabled_checkbox.isChecked()

            if zone_name == "Unassigned":
                unassigned_serials.append(serial)
                continue

            zone_key = ZONE_KEYS[zone_name]

            settings = self.camera_settings_by_serial.get(
                serial,
                DEFAULT_CAMERA_SETTINGS.copy()
            )

            settings["serial"] = serial
            settings["enabled"] = enabled

            # Convert Device Page height name to Live camera profile name
            if "height" in settings:
                settings["camera_height"] = int(settings.get("height", 14000))

            # These are GLOBAL for Apollo Live.
            # They come from .env / HARDWARE_TRIGGER.py, not SKU profile.
            for key in [
                "use_hardware_trigger",
                "line_selector",
                "line_mode",
                "line_source",
                "trigger_selector",
                "trigger_source",
                "trigger_activation",
                "trigger_mode",
            ]:
                settings.pop(key, None)

            profile["cameras"][zone_key] = settings
            saved_count += 1

        path = self.sku_profile_store.save_camera_profile(sku, profile)

        msg = f"Saved {saved_count} camera profile(s):\n{path}"

        if unassigned_serials:
            msg += "\n\nNot saved because zone is Unassigned:\n"
            msg += "\n".join(unassigned_serials)

        QMessageBox.information(self, "Profile Saved", msg)

    def load_profile(self):
        sku = self.sku_input.text().strip()

        if not sku:
            QMessageBox.warning(self, "Missing SKU", "Please enter SKU name.")
            return

        profile = self.sku_profile_store.load_camera_profile(sku)
        cameras_config = profile.get("cameras", {})

        for zone_name, zone_key in ZONE_KEYS.items():
            cam_cfg = cameras_config.get(zone_key, {})
            serial = cam_cfg.get("serial", "")

            if serial:
                self.camera_settings_by_serial[serial] = cam_cfg

        for row in range(self.camera_table.rowCount()):
            serial_item = self.camera_table.item(row, 0)

            if not serial_item:
                continue

            table_serial = serial_item.text()
            zone_combo = self.camera_table.cellWidget(row, 4)
            enabled_checkbox = self.camera_table.cellWidget(row, 5)

            zone_combo.setCurrentText("Unassigned")

            for zone_name, zone_key in ZONE_KEYS.items():
                cam_cfg = cameras_config.get(zone_key, {})

                if cam_cfg.get("serial", "") == table_serial:
                    zone_combo.setCurrentText(zone_name)
                    enabled_checkbox.setChecked(bool(cam_cfg.get("enabled", True)))
                    break

        self.capture_status_label.setText("Status: Profile loaded")
        QMessageBox.information(self, "Profile Loaded", f"Loaded profile for SKU: {sku}")

    def cleanup_device_page(self, destroy_devices=True):
        """
        Called when leaving Device page or closing app.
        Stops preview thread and releases camera handles.
        """

        print("[DEVICE PAGE] Cleanup started")

        # Stop live preview worker
        try:
            if getattr(self, "live_worker", None) is not None:
                if self.live_worker.isRunning():
                    self.live_worker.stop()
                    self.live_worker.wait(3000)

                self.live_worker = None
        except Exception as e:
            print(f"[DEVICE PAGE] live_worker cleanup warning: {e}")

        # If capture thread is running, wait shortly.
        # Capture usually ends after timeout/frame.
        try:
            if getattr(self, "capture_worker", None) is not None:
                if self.capture_worker.isRunning():
                    print("[DEVICE PAGE] Waiting for capture worker to finish...")
                    self.capture_worker.wait(5000)

                self.capture_worker = None
        except Exception as e:
            print(f"[DEVICE PAGE] capture_worker cleanup warning: {e}")

        # Release Arena camera handles
        if destroy_devices:
            try:
                if getattr(self, "camera_manager", None) is not None:
                    self.camera_manager.close_all()
            except Exception as e:
                print(f"[DEVICE PAGE] camera_manager cleanup warning: {e}")
        # Stop laser live worker
        try:
            if getattr(self, "laser_live_worker", None) is not None:
                if self.laser_live_worker.isRunning():
                    self.laser_live_worker.stop()
                    self.laser_live_worker.wait(3000)

                self.laser_live_worker = None
        except Exception as e:
            print(f"[DEVICE PAGE] laser_live_worker cleanup warning: {e}")

        # Wait for laser capture worker
        try:
            if getattr(self, "laser_capture_worker", None) is not None:
                if self.laser_capture_worker.isRunning():
                    print("[DEVICE PAGE] Waiting for laser capture worker to finish...")
                    self.laser_capture_worker.wait(5000)

                self.laser_capture_worker = None
        except Exception as e:
            print(f"[DEVICE PAGE] laser_capture_worker cleanup warning: {e}")

        # Release laser handles
        try:
            if getattr(self, "laser_manager", None) is not None:
                self.laser_manager.close_all()
        except Exception as e:
            print(f"[DEVICE PAGE] laser_manager cleanup warning: {e}")

        # Reset buttons safely
        try:
            self.start_preview_btn.setEnabled(True)
            self.stop_preview_btn.setEnabled(False)
            self.capture_one_btn.setEnabled(True)
            self.capture_status_label.setText("Status: Device page closed safely")
        except Exception:
            pass

        print("[DEVICE PAGE] Cleanup completed")

    def closeEvent(self, event):
        self.cleanup_device_page(destroy_devices=True)
        event.accept()
    
    def create_laser_settings_box(self):
        box = QGroupBox("Selected Laser Settings")
        layout = QFormLayout(box)

        self.selected_laser_label = QLabel("-")

        self.laser_use_user_set_checkbox = QCheckBox("Load UserSet before applying settings")
        self.laser_use_user_set_checkbox.setChecked(False)

        self.laser_user_set_input = QLineEdit("UserSet1")

        self.laser_device_output_combo = QComboBox()
        self.laser_device_output_combo.addItems(["Linescan3D"])

        self.laser_scan3d_data_type_combo = QComboBox()
        self.laser_scan3d_data_type_combo.addItems(["UniformX Z"])

        self.laser_profiles_per_scan_input = QLineEdit("1")

        self.laser_scan_rate_input = QLineEdit("4000.0")
        self.laser_exposure_input = QLineEdit("100.0")

        self.laser_trigger_mode_combo = QComboBox()
        self.laser_trigger_mode_combo.addItems(["Off", "On"])

        self.laser_trigger_source_combo = QComboBox()
        self.laser_trigger_source_combo.addItems(["Software", "Line0", "Encoder"])

        self.laser_packet_size_input = QLineEdit("9000")
        self.laser_invalid_value_input = QLineEdit("")

        self.laser_range_mode_combo = QComboBox()
        self.laser_range_mode_combo.addItems(["Near", "Mid", "Far"])

        self.laser_resolution_combo = QComboBox()
        self.laser_resolution_combo.addItems(["High", "Medium", "Low"])

        self.laser_roi_x_start_input = QLineEdit("0")
        self.laser_roi_width_input = QLineEdit("4096")
        self.laser_roi_z_start_input = QLineEdit("0")
        self.laser_roi_height_input = QLineEdit("2048")

        self.laser_profile_avg_input = QLineEdit("1")
        self.laser_threshold_input = QLineEdit("50.0")

        self.laser_x_scale_input = QLineEdit("1.0")
        self.laser_z_scale_input = QLineEdit("1.0")

        self.laser_aspect_lock_checkbox = QCheckBox("Lock Aspect Ratio")
        self.laser_aspect_lock_checkbox.setChecked(True)

        self.laser_output_format_combo = QComboBox()
        self.laser_output_format_combo.addItems(["Profile", "Point Cloud"])

        self.apply_laser_settings_btn = QPushButton("Apply Laser Settings")
        self.start_laser_preview_btn = QPushButton("Start Live Profile")
        self.stop_laser_preview_btn = QPushButton("Stop Live Profile")
        self.capture_laser_profile_btn = QPushButton("Capture One Profile")

        self.stop_laser_preview_btn.setEnabled(False)

        layout.addRow("Selected Laser:", self.selected_laser_label)
        layout.addRow(self.laser_use_user_set_checkbox)
        layout.addRow("User Set:", self.laser_user_set_input)
        layout.addRow("Device Output Type:", self.laser_device_output_combo)
        layout.addRow("3D Data Type:", self.laser_scan3d_data_type_combo)
        layout.addRow("Profiles Per Scan:", self.laser_profiles_per_scan_input)
        layout.addRow("Trigger Mode:", self.laser_trigger_mode_combo)
        layout.addRow("Trigger Source:", self.laser_trigger_source_combo)
        layout.addRow("Packet Size:", self.laser_packet_size_input)
        layout.addRow("Invalid Raw Value:", self.laser_invalid_value_input)
        layout.addRow("Scan Rate:", self.laser_scan_rate_input)
        layout.addRow("Exposure:", self.laser_exposure_input)
        layout.addRow("Range Mode:", self.laser_range_mode_combo)
        layout.addRow("Resolution:", self.laser_resolution_combo)

        layout.addRow("ROI X Start:", self.laser_roi_x_start_input)
        layout.addRow("ROI Width:", self.laser_roi_width_input)
        layout.addRow("ROI Z Start:", self.laser_roi_z_start_input)
        layout.addRow("ROI Height:", self.laser_roi_height_input)

        layout.addRow("Profile Averaging:", self.laser_profile_avg_input)
        layout.addRow("Threshold:", self.laser_threshold_input)

        layout.addRow("X Scale:", self.laser_x_scale_input)
        layout.addRow("Z Scale:", self.laser_z_scale_input)
        layout.addRow(self.laser_aspect_lock_checkbox)

        layout.addRow("Output Format:", self.laser_output_format_combo)

        layout.addRow(self.apply_laser_settings_btn)
        layout.addRow(self.start_laser_preview_btn)
        layout.addRow(self.stop_laser_preview_btn)
        layout.addRow(self.capture_laser_profile_btn)

        self.apply_laser_settings_btn.clicked.connect(self.apply_laser_settings_to_selected)
        self.start_laser_preview_btn.clicked.connect(self.start_laser_live_profile)
        self.stop_laser_preview_btn.clicked.connect(self.stop_laser_live_profile)
        self.capture_laser_profile_btn.clicked.connect(self.capture_one_laser_profile)

        for widget in box.findChildren(QLineEdit):
            widget.setMinimumHeight(22)

        for widget in box.findChildren(QComboBox):
            widget.setMinimumHeight(22)

        for widget in box.findChildren(QPushButton):
            widget.setMinimumHeight(26)

        box.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        return box


    def create_laser_preview_box(self):
        box = QGroupBox("Live 2D Laser Profile + Quality Metrics")
        layout = QVBoxLayout(box)

        self.laser_preview_label = QLabel("No Laser Profile")
        self.laser_preview_label.setAlignment(Qt.AlignCenter)
        self.laser_preview_label.setMinimumHeight(300)
        self.laser_preview_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.laser_preview_label.setStyleSheet("""
            QLabel {
                background-color: #111;
                color: white;
                border-radius: 8px;
                font-size: 16px;
            }
        """)

        self.laser_status_label = QLabel("Status: Waiting")
        self.laser_status_label.setStyleSheet("font-weight: bold;")

        self.laser_metrics_label = QLabel(
            "Valid Points: -\n"
            "Missing Points: -\n"
            "Outliers: -\n"
            "Z Range: -\n"
            "SNR: -\n"
            "Decision: -"
        )
        self.laser_metrics_label.setStyleSheet("""
            QLabel {
                background: #f7f7f7;
                color: #222;
                border-radius: 8px;
                padding: 8px;
                font: 12px 'Segoe UI';
            }
        """)

        help_label = QLabel(
            "Use Start Live Profile to verify laser shape.\n"
            "Use Capture One Profile to save .npy, .csv, .png, and metrics .json."
        )
        help_label.setStyleSheet("color: gray;")

        layout.addWidget(self.laser_preview_label)
        layout.addWidget(self.laser_status_label)
        layout.addWidget(self.laser_metrics_label)
        layout.addWidget(help_label)

        return box

    def refresh_lasers(self):
        if self.laser_live_worker:
            self.stop_laser_live_profile()

        lasers = self.laser_manager.refresh_lasers()

        self.laser_table.setRowCount(0)

        for laser in lasers:
            row = self.laser_table.rowCount()
            self.laser_table.insertRow(row)

            self.laser_table.setItem(row, 0, QTableWidgetItem(laser.laser_id))
            self.laser_table.setItem(row, 1, QTableWidgetItem(laser.laser_name))
            self.laser_table.setItem(row, 2, QTableWidgetItem(laser.model))
            self.laser_table.setItem(row, 3, QTableWidgetItem(laser.status))

            zone_combo = QComboBox()
            zone_combo.addItems(["Unassigned"] + LASER_ZONE_NAMES)
            self.laser_table.setCellWidget(row, 4, zone_combo)

            enabled_checkbox = QCheckBox()
            enabled_checkbox.setChecked(True)
            enabled_checkbox.setStyleSheet("margin-left: 20px;")
            self.laser_table.setCellWidget(row, 5, enabled_checkbox)

            if laser.laser_id not in self.laser_settings_by_id:
                self.laser_settings_by_id[laser.laser_id] = DEFAULT_LASER_SETTINGS.copy()
                self.laser_settings_by_id[laser.laser_id]["laser_id"] = laser.laser_id
                self.laser_settings_by_id[laser.laser_id]["laser_name"] = laser.laser_name

        self.laser_status_label.setText(f"Status: Found {len(lasers)} laser(s)")


    def on_laser_selected(self, row, col):
        laser_item = self.laser_table.item(row, 0)

        if not laser_item:
            return

        if self.selected_laser_id:
            self.save_laser_form_to_memory(self.selected_laser_id)

        laser_id = laser_item.text()
        self.selected_laser_id = laser_id

        if laser_id not in self.laser_settings_by_id:
            self.laser_settings_by_id[laser_id] = DEFAULT_LASER_SETTINGS.copy()
            self.laser_settings_by_id[laser_id]["laser_id"] = laser_id

        self.load_laser_memory_to_form(laser_id)


    def save_laser_form_to_memory(self, laser_id):
        try:
            laser_name = laser_id

            for row in range(self.laser_table.rowCount()):
                laser_item = self.laser_table.item(row, 0)
                name_item = self.laser_table.item(row, 1)

                if laser_item and laser_item.text() == laser_id and name_item:
                    laser_name = name_item.text()
                    break

            settings = {
                "laser_id": laser_id,
                "laser_name": laser_name,
                "enabled": True,

                # Direct GUI configuration / optional UserSet
                "use_user_set": self.laser_use_user_set_checkbox.isChecked(),
                "user_set": self.laser_user_set_input.text().strip() or "UserSet1",

                # Z-Trak output settings
                "device_output_type": self.laser_device_output_combo.currentText(),
                "scan3d_data_type": self.laser_scan3d_data_type_combo.currentText(),
                "profiles_per_scan": int(self.laser_profiles_per_scan_input.text()),

                # Acquisition
                "scan_rate": float(self.laser_scan_rate_input.text()),
                "exposure": float(self.laser_exposure_input.text()),
                "range_mode": self.laser_range_mode_combo.currentText(),
                "resolution": self.laser_resolution_combo.currentText(),

                # ROI
                "roi_x_start": int(self.laser_roi_x_start_input.text()),
                "roi_width": int(self.laser_roi_width_input.text()),
                "roi_z_start": int(self.laser_roi_z_start_input.text()),
                "roi_height": int(self.laser_roi_height_input.text()),

                # Filtering
                "profile_averaging": int(self.laser_profile_avg_input.text()),
                "threshold": float(self.laser_threshold_input.text()),

                # Trigger / network
                "trigger_mode": self.laser_trigger_mode_combo.currentText(),
                "trigger_source": self.laser_trigger_source_combo.currentText(),
                "trigger_activation": "RisingEdge",
                "packet_size": int(self.laser_packet_size_input.text()),
                "invalid_value": self.laser_invalid_value_input.text().strip(),

                # Display
                "x_scale": float(self.laser_x_scale_input.text()),
                "z_scale": float(self.laser_z_scale_input.text()),
                "aspect_lock": self.laser_aspect_lock_checkbox.isChecked(),

                # Output
                "output_format": self.laser_output_format_combo.currentText(),
            }

            self.laser_settings_by_id[laser_id] = settings

        except Exception as e:
            QMessageBox.warning(self, "Invalid Laser Settings", str(e))
            
    def make_scrollable_widget(self, widget, min_width=420):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(widget)
        scroll.setMinimumWidth(min_width)
        scroll.setFrameShape(QFrame.NoFrame) 
        scroll.setStyleSheet("""
            QScrollArea {
                background: transparent;
                border: none;
            }
            QScrollBar:vertical {
                background: #f1f1f1;
                width: 10px;
                border-radius: 5px;
            }
            QScrollBar::handle:vertical {
                background: #b5b5b5;
                border-radius: 5px;
                min-height: 30px;
            }
        """)
        return scroll

    def load_laser_memory_to_form(self, laser_id):
        settings = self.laser_settings_by_id.get(
            laser_id,
            DEFAULT_LASER_SETTINGS.copy()
        )

        self.selected_laser_label.setText(laser_id)

        self.laser_use_user_set_checkbox.setChecked(
            bool(settings.get("use_user_set", False))
        )

        self.laser_user_set_input.setText(
            str(settings.get("user_set", "UserSet1"))
        )

        self.laser_device_output_combo.setCurrentText(
            settings.get("device_output_type", "Linescan3D")
        )

        self.laser_scan3d_data_type_combo.setCurrentText(
            settings.get("scan3d_data_type", "UniformX Z")
        )

        self.laser_profiles_per_scan_input.setText(
            str(settings.get("profiles_per_scan", 1))
        )

        self.laser_scan_rate_input.setText(str(settings.get("scan_rate", 4000.0)))
        self.laser_exposure_input.setText(str(settings.get("exposure", 100.0)))

        self.laser_range_mode_combo.setCurrentText(settings.get("range_mode", "Mid"))
        self.laser_resolution_combo.setCurrentText(settings.get("resolution", "High"))

        self.laser_roi_x_start_input.setText(str(settings.get("roi_x_start", 0)))
        self.laser_roi_width_input.setText(str(settings.get("roi_width", 4096)))
        self.laser_roi_z_start_input.setText(str(settings.get("roi_z_start", 0)))
        self.laser_roi_height_input.setText(str(settings.get("roi_height", 2048)))

        self.laser_profile_avg_input.setText(str(settings.get("profile_averaging", 1)))
        self.laser_threshold_input.setText(str(settings.get("threshold", 50.0)))

        self.laser_trigger_mode_combo.setCurrentText(
            settings.get("trigger_mode", "Off")
        )

        self.laser_trigger_source_combo.setCurrentText(
            settings.get("trigger_source", "Software")
        )

        self.laser_packet_size_input.setText(
            str(settings.get("packet_size", 9000))
        )

        self.laser_invalid_value_input.setText(
            str(settings.get("invalid_value", ""))
        )

        self.laser_x_scale_input.setText(str(settings.get("x_scale", 1.0)))
        self.laser_z_scale_input.setText(str(settings.get("z_scale", 1.0)))
        self.laser_aspect_lock_checkbox.setChecked(bool(settings.get("aspect_lock", True)))

        self.laser_output_format_combo.setCurrentText(settings.get("output_format", "Profile"))


    def get_selected_laser_settings(self):
        if not self.selected_laser_id:
            raise RuntimeError("No laser selected")

        self.save_laser_form_to_memory(self.selected_laser_id)
        return self.laser_settings_by_id[self.selected_laser_id]


    def apply_laser_settings_to_selected(self):
        if not self.selected_laser_id:
            QMessageBox.warning(self, "No Laser", "Please select a laser first.")
            return

        settings = self.get_selected_laser_settings()

        ok, msg = self.laser_manager.apply_settings(
            self.selected_laser_id,
            settings
        )

        if ok:
            self.laser_status_label.setText(f"Status: {msg}")
        else:
            QMessageBox.warning(self, "Laser Apply Failed", msg)


    def start_laser_live_profile(self):
        if not self.selected_laser_id:
            QMessageBox.warning(self, "No Laser", "Please select a laser first.")
            return

        if self.laser_live_worker:
            QMessageBox.warning(self, "Laser Preview Running", "Laser live profile is already running.")
            return

        settings = self.get_selected_laser_settings()

        self.start_laser_preview_btn.setEnabled(False)
        self.stop_laser_preview_btn.setEnabled(True)
        self.capture_laser_profile_btn.setEnabled(False)

        self.laser_status_label.setText("Status: Starting laser live profile...")

        self.laser_live_worker = LaserLiveProfileWorker(
            self.laser_manager,
            self.selected_laser_id,
            settings
        )

        self.laser_live_worker.frame_ready.connect(self.on_laser_frame_ready)
        self.laser_live_worker.status_signal.connect(self.on_laser_status)
        self.laser_live_worker.error_signal.connect(self.on_laser_error)

        self.laser_live_worker.start()


    def stop_laser_live_profile(self):
        if self.laser_live_worker:
            self.laser_live_worker.stop()
            self.laser_live_worker.wait(3000)
            self.laser_live_worker = None

        self.start_laser_preview_btn.setEnabled(True)
        self.stop_laser_preview_btn.setEnabled(False)
        self.capture_laser_profile_btn.setEnabled(True)

        self.laser_status_label.setText("Status: Laser live profile stopped")


    def on_laser_frame_ready(self, qimg, metrics):
        pixmap = QPixmap.fromImage(qimg)

        scaled = pixmap.scaled(
            self.laser_preview_label.width(),
            self.laser_preview_label.height(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )

        self.laser_preview_label.setPixmap(scaled)
        self.update_laser_metrics_label(metrics)


    def on_laser_status(self, msg):
        self.laser_status_label.setText(f"Status: {msg}")


    def on_laser_error(self, error_msg):
        self.start_laser_preview_btn.setEnabled(True)
        self.stop_laser_preview_btn.setEnabled(False)
        self.capture_laser_profile_btn.setEnabled(True)
        self.laser_live_worker = None

        self.laser_status_label.setText("Status: Laser live profile error")
        QMessageBox.critical(self, "Laser Live Profile Error", error_msg)


    def update_laser_metrics_label(self, metrics):
        decision = metrics.get("decision", "-")

        self.laser_metrics_label.setText(
            f"Valid Points: {metrics.get('valid_points_percent', '-')} %\n"
            f"Missing Points: {metrics.get('missing_points_percent', '-')} %\n"
            f"Outliers: {metrics.get('outlier_points_percent', '-')} %\n"
            f"Z Range: {metrics.get('z_range', '-')}\n"
            f"SNR: {metrics.get('snr_score', '-')}\n"
            f"Decision: {decision}\n"
            f"Reason: {metrics.get('reason', '-')}"
        )

        if decision == "ACCEPT":
            self.laser_metrics_label.setStyleSheet("""
                QLabel {
                    background: #e8fff0;
                    color: #0b6b2b;
                    border-radius: 8px;
                    padding: 8px;
                    font: 12px 'Segoe UI';
                }
            """)
        elif decision == "REJECT":
            self.laser_metrics_label.setStyleSheet("""
                QLabel {
                    background: #ffecec;
                    color: #a00000;
                    border-radius: 8px;
                    padding: 8px;
                    font: 12px 'Segoe UI';
                }
            """)
        else:
            self.laser_metrics_label.setStyleSheet("""
                QLabel {
                    background: #f7f7f7;
                    color: #222;
                    border-radius: 8px;
                    padding: 8px;
                    font: 12px 'Segoe UI';
                }
            """)


    def capture_one_laser_profile(self):
        if not self.selected_laser_id:
            QMessageBox.warning(self, "No Laser", "Please select a laser first.")
            return

        if self.laser_live_worker:
            QMessageBox.warning(
                self,
                "Laser Preview Running",
                "Stop live profile before Capture One Profile."
            )
            return

        settings = self.get_selected_laser_settings()

        self.capture_laser_profile_btn.setEnabled(False)
        self.start_laser_preview_btn.setEnabled(False)
        self.laser_status_label.setText("Status: Capturing one laser profile...")

        self.laser_capture_worker = LaserCaptureWorker(
            self.laser_manager,
            self.selected_laser_id,
            settings
        )

        self.laser_capture_worker.capture_done.connect(self.on_laser_capture_done)
        self.laser_capture_worker.capture_failed.connect(self.on_laser_capture_failed)
        self.laser_capture_worker.start()


    def on_laser_capture_done(self, result):
        self.capture_laser_profile_btn.setEnabled(True)
        self.start_laser_preview_btn.setEnabled(True)

        png_path = result.get("png_path", "")
        metrics = result.get("metrics", {})

        self.laser_status_label.setText(f"Status: Laser profile saved: {png_path}")
        self.update_laser_metrics_label(metrics)

        pixmap = QPixmap(png_path)

        if not pixmap.isNull():
            scaled = pixmap.scaled(
                self.laser_preview_label.width(),
                self.laser_preview_label.height(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            self.laser_preview_label.setPixmap(scaled)
        else:
            self.laser_preview_label.setText("Profile saved but preview failed")


    def on_laser_capture_failed(self, error_msg):
        self.capture_laser_profile_btn.setEnabled(True)
        self.start_laser_preview_btn.setEnabled(True)

        self.laser_status_label.setText("Status: Laser capture failed")
        QMessageBox.critical(self, "Laser Capture Failed", error_msg)
    
    def save_laser_profile(self):
        sku = self.laser_sku_input.text().strip()

        if not sku:
            QMessageBox.warning(self, "Missing SKU", "Please enter SKU name.")
            return

        if self.selected_laser_id:
            self.save_laser_form_to_memory(self.selected_laser_id)

        profile = self.laser_profile_manager.default_profile(sku)

        saved_count = 0
        unassigned_lasers = []

        for row in range(self.laser_table.rowCount()):
            laser_item = self.laser_table.item(row, 0)
            name_item = self.laser_table.item(row, 1)

            if not laser_item:
                continue

            laser_id = laser_item.text()
            laser_name = name_item.text() if name_item else laser_id

            zone_combo = self.laser_table.cellWidget(row, 4)
            enabled_checkbox = self.laser_table.cellWidget(row, 5)

            zone_name = zone_combo.currentText()
            enabled = enabled_checkbox.isChecked()

            if zone_name == "Unassigned":
                unassigned_lasers.append(laser_id)
                continue

            zone_key = LASER_ZONE_KEYS[zone_name]

            settings = self.laser_settings_by_id.get(
                laser_id,
                DEFAULT_LASER_SETTINGS.copy()
            )

            settings["laser_id"] = laser_id
            settings["laser_name"] = laser_name
            settings["enabled"] = enabled

            profile["lasers"][zone_key] = settings
            saved_count += 1

        path = self.sku_profile_store.save_laser_profile(sku, profile)

        msg = f"Saved {saved_count} laser profile(s):\n{path}"

        if unassigned_lasers:
            msg += "\n\nNot saved because zone is Unassigned:\n"
            msg += "\n".join(unassigned_lasers)

        QMessageBox.information(self, "Laser Profile Saved", msg)


    def load_laser_profile(self):
        sku = self.laser_sku_input.text().strip()

        if not sku:
            QMessageBox.warning(self, "Missing SKU", "Please enter SKU name.")
            return

        profile = self.sku_profile_store.load_laser_profile(sku)
        lasers_config = profile.get("lasers", {})

        for zone_name, zone_key in LASER_ZONE_KEYS.items():
            laser_cfg = lasers_config.get(zone_key, {})
            laser_id = laser_cfg.get("laser_id", "")

            if laser_id:
                self.laser_settings_by_id[laser_id] = laser_cfg

        for row in range(self.laser_table.rowCount()):
            laser_item = self.laser_table.item(row, 0)

            if not laser_item:
                continue

            table_laser_id = laser_item.text()
            zone_combo = self.laser_table.cellWidget(row, 4)
            enabled_checkbox = self.laser_table.cellWidget(row, 5)

            zone_combo.setCurrentText("Unassigned")

            for zone_name, zone_key in LASER_ZONE_KEYS.items():
                laser_cfg = lasers_config.get(zone_key, {})

                if laser_cfg.get("laser_id", "") == table_laser_id:
                    zone_combo.setCurrentText(zone_name)
                    enabled_checkbox.setChecked(bool(laser_cfg.get("enabled", True)))
                    break

        self.laser_status_label.setText("Status: Laser profile loaded")
        QMessageBox.information(self, "Laser Profile Loaded", f"Loaded laser profile for SKU: {sku}")