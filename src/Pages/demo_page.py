from PyQt5.QtCore import Qt, QEvent, pyqtSignal
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import (
    QWidget,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QHBoxLayout,
    QFrame,
    QSizePolicy,
    QDialog,
    QScrollArea,
    QMessageBox,
    QProgressBar,
)


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


class ClickableImageLabel(QLabel):
    clicked = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setCursor(Qt.PointingHandCursor)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(event)


class PreviewCard(QFrame):
    clicked = pyqtSignal()

    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self.image_path = None

        self.setObjectName("previewCard")
        self.setFixedWidth(360)
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Preferred)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 16)
        layout.setSpacing(10)

        self.title_label = QLabel(title)
        self.title_label.setObjectName("cardTitle")
        self.title_label.setAlignment(Qt.AlignCenter)

        self.image_label = ClickableImageLabel()
        self.image_label.setObjectName("previewImage")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setFixedSize(250, 360)
        self.image_label.setText("No Preview")
        self.image_label.clicked.connect(self.clicked.emit)

        layout.addWidget(self.title_label)
        layout.addWidget(self.image_label, 0, Qt.AlignCenter)

    def set_image(self, image_path: str):
        self.image_path = image_path

        pixmap = QPixmap(image_path)
        if pixmap.isNull():
            self.image_label.setPixmap(QPixmap())
            self.image_label.setText("No Preview")
            return

        scaled = pixmap.scaled(
            self.image_label.width() - 8,
            self.image_label.height() - 8,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.image_label.setPixmap(scaled)
        self.image_label.setText("")

    def clear_image(self):
        self.image_path = None
        self.image_label.setPixmap(QPixmap())
        self.image_label.setText("No Preview")


class DemoInspectionPage(QWidget):
    start_capture_clicked = pyqtSignal()
    run_inspect_clicked = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.sidewall_image_path = None
        self.innerside_image_path = None
        self.viewer_windows = []
        self.setup_ui()
        self.apply_styles()

    def setup_ui(self):
        self.setObjectName("demoInspectionPage")

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(22, 18, 22, 18)
        main_layout.setSpacing(14)

        header_layout = QHBoxLayout()
        header_layout.setContentsMargins(0, 0, 0, 0)

        title_box = QVBoxLayout()
        title_box.setSpacing(1)

        self.page_title = QLabel("Demo Inspection")
        self.page_title.setObjectName("pageTitle")

        self.page_subtitle = QLabel("Preview Sidewall and Inner Side camera images")
        self.page_subtitle.setObjectName("pageSubtitle")

        title_box.addWidget(self.page_title)
        title_box.addWidget(self.page_subtitle)

        header_layout.addLayout(title_box)
        header_layout.addStretch()

        self.version_label = QLabel("v1.0")
        self.version_label.setObjectName("versionLabel")
        header_layout.addWidget(self.version_label)

        main_layout.addLayout(header_layout)

        self.preview_outer = QFrame()
        self.preview_outer.setObjectName("previewOuter")

        preview_outer_layout = QVBoxLayout(self.preview_outer)
        preview_outer_layout.setContentsMargins(18, 18, 18, 18)
        preview_outer_layout.setSpacing(0)

        preview_row = QHBoxLayout()
        preview_row.setSpacing(18)
        preview_row.addStretch()

        self.sidewall_card = PreviewCard("Side Wall Preview")
        self.innerside_card = PreviewCard("Inner Side Preview")

        self.sidewall_card.clicked.connect(self.open_sidewall_viewer)
        self.innerside_card.clicked.connect(self.open_innerside_viewer)

        preview_row.addWidget(self.sidewall_card)
        preview_row.addWidget(self.innerside_card)
        preview_row.addStretch()

        preview_outer_layout.addLayout(preview_row)

        main_layout.addWidget(self.preview_outer)

        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(14)

        self.start_btn = QPushButton("Start")
        self.start_btn.setObjectName("primaryBtn")
        self.start_btn.setFixedSize(110, 42)

        self.inspect_btn = QPushButton("Run Inspect")
        self.inspect_btn.setObjectName("secondaryBtn")
        self.inspect_btn.setFixedSize(140, 42)

        self.start_btn.clicked.connect(self.on_start_clicked)
        self.inspect_btn.clicked.connect(self.on_run_inspect_clicked)

        btn_layout.addWidget(self.start_btn)
        btn_layout.addWidget(self.inspect_btn)
        btn_layout.addStretch()

        main_layout.addLayout(btn_layout)

        self.progress_text = QLabel("Ready")
        self.progress_text.setObjectName("progressText")
        self.progress_text.setVisible(False)

        self.progress_bar = QProgressBar()
        self.progress_bar.setObjectName("captureProgressBar")
        self.progress_bar.setFixedHeight(16)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setVisible(False)

        main_layout.addWidget(self.progress_text)
        main_layout.addWidget(self.progress_bar)

        main_layout.addStretch()
    
    def start_capture_progress(self, total_steps: int):
        self.progress_text.setText(f"Capturing images... 0/{total_steps}")
        self.progress_text.setVisible(True)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, total_steps)
        self.progress_bar.setValue(0)

    def update_capture_progress(self, current_step: int, total_steps: int, message: str = ""):
        self.progress_bar.setRange(0, total_steps)
        self.progress_bar.setValue(current_step)
        if message:
            self.progress_text.setText(message)
        else:
            self.progress_text.setText(f"Capturing images... {current_step}/{total_steps}")

    def finish_capture_progress(self, success: bool = True):
        if success:
            self.progress_text.setText("Capture completed")
        else:
            self.progress_text.setText("Capture failed")

    def reset_capture_progress(self):
        self.progress_text.setVisible(False)
        self.progress_bar.setVisible(False)
        self.progress_bar.setValue(0)

    def apply_styles(self):
        self.setStyleSheet("""
            QWidget#demoInspectionPage {
                background-color: #efedf2;
            }

            QLabel#pageTitle {
                font: 700 14px 'Segoe UI';
                color: #5f2db6;
            }

            QLabel#pageSubtitle {
                font: 11px 'Segoe UI';
                color: #6f628b;
            }

            QLabel#versionLabel {
                font: 11px 'Segoe UI';
                color: #9b93ad;
                padding-right: 4px;
            }

            QFrame#previewOuter {
                background-color: #f4f1f7;
                border: 1px solid #d9d0e6;
                border-radius: 20px;
            }

            QFrame#previewCard {
                background-color: #f6f3f9;
                border: 1px solid #d7cde4;
                border-radius: 18px;
            }

            QLabel#cardTitle {
                font: 700 12px 'Segoe UI';
                color: #5f2db6;
            }

            QLabel#previewImage {
                background-color: #ebe6f2;
                border: 1px solid #d4cae1;
                border-radius: 14px;
                font: 11px 'Segoe UI';
                color: #8a7ba4;
            }

            QPushButton#primaryBtn {
                background-color: #6c22c5;
                color: white;
                border: none;
                border-radius: 21px;
                font: 700 12px 'Segoe UI';
                padding: 0 18px;
            }

            QPushButton#primaryBtn:hover {
                background-color: #7b31d2;
            }

            QPushButton#secondaryBtn {
                background-color: white;
                color: #5f2db6;
                border: 1px solid #cfc3df;
                border-radius: 21px;
                font: 700 12px 'Segoe UI';
                padding: 0 18px;
            }

            QPushButton#secondaryBtn:hover {
                background-color: #f7f2fd;
            }
            
                QLabel#progressText {
                font: 700 11px 'Segoe UI';
                color: #5f2db6;
                padding-left: 2px;
            }

            QProgressBar#captureProgressBar {
                background-color: #ebe6f2;
                border: 1px solid #d4cae1;
                border-radius: 8px;
                text-align: center;
                color: #4a3b68;
                font: 700 10px 'Segoe UI';
            }

            QProgressBar#captureProgressBar::chunk {
                background-color: #6c22c5;
                border-radius: 7px;
            }
        """)

    def set_sidewall_image(self, image_path: str):
        self.sidewall_image_path = image_path
        self.sidewall_card.set_image(image_path)

    def set_innerside_image(self, image_path: str):
        self.innerside_image_path = image_path
        self.innerside_card.set_image(image_path)

    def clear_sidewall_image(self):
        self.sidewall_image_path = None
        self.sidewall_card.clear_image()

    def clear_innerside_image(self):
        self.innerside_image_path = None
        self.innerside_card.clear_image()

    def open_sidewall_viewer(self):
        if not self.sidewall_image_path:
            QMessageBox.information(self, "Preview", "No Side Wall image loaded.")
            return
        self._open_image_viewer(self.sidewall_image_path, "Side Wall Preview")

    def open_innerside_viewer(self):
        if not self.innerside_image_path:
            QMessageBox.information(self, "Preview", "No Inner Side image loaded.")
            return
        self._open_image_viewer(self.innerside_image_path, "Inner Side Preview")

    def _open_image_viewer(self, image_path: str, title: str):
        viewer = ImageViewerDialog(image_path=image_path, title=title, parent=self)
        viewer.show()
        self.viewer_windows.append(viewer)

    def on_start_clicked(self):
        self.start_capture_clicked.emit()

    def on_run_inspect_clicked(self):
        self.run_inspect_clicked.emit()


if __name__ == "__main__":
    import sys
    from PyQt5.QtWidgets import QApplication

    app = QApplication(sys.argv)

    demo = DemoInspectionPage()
    demo.resize(1100, 700)

    # Replace these with your actual image paths
    demo.set_sidewall_image(r"media/sidewall_sample.png")
    demo.set_innerside_image(r"media/innerside_sample.png")

    demo.show()
    sys.exit(app.exec_())