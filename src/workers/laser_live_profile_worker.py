import time
import cv2

from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtGui import QImage


class LaserLiveProfileWorker(QThread):
    frame_ready = pyqtSignal(QImage, dict)
    status_signal = pyqtSignal(str)
    error_signal = pyqtSignal(str)

    def __init__(self, laser_manager, laser_id, settings, parent=None):
        super().__init__(parent)

        self.laser_manager = laser_manager
        self.laser_id = laser_id
        self.settings = settings
        self.running = False

    def stop(self):
        self.running = False

    def run(self):
        self.running = True

        try:
            self.status_signal.emit("Starting laser live profile...")

            self.laser_manager.start_live_stream(
                self.laser_id,
                self.settings
            )

            while self.running:
                profile = self.laser_manager.get_live_profile(self.laser_id)
                metrics = self.laser_manager.compute_quality_metrics(profile)

                preview = self.laser_manager.profile_to_preview_image(
                    profile,
                    x_scale=float(self.settings.get("x_scale", 1.0)),
                    z_scale=float(self.settings.get("z_scale", 1.0)),
                )

                qimg = self.cv_to_qimage(preview)
                self.frame_ready.emit(qimg, metrics)

                self.status_signal.emit(
                    f"Live profile running | Decision: {metrics.get('decision', '-')}"
                )

                time.sleep(0.05)

            self.laser_manager.stop_live_stream(self.laser_id)
            self.status_signal.emit("Laser live profile stopped")

        except Exception as e:
            try:
                self.laser_manager.stop_live_stream(self.laser_id)
            except Exception:
                pass

            self.error_signal.emit(str(e))

    def cv_to_qimage(self, bgr_image):
        rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w

        qimg = QImage(
            rgb.data,
            w,
            h,
            bytes_per_line,
            QImage.Format_RGB888
        )

        return qimg.copy()