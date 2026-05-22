import time
import cv2
import numpy as np

from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtGui import QImage


class CameraLivePreviewWorker(QThread):
    frame_ready = pyqtSignal(QImage, int)
    status_signal = pyqtSignal(str)
    error_signal = pyqtSignal(str)

    def __init__(self, camera_manager, serial, settings, mode, parent=None):
        super().__init__(parent)

        self.camera_manager = camera_manager
        self.serial = serial
        self.settings = settings
        self.mode = mode
        self.running = False

    def stop(self):
        self.running = False

    def run(self):
        self.running = True

        try:
            self.status_signal.emit("Starting live preview...")

            self.camera_manager.start_live_stream(
                self.serial,
                self.settings,
                mode=self.mode
            )

            expected_height = int(self.settings.get("height", 6000))

            if self.mode == "preview_free_run":
                self.status_signal.emit("Software/free-run preview running")
            else:
                self.status_signal.emit("Waiting for Line0 hardware trigger...")

            while self.running:
                try:
                    frame = self.camera_manager.get_live_frame(
                        self.serial,
                        timeout=1000
                    )

                    qimg = self.numpy_to_qimage(frame)
                    line_count = frame.shape[0]

                    self.frame_ready.emit(qimg, line_count)

                    if self.mode == "preview_free_run":
                        self.status_signal.emit(
                            f"Free-run preview | Lines: {line_count}/{expected_height}"
                        )
                    else:
                        self.status_signal.emit(
                            f"Hardware trigger preview | Lines: {line_count}/{expected_height}"
                        )

                    time.sleep(0.03)

                except Exception:
                    if self.mode == "preview_free_run":
                        self.status_signal.emit("Waiting for camera frame...")
                    else:
                        self.status_signal.emit("Waiting for Line0 trigger / frame...")

                    time.sleep(0.05)

            self.camera_manager.stop_live_stream(self.serial)
            self.status_signal.emit("Live preview stopped")

        except Exception as e:
            try:
                self.camera_manager.stop_live_stream(self.serial)
            except Exception:
                pass

            self.error_signal.emit(str(e))

    def numpy_to_qimage(self, frame):
        # 1. Handle 16-bit to 8-bit conversion with smart contrast
        if frame.dtype == np.uint16:
            # Use 1st percentile as black and 99th percentile as white
            # This ignores dead pixels or extreme reflections that ruin contrast
            p_low = np.percentile(frame, 1)
            p_high = np.percentile(frame, 99)
            
            # Avoid division by zero if the image is flat
            if p_high <= p_low:
                p_low = frame.min()
                p_high = frame.max()
            
            # Normalize to 0-255 range
            display = np.clip((frame - p_low) * (255.0 / (p_high - p_low)), 0, 255).astype(np.uint8)
        else:
            # 2. Handle 8-bit images directly
            display = frame.astype(np.uint8)
        
        # 3. Get dimensions for resizing
        h, w = display.shape
        
        # 4. Resize for performance if the image is too large
        max_w = 1200
        if w > max_w:
            scale = max_w / float(w)
            new_w = int(w * scale)
            new_h = int(h * scale)
            display = cv2.resize(display, (new_w, new_h), interpolation=cv2.INTER_AREA)
            h, w = display.shape
        
        # 5. Convert to QImage format
        qimg = QImage(
            display.data,
            w,
            h,
            w,
            QImage.Format_Grayscale8
        )
        
        return qimg.copy()