from PyQt5.QtCore import QThread, pyqtSignal


class CameraCaptureWorker(QThread):
    capture_done = pyqtSignal(str, int)
    capture_failed = pyqtSignal(str)

    def __init__(self, camera_manager, serial, settings, mode, parent=None):
        super().__init__(parent)

        self.camera_manager = camera_manager
        self.serial = serial
        self.settings = settings
        self.mode = mode

    def run(self):
        try:
            image_path, line_count = self.camera_manager.capture_one_image(
                self.serial,
                self.settings,
                mode=self.mode
            )

            self.capture_done.emit(image_path, line_count)

        except Exception as e:
            self.capture_failed.emit(str(e))