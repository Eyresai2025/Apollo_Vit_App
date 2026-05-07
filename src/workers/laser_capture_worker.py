from PyQt5.QtCore import QThread, pyqtSignal


class LaserCaptureWorker(QThread):
    capture_done = pyqtSignal(dict)
    capture_failed = pyqtSignal(str)

    def __init__(self, laser_manager, laser_id, settings, parent=None):
        super().__init__(parent)

        self.laser_manager = laser_manager
        self.laser_id = laser_id
        self.settings = settings

    def run(self):
        try:
            result = self.laser_manager.capture_one_profile(
                self.laser_id,
                self.settings
            )

            self.capture_done.emit(result)

        except Exception as e:
            self.capture_failed.emit(str(e))