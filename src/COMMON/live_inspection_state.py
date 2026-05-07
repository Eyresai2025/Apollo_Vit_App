# src/COMMON/live_inspection_state.py

from threading import Lock
from datetime import datetime


_LOCK = Lock()

_STATE = {
    "phase": "WAITING",
    "active_zone": "-",
    "images_captured": 0,
    "total_images": 5,
    "message": "Waiting for trigger",
    "updated_at": "",
}


VALID_PHASES = {
    "WAITING",
    "CAPTURING",
    "INFERENCE",
    "COMPLETED",
    "FAILED",
}


def reset_live_progress(total_images=5):
    with _LOCK:
        _STATE.update({
            "phase": "WAITING",
            "active_zone": "-",
            "images_captured": 0,
            "total_images": int(total_images or 5),
            "message": "Waiting for trigger",
            "updated_at": datetime.now().strftime("%H:%M:%S"),
        })


def set_live_progress(
    phase=None,
    active_zone=None,
    images_captured=None,
    total_images=None,
    message=None,
):
    with _LOCK:
        if phase:
            phase = str(phase).upper().strip()
            _STATE["phase"] = phase if phase in VALID_PHASES else phase

        if active_zone is not None:
            _STATE["active_zone"] = str(active_zone)

        if images_captured is not None:
            try:
                _STATE["images_captured"] = int(images_captured)
            except Exception:
                pass

        if total_images is not None:
            try:
                _STATE["total_images"] = int(total_images)
            except Exception:
                pass

        if message is not None:
            _STATE["message"] = str(message)

        _STATE["updated_at"] = datetime.now().strftime("%H:%M:%S")


def get_live_progress():
    with _LOCK:
        return dict(_STATE)