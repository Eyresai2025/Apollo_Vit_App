# src/COMMON/live_result_state.py

from threading import Lock
from datetime import datetime


_LOCK = Lock()

_STATE = {
    "final_result": "WAITING",
    "cycle_id": "-",
    "worst_zone": "-",
    "defect_zones": "0 / 5",
    "cycle_time": "-",
    "plc_output": "Not Sent",
    "message": "Waiting for inspection",
    "updated_at": "",
}


SEVERITY = {
    "OK": 0,
    "PASS": 0,
    "GOOD": 0,

    "SUSPECT": 1,

    "DEFECT": 2,
    "NG": 2,
    "FAIL": 2,
    "FAILED": 3,
    "INVALID": 3,
}


BAD_LABELS = {"DEFECT", "NG", "FAIL", "FAILED", "INVALID", "SUSPECT"}


def reset_live_result():
    with _LOCK:
        _STATE.update({
            "final_result": "WAITING",
            "cycle_id": "-",
            "worst_zone": "-",
            "defect_zones": "0 / 5",
            "cycle_time": "-",
            "plc_output": "Not Sent",
            "message": "Waiting for inspection",
            "updated_at": datetime.now().strftime("%H:%M:%S"),
        })


def _normalize_label(label):
    label = str(label or "").strip().upper()

    if label in ("OK", "PASS", "GOOD"):
        return "OK"

    if label in ("DEFECT", "NG", "FAIL"):
        return "NG"

    if label == "SUSPECT":
        return "SUSPECT"

    if label in ("INVALID", "FAILED"):
        return "INVALID"

    if not label:
        return "UNKNOWN"

    return label


def _side_display_name(side_name):
    mapping = {
        "sidewall1": "Side Wall 1",
        "sidewall2": "Side Wall 2",
        "innerwall": "Inner Side",
        "tread": "Tread",
        "bead": "Bead",
    }
    return mapping.get(str(side_name), str(side_name))


def update_live_result_from_cycle_result(result, total_zones=5):
    """
    Reads cycle result from cycle_engine.run_cycle() / GUI worker result
    and updates F-025 summary.

    Expected result keys:
        cycle_id
        final_label
        cycle_latency_sec
        side_results
    """
    if not isinstance(result, dict):
        result = {}

    side_results = result.get("side_results", {}) or {}
    final_label_raw = result.get("final_label", "UNKNOWN")
    final_result = _normalize_label(final_label_raw)

    total = len(side_results) if side_results else int(total_zones or 5)

    bad_zones = []
    worst_zone = "-"
    worst_score = -1

    for side_name, side_result in side_results.items():
        side_label_raw = ""
        if isinstance(side_result, dict):
            side_label_raw = side_result.get("final_label", "")
        side_label = str(side_label_raw or "").strip().upper()

        score = SEVERITY.get(side_label, 0)

        if side_label in BAD_LABELS:
            bad_zones.append(side_name)

        if score > worst_score:
            worst_score = score
            worst_zone = _side_display_name(side_name)

    if not bad_zones:
        worst_zone = "-"

    cycle_time = result.get("cycle_latency_sec", "-")
    try:
        cycle_time = f"{float(cycle_time):.2f} sec"
    except Exception:
        cycle_time = "-"

    summary = {
        "final_result": final_result,
        "cycle_id": result.get("cycle_id", "-") or "-",
        "worst_zone": worst_zone,
        "defect_zones": f"{len(bad_zones)} / {total}",
        "cycle_time": cycle_time,
        "plc_output": _STATE.get("plc_output", "Not Sent"),
        "message": f"Inspection completed: {final_result}",
        "updated_at": datetime.now().strftime("%H:%M:%S"),
    }

    with _LOCK:
        _STATE.update(summary)

    return dict(summary)


def set_live_result_plc_output(plc_output):
    with _LOCK:
        _STATE["plc_output"] = str(plc_output or "Not Sent")
        _STATE["updated_at"] = datetime.now().strftime("%H:%M:%S")


def set_live_result_failed(message="Inspection failed"):
    with _LOCK:
        _STATE.update({
            "final_result": "FAILED",
            "cycle_id": "-",
            "worst_zone": "-",
            "defect_zones": "0 / 5",
            "cycle_time": "-",
            "plc_output": "Not Sent",
            "message": str(message),
            "updated_at": datetime.now().strftime("%H:%M:%S"),
        })


def get_live_result():
    with _LOCK:
        return dict(_STATE)