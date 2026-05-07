# src/UI/live_result_ui.py

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QFrame, QLabel, QVBoxLayout, QHBoxLayout

from src.COMMON.live_result_state import get_live_result


def _result_color(result):
    result = str(result or "").upper()

    if result in ("OK", "PASS", "GOOD"):
        return "#2f9e44"

    if result in ("NG", "DEFECT", "FAIL", "FAILED", "INVALID"):
        return "#e03131"

    if result == "SUSPECT":
        return "#ff9800"

    if result == "WAITING":
        return "#666666"

    return "#571c86"


def _make_row(main_window, title, default="-"):
    row = QFrame()
    row.setStyleSheet("""
        QFrame {
            background-color: #F8F8F8;
            border-radius: 8px;
        }
    """)

    layout = QHBoxLayout(row)
    layout.setContentsMargins(8, 5, 8, 5)
    layout.setSpacing(6)

    title_lbl = QLabel(title)
    title_lbl.setStyleSheet("font: bold 10px 'Segoe UI'; color:#222;")
    layout.addWidget(title_lbl)

    layout.addStretch()

    value_lbl = QLabel(default)
    value_lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
    value_lbl.setStyleSheet("font: bold 10px 'Segoe UI'; color:#333;")
    layout.addWidget(value_lbl)

    return row, value_lbl


def create_tyre_result_summary_widget(main_window):
    """
    Creates F-025 Tyre Result Summary card.
    Add this between Component Health and Defect Info.
    """

    outer = QFrame()
    outer.setStyleSheet("""
        QFrame {
            background-color: white;
            border-radius: 10px;
        }
    """)

    layout = QVBoxLayout(outer)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(6)

    title = QLabel("Tyre Result Summary")
    title.setAlignment(Qt.AlignCenter)
    title.setStyleSheet("font: bold 15px 'Arial'; color:#111;")
    layout.addWidget(title)

    main_window.tyre_result_final_lbl = QLabel("WAITING")
    main_window.tyre_result_final_lbl.setAlignment(Qt.AlignCenter)
    main_window.tyre_result_final_lbl.setStyleSheet("""
        QLabel {
            background: #666666;
            color: white;
            border-radius: 10px;
            padding: 8px;
            font: 900 16px 'Segoe UI';
        }
    """)
    layout.addWidget(main_window.tyre_result_final_lbl)

    row, main_window.tyre_result_cycle_lbl = _make_row(main_window, "Cycle ID", "-")
    layout.addWidget(row)

    row, main_window.tyre_result_worst_zone_lbl = _make_row(main_window, "Worst Zone", "-")
    layout.addWidget(row)

    row, main_window.tyre_result_defect_zones_lbl = _make_row(main_window, "Defect Zones", "0 / 5")
    layout.addWidget(row)

    row, main_window.tyre_result_cycle_time_lbl = _make_row(main_window, "Cycle Time", "-")
    layout.addWidget(row)

    row, main_window.tyre_result_plc_lbl = _make_row(main_window, "PLC Output", "Not Sent")
    layout.addWidget(row)

    return outer


def _set_label(label, text):
    if label is not None:
        label.setText(str(text))


def apply_tyre_result_to_gui(main_window):
    """
    Updates Tyre Result Summary from memory-only state.
    """

    state = get_live_result()

    final_result = state.get("final_result", "WAITING")
    color = _result_color(final_result)

    final_lbl = getattr(main_window, "tyre_result_final_lbl", None)
    if final_lbl is not None:
        final_lbl.setText(str(final_result))
        final_lbl.setStyleSheet(f"""
            QLabel {{
                background: {color};
                color: white;
                border-radius: 10px;
                padding: 8px;
                font: 900 16px 'Segoe UI';
            }}
        """)

    _set_label(
        getattr(main_window, "tyre_result_cycle_lbl", None),
        state.get("cycle_id", "-")
    )

    _set_label(
        getattr(main_window, "tyre_result_worst_zone_lbl", None),
        state.get("worst_zone", "-")
    )

    _set_label(
        getattr(main_window, "tyre_result_defect_zones_lbl", None),
        state.get("defect_zones", "0 / 5")
    )

    _set_label(
        getattr(main_window, "tyre_result_cycle_time_lbl", None),
        state.get("cycle_time", "-")
    )

    _set_label(
        getattr(main_window, "tyre_result_plc_lbl", None),
        state.get("plc_output", "Not Sent")
    )