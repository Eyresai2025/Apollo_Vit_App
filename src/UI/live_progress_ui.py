# src/UI/live_progress_ui.py

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QFrame, QLabel, QVBoxLayout, QHBoxLayout, QSizePolicy, QProgressBar
)

from src.COMMON.live_inspection_state import get_live_progress


def _phase_color(phase):
    phase = str(phase or "").upper()

    if phase == "WAITING":
        return "#666666"
    if phase == "CAPTURING":
        return "#1971c2"
    if phase == "INFERENCE":
        return "#ff9800"
    if phase == "COMPLETED":
        return "#2f9e44"
    if phase == "FAILED":
        return "#e03131"

    return "#571c86"


def _make_progress_card(main_window, title, value="--"):
    card = QFrame()
    card.setMinimumHeight(main_window.s(58))
    card.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
    card.setStyleSheet("""
        QFrame {
            background-color: white;
            border-radius: 10px;
        }
    """)

    lay = QVBoxLayout(card)
    lay.setContentsMargins(10, 6, 10, 6)
    lay.setSpacing(2)

    title_lbl = QLabel(title)
    title_lbl.setStyleSheet("font: bold 11px 'Segoe UI'; color:#333;")
    title_lbl.setAlignment(Qt.AlignLeft)
    lay.addWidget(title_lbl)

    value_lbl = QLabel(value)
    value_lbl.setAlignment(Qt.AlignCenter)
    value_lbl.setStyleSheet("""
        QLabel {
            font: 900 13px 'Segoe UI';
            color: white;
            background: #666666;
            border-radius: 7px;
            padding: 5px 8px;
        }
    """)
    lay.addWidget(value_lbl)

    return card, value_lbl


def create_live_progress_widget(main_window):
    """
    Creates the progress widget.
    Call this from GUI.py inside setup_main_content().
    """

    outer = QFrame()
    outer.setStyleSheet("background-color: #f5f5f5;")

    root = QVBoxLayout(outer)
    root.setContentsMargins(0, 6, 0, 0)
    root.setSpacing(6)

    row = QHBoxLayout()
    row.setSpacing(main_window.s(10))
    row.setContentsMargins(0, 0, 0, 0)

    phase_card, main_window.live_phase_value_lbl = _make_progress_card(
        main_window,
        "Inspection Phase",
        "WAITING"
    )

    zone_card, main_window.live_zone_value_lbl = _make_progress_card(
        main_window,
        "Active Zone",
        "-"
    )

    count_card, main_window.live_count_value_lbl = _make_progress_card(
        main_window,
        "Images Captured",
        "0 / 5"
    )

    row.addWidget(phase_card)
    row.addWidget(zone_card)
    row.addWidget(count_card)

    root.addLayout(row)

    main_window.live_progress_bar = QProgressBar()
    main_window.live_progress_bar.setRange(0, 100)
    main_window.live_progress_bar.setValue(0)
    main_window.live_progress_bar.setTextVisible(True)
    main_window.live_progress_bar.setFixedHeight(16)
    main_window.live_progress_bar.setStyleSheet("""
        QProgressBar {
            background: white;
            border: 1px solid #dddddd;
            border-radius: 8px;
            text-align: center;
            font: bold 10px 'Segoe UI';
            color: #333333;
        }
        QProgressBar::chunk {
            background-color: #571c86;
            border-radius: 8px;
        }
    """)
    root.addWidget(main_window.live_progress_bar)

    return outer


def _set_value_label(label, text, color):
    if label is None:
        return

    label.setText(str(text))
    label.setStyleSheet(f"""
        QLabel {{
            font: 900 13px 'Segoe UI';
            color: white;
            background: {color};
            border-radius: 7px;
            padding: 5px 8px;
        }}
    """)


def apply_live_progress_to_gui(main_window):
    """
    Reads memory-only progress state and updates GUI labels.
    No hardware access. No PLC read. No camera access.
    """

    state = get_live_progress()

    phase = state.get("phase", "WAITING")
    active_zone = state.get("active_zone", "-")
    captured = int(state.get("images_captured", 0) or 0)
    total = int(state.get("total_images", 5) or 5)

    color = _phase_color(phase)

    _set_value_label(
        getattr(main_window, "live_phase_value_lbl", None),
        phase,
        color,
    )

    _set_value_label(
        getattr(main_window, "live_zone_value_lbl", None),
        active_zone,
        color,
    )

    _set_value_label(
        getattr(main_window, "live_count_value_lbl", None),
        f"{captured} / {total}",
        color,
    )

    progress = 0
    if total > 0:
        progress = int((captured / total) * 100)

    bar = getattr(main_window, "live_progress_bar", None)
    if bar is not None:
        bar.setValue(max(0, min(100, progress)))
        bar.setFormat(f"{captured} / {total} images")