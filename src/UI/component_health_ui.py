# src/COMMON/component_health_ui.py


def set_health_label(main_window, key, item):
    if not hasattr(main_window, "health_labels"):
        return

    label = main_window.health_labels.get(key)

    if label is None:
        return

    ok = bool(item.get("ok", False))
    text = item.get("text", "-")

    if ok:
        label.setText(f"● {text}")
        label.setStyleSheet("font: bold 10px 'Segoe UI'; color:#2f9e44;")
    else:
        label.setText(f"● {text}")
        label.setStyleSheet("font: bold 10px 'Segoe UI'; color:#e03131;")


def update_mode_indicator(main_window, mode_info):
    if not hasattr(main_window, "mode_indicator_label"):
        return

    mode_text = mode_info.get("mode", "UNKNOWN")
    mode_ok = bool(mode_info.get("mode_ok", False))

    if mode_text == "AUTO":
        bg = "#2f9e44"
        fg = "white"
    elif mode_text == "MANUAL":
        bg = "#1971c2"
        fg = "white"
    elif mode_text == "TEACHING":
        bg = "#ff9800"
        fg = "white"
    elif mode_text == "FAULT":
        bg = "#e03131"
        fg = "white"
    elif mode_ok:
        bg = "#2f9e44"
        fg = "white"
    else:
        bg = "#eeeeee"
        fg = "#333333"

    main_window.mode_indicator_label.setText(f"Mode: {mode_text}")
    main_window.mode_indicator_label.setStyleSheet(f"""
        QLabel {{
            background: {bg};
            color: {fg};
            border-radius: 10px;
            padding: 6px 12px;
            font: 800 12px 'Segoe UI';
        }}
    """)


def update_live_system_status(main_window, system_text, system_ok):
    if not hasattr(main_window, "live_system_status_label"):
        return

    if system_text == "INSPECTION RUNNING":
        bg = "#1971c2"
        fg = "white"
    elif system_ok:
        bg = "#2f9e44"
        fg = "white"
    else:
        bg = "#e03131"
        fg = "white"

    main_window.live_system_status_label.setText(f"System: {system_text}")
    main_window.live_system_status_label.setStyleSheet(f"""
        QLabel {{
            background: {bg};
            color: {fg};
            border-radius: 10px;
            padding: 6px 12px;
            font: 800 12px 'Segoe UI';
        }}
    """)


def apply_component_health_to_gui(main_window, health):
    """
    Updates Live page component health labels.

    This file only updates UI.
    It does not check hardware.
    It does not connect cameras.
    It does not connect PLC.
    """

    items = health.get("items", {})

    set_health_label(main_window, "plc", items.get("plc", {}))
    set_health_label(main_window, "cameras", items.get("cameras", {}))
    set_health_label(main_window, "laser", items.get("laser", {}))
    set_health_label(main_window, "gpu", items.get("gpu", {}))
    set_health_label(main_window, "storage", items.get("storage", {}))
    set_health_label(main_window, "app_ok", items.get("app_ok", {}))

    update_mode_indicator(
        main_window,
        health.get("mode", {}),
    )

    update_live_system_status(
        main_window,
        health.get("system_text", "NOT READY"),
        bool(health.get("system_ok", False)),
    )