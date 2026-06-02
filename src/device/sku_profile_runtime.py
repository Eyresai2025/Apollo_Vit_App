# src/device/sku_profile_runtime.py

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


def _safe_sku(sku_name: str) -> str:
    sku = str(sku_name or "").strip()
    return sku or "SKU_001"


def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Profile file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_sku_camera_profile(media_root: str, sku_name: str) -> Dict[str, Any]:
    sku = _safe_sku(sku_name)

    path = (
        Path(media_root)
        / "Camera_Profiles"
        / sku
        / "camera_profile.json"
    )

    profile = load_json(path)

    if profile.get("profile_type") != "camera":
        raise ValueError(f"Invalid camera profile file: {path}")

    cameras = profile.get("cameras", {})
    if not isinstance(cameras, dict) or not cameras:
        raise ValueError(f"No camera settings found in profile: {path}")

    return profile


def load_sku_laser_profile(media_root: str, sku_name: str) -> Dict[str, Any]:
    sku = _safe_sku(sku_name)

    path = (
        Path(media_root)
        / "Laser_Profiles"
        / sku
        / "laser_profile.json"
    )

    profile = load_json(path)

    if profile.get("profile_type") != "laser":
        raise ValueError(f"Invalid laser profile file: {path}")

    lasers = profile.get("lasers", {})
    if not isinstance(lasers, dict) or not lasers:
        raise ValueError(f"No laser settings found in profile: {path}")

    return profile


def apply_sku_laser_profile_to_manager(media_root: str, sku_name: str, laser_manager) -> Dict[str, Any]:
    """
    Optional helper for later Live laser integration.
    Current camera Live flow does not yet pass laser_manager into ContinuousCycleWorker.
    """

    profile = load_sku_laser_profile(media_root, sku_name)
    lasers = profile.get("lasers", {}) or {}

    if laser_manager is None:
        raise RuntimeError("laser_manager is None. Cannot apply laser profile.")

    if not hasattr(laser_manager, "apply_settings"):
        raise RuntimeError("laser_manager does not have apply_settings().")

    for zone, settings in lasers.items():
        if not bool(settings.get("enabled", True)):
            print(f"[LASER PROFILE] Skipped disabled zone: {zone}")
            continue

        laser_id = str(settings.get("laser_id", "")).strip()

        if not laser_id:
            print(f"[LASER PROFILE][WARN] Missing laser_id for zone={zone}")
            continue

        ok, msg = laser_manager.apply_settings(laser_id, settings)

        if not ok:
            raise RuntimeError(
                f"Laser profile apply failed | zone={zone} | laser_id={laser_id} | {msg}"
            )

        print(f"[LASER PROFILE] Applied zone={zone} laser_id={laser_id}: {msg}")

    return profile