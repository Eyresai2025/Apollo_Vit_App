# src/camera/new_sku_software_capture.py
# =========================================================
# New SKU software-trigger capture
#
# Uses existing Test Mode connected MultiCameraManager if passed.
# Forces SOFTWARE trigger only during this capture.
#
# Save structure:
# media/new_sku_images/<SKU>/<serial>/train/good/  -> first train_good_count images
# media/new_sku_images/<SKU>/<serial>/             -> remaining images
# =========================================================
 
import os
import re
import time
from datetime import datetime
from typing import Dict, Optional, Any
 
import cv2
import numpy as np
 
from src.camera import HARDWARE_TRIGGER as HT
 
try:
    from src.COMMON.db import save_new_sku_image
except Exception:
    save_new_sku_image = None
 
 
def _safe_name(text: str) -> str:
    text = str(text or "").strip()
    if not text:
        return "unknown"
    text = re.sub(r'[<>:"/\\|?*]+', "_", text)
    text = re.sub(r"\s+", "_", text)
    return text.strip("._") or "unknown"
 
 
def _ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path
 
 
def _save_image_keep_depth(img: np.ndarray, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
 
    # Keep Mono16 as PNG if possible
    ok = cv2.imwrite(path, img)
    if not ok:
        raise RuntimeError(f"Failed to save image: {path}")
 
 
def _get_connected_manager(multi_camera_manager=None):
    """
    Prefer Test Mode connected camera manager.
    If not passed, create and connect one as fallback.
    """
    if multi_camera_manager is not None:
        return multi_camera_manager, False
 
    manager = HT.MultiCameraManager()
    manager.connect_all(fail_fast=False)
    return manager, True
 
 
def capture_new_sku_images(
    sku_name: str,
    media_path: str,
    images_per_camera: int = 20,
    train_good_count: int = 10,
    multi_camera_manager=None,
    sku_meta: Optional[Dict[str, Any]] = None,
    meta_collection: str = "New SKU",
    gridfs_bucket: str = "fs",
    capture_delay_sec: float = 0.25,
    logger=print,
) -> Dict[str, str]:
    """
    Returns:
        {
            "254901432": "latest_saved_path.png",
            "254901428": "latest_saved_path.png",
            ...
        }
    """
 
    sku_folder = _safe_name(sku_name)
    base_out_dir = _ensure_dir(os.path.join(media_path, "new_sku_images", sku_folder))
 
    images_per_camera = int(images_per_camera or 20)
    train_good_count = int(train_good_count or 10)
 
    if train_good_count >= images_per_camera:
        train_good_count = max(1, images_per_camera // 2)
 
    logger("=" * 70)
    logger("[NEW SKU CAPTURE] Software trigger capture started")
    logger(f"[NEW SKU CAPTURE] SKU              : {sku_folder}")
    logger(f"[NEW SKU CAPTURE] Images/camera    : {images_per_camera}")
    logger(f"[NEW SKU CAPTURE] Train good count : {train_good_count}")
    logger(f"[NEW SKU CAPTURE] Save root        : {base_out_dir}")
    logger("=" * 70)
 
    manager, created_here = _get_connected_manager(multi_camera_manager)
 
    old_trigger_mode = HT.TRIGGER_MODE
    latest_paths: Dict[str, str] = {}
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
 
    try:
        # Force only this New SKU capture to software trigger.
        # Live inspection .env remains plc_software.
        HT.TRIGGER_MODE = "software"
 
        logger("[NEW SKU CAPTURE] Configuring connected cameras for SOFTWARE trigger...")
        manager.start_all_streams()
 
        for shot_idx in range(1, images_per_camera + 1):
            logger("")
            logger(f"[NEW SKU CAPTURE] Capturing set {shot_idx}/{images_per_camera}")
 
            captured = manager.capture_all()
 
            if not captured or not any(img is not None for img in captured.values()):
                raise RuntimeError(f"No images captured in set {shot_idx}")
 
            capture_stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
 
            for serial, img in captured.items():
                serial_str = _safe_name(str(serial))
 
                if img is None:
                    logger(f"[NEW SKU CAPTURE][WARN] No image from serial {serial_str}")
                    continue
 
                serial_root = _ensure_dir(os.path.join(base_out_dir, serial_str))
 
                if shot_idx <= train_good_count:
                    save_dir = _ensure_dir(os.path.join(serial_root, "train", "good"))
                    save_group = "train_good"
                else:
                    save_dir = serial_root
                    save_group = "serial_root"
 
                file_name = f"{serial_str}_{capture_stamp}_{shot_idx:03d}.png"
                file_path = os.path.join(save_dir, file_name)
 
                _save_image_keep_depth(img, file_path)
 
                latest_paths[serial_str] = file_path
 
                logger(f"[SAVE OK] {serial_str} -> {file_path}")
 
                if save_new_sku_image is not None:
                    try:
                        db_meta = dict(sku_meta or {})
                        db_meta.pop("machine_serial", None)
                        db_meta.update({
                            "sku_name": sku_folder,
                            "camera_serial": serial_str,
                            "session_id": session_id,
                            "capture_index": shot_idx,
                            "total_images_per_camera": images_per_camera,
                            "train_good_count": train_good_count,
                            "save_group": save_group,
                            "saved_dir": save_dir,
                            "saved_file": file_name,
                            "saved_path": file_path,
                            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        })
 
                        save_new_sku_image(
                            file_path=file_path,
                            label=serial_str,
                            capture_id=session_id,
                            sku_meta=db_meta,
                            meta_collection=meta_collection,
                            gridfs_bucket=gridfs_bucket,
                        )
                    except Exception as e:
                        logger(f"[DB WARN] Could not save metadata for {file_path}: {e}")
 
            if capture_delay_sec > 0:
                time.sleep(capture_delay_sec)
 
        logger("")
        logger("[NEW SKU CAPTURE] Completed successfully")
        return latest_paths
 
    finally:
        HT.TRIGGER_MODE = old_trigger_mode
 
        try:
            manager.stop_all_streams()
        except Exception as e:
            logger(f"[NEW SKU CAPTURE][WARN] stop_all_streams failed: {e}")
 
        if created_here:
            try:
                manager.close_all()
            except Exception:
                pass
 