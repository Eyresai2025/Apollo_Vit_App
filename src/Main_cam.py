import os
import threading
import time
import logging
logger = logging.getLogger(__name__)
from datetime import datetime
from typing import Any, Dict, List, Optional, Callable
import traceback
import numpy as np
from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot
 
from src.COMMON.db import save_cycle_metadata
from src.COMMON.cycle_engine import (
    DEVICE,
    CAMERA_CAPTURE_ENABLED,
    R_ALIGN_GPU_CONCURRENCY,
    VIT_GPU_CONCURRENCY,
    YOLO_GPU_CONCURRENCY,
    _normalize_device,
    _resolve_sides,
    _required_file,
    _get_sku_calibration_dir,
    _get_sku_artifacts_dir,
    _get_today_capture_root,
    build_cycle_capture_dir,
    capture_and_save_images,
    build_image_map_from_capture_dir,
    build_all_runtimes,
    _apply_tyre_name_to_runtimes,
    _maybe_warmup_runtimes,
    run_cycle,
    preload_live_runtimes,
    clear_runtime_cache,
)
 
from src.camera.HARDWARE_TRIGGER import (
    TRIGGER_MODE,
    get_camera_to_side_map,
    get_side_to_camera_map,
)
from src.device.sku_profile_runtime import load_sku_camera_profile
try:
    from src.COMMON.live_inspection_state import set_live_progress
except Exception:
    def set_live_progress(*args, **kwargs):
        pass
 
# =========================================================
# CONTINUOUS CYCLE WORKER (Runs in background thread)
# =========================================================
 
class ContinuousCycleWorker(QObject):
    """
    Worker that runs in a QThread.
    Monitors PLC tag (software mode) or waits for hardware trigger.
    Orchestrates camera capture + AI pipeline.
    """
   
    # Signals for GUI communication
    capture_started = pyqtSignal(str)
    capture_completed = pyqtSignal(dict)
    images_saved = pyqtSignal(dict)
    processing_started = pyqtSignal(str)
    processing_progress = pyqtSignal(str, str)
    processing_completed = pyqtSignal(dict)
    processing_error = pyqtSignal(str)
    status_update = pyqtSignal(str)
    finished = pyqtSignal()
    error = pyqtSignal(str)
    ready_for_inspection = pyqtSignal(str)
   
    def __init__(
        self,
        media_root: str,
        sku_name: str,
        tyre_name: str,
        device: str,
        seg_model_a_path: str,
        seg_model_b_path: str,
        vit_checkpoint_path: str,
        r_detector_path: str,
        multi_camera_manager,
        min_capture_interval: float = 2.0,
        sides_to_run: Optional[List[str]] = None,
        capture_sides: Optional[List[str]] = None,
        side_configs: Optional[Dict[str, Dict[str, Any]]] = None,
        auto_preload: bool = True,
    ):
        super().__init__()
        self.media_root = os.path.abspath(media_root)
        self.sku_name = sku_name
        self.tyre_name = tyre_name
        self.device = _normalize_device(device)
        self.seg_model_a_path = seg_model_a_path
        self.seg_model_b_path = seg_model_b_path
        self.vit_checkpoint_path = vit_checkpoint_path
        self.r_detector_path = r_detector_path
        self.multi_camera_manager = multi_camera_manager
        self.min_capture_interval = min_capture_interval
        self.sides_to_run = _resolve_sides(sides_to_run)
        self.capture_sides = capture_sides or [
            "sidewall1",
            "sidewall2",
            "innerwall",
            "tread",
            "bead",
        ]
        self.side_configs = side_configs
        self.auto_preload = auto_preload
        self.require_ready_confirmation = True
        self._ready_confirm_event = threading.Event()
        self._stop_event = threading.Event()
        self._is_running = False
        self._cleanup_lock = threading.Lock()
        self._cleanup_done = False
        self._runtimes_preloaded = False
        self._runtimes = None
        self.is_hardware = (TRIGGER_MODE == "hardware")
       
        self.camera_to_side = get_camera_to_side_map()
        self.side_to_camera = get_side_to_camera_map()
        os.makedirs(self.media_root, exist_ok=True)
   
    @pyqtSlot()
    def run(self):
        """Main loop - monitors trigger and orchestrates everything"""
        self._is_running = True
        capture_count = 0
        last_capture_time = 0
       
        self.status_update.emit("=" * 50)
        self.status_update.emit(" Starting Continuous Inspection System")
        self.status_update.emit(f"   Trigger Mode: {TRIGGER_MODE.upper()}")
        self.status_update.emit(f"   SKU: {self.sku_name}")
        self.status_update.emit(f"   Tyre: {self.tyre_name}")
        self.status_update.emit(f"   Device: {self.device}")
        self.status_update.emit(f"   Min Interval: {self.min_capture_interval}s")
        self.status_update.emit(f"   Sides: {', '.join(self.sides_to_run)}")
        self.status_update.emit("=" * 50)
       
        # Preload AI runtimes
        if self.auto_preload and not self._runtimes_preloaded:
            self._preload_runtimes()
       
        # Configure + start camera streams in Live
        if not hasattr(self.multi_camera_manager, "start_all_streams"):
            raise RuntimeError(
                "multi_camera_manager does not have start_all_streams(). "
                "Update src/camera/HARDWARE_TRIGGER.py."
            )
 
        self.status_update.emit(f" Loading camera profile for SKU: {self.sku_name}")

        camera_profile = load_sku_camera_profile(
            media_root=self.media_root,
            sku_name=self.sku_name,
        )

        if not hasattr(self.multi_camera_manager, "apply_camera_profile"):
            raise RuntimeError(
                "multi_camera_manager does not support apply_camera_profile(). "
                "Update src/camera/HARDWARE_TRIGGER.py."
            )

        self.multi_camera_manager.apply_camera_profile(camera_profile)

        self.status_update.emit(" Configuring cameras for Live...")
        self.multi_camera_manager.start_all_streams()
 
        if self.is_hardware:
            self.status_update.emit(" Camera streams started - waiting for HARDWARE triggers")
        else:
            self.status_update.emit(" Camera streams started - waiting for PLC triggers")
       
        ready_msg = (
            f"All AI files loaded and camera configuration completed.\n\n"
            f"SKU: {self.sku_name}\n"
            f"Tyre: {self.tyre_name}\n\n"
            f"Click OK to start waiting for trigger."
        )
 
        self.ready_for_inspection.emit(ready_msg)
 
        while not self._stop_event.is_set() and not self._ready_confirm_event.wait(0.1):
            pass
 
        if self._stop_event.is_set():
            self._cleanup()
            self._is_running = False
            self.finished.emit()
            return
       
        if self.is_hardware:
            self.status_update.emit(" Waiting for HARDWARE trigger signal...")
        else:
            self.status_update.emit(" Waiting for PLC software trigger signal...")
 
        # MAIN LOOP - capture_all() blocks internally based on CAM_TRIGGER_MODE
        while not self._stop_event.is_set():
            try:
                should_capture = False
 
                # capture_all() blocks internally until PLC/software trigger capture is completed.
                current_time = time.time()
                if current_time - last_capture_time >= self.min_capture_interval:
                    should_capture = True
 
                if should_capture:
                    capture_count += 1
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
 
                    self.status_update.emit("")
                    self.status_update.emit(f" ═══ INSPECTION TRIGGER #{capture_count} ═══")
                    self.status_update.emit(f"   Time: {timestamp}")
                    self.capture_started.emit(timestamp)
 
                    capture_success = self._execute_capture(capture_count, timestamp)
 
                    if capture_success:
                        last_capture_time = time.time()
                    else:
                        self.status_update.emit("  Capture skipped or failed")
 
            except Exception as e:
                error_msg = f"Continuous cycle error: {e}"
                self.status_update.emit(f" {error_msg}")
                self.processing_error.emit(error_msg)
                traceback.print_exc()
                time.sleep(1)
       
        self._cleanup()
        self._is_running = False
        self.status_update.emit(" Continuous cycle stopped")
        self.finished.emit()
   
    def confirm_ready_to_start(self):
        self._ready_confirm_event.set()
 
   
    def _preload_runtimes(self):
        """Preload AI runtimes"""
        try:
            self.status_update.emit(" Preloading AI models...")
           
            self._runtimes = build_all_runtimes(
                sku_name=self.sku_name,
                media_root=self.media_root,
                seg_model_a_path=self.seg_model_a_path,
                seg_model_b_path=self.seg_model_b_path,
                vit_checkpoint_path=self.vit_checkpoint_path,
                r_detector_path=self.r_detector_path,
                device=self.device,
                capture_root=self.media_root,
                tyre_name=self.tyre_name,
                side_configs=self.side_configs,
                sides_to_run=self.sides_to_run,
            )
           
            _apply_tyre_name_to_runtimes(self._runtimes, self.tyre_name)
           
            _maybe_warmup_runtimes(
                runtimes=self._runtimes,
                sku_name=self.sku_name,
                device=self.device,
                capture_root=self.media_root,
                seg_model_a_path=self.seg_model_a_path,
                seg_model_b_path=self.seg_model_b_path,
                vit_checkpoint_path=self.vit_checkpoint_path,
                r_detector_path=self.r_detector_path,
                tyre_name=self.tyre_name,
                media_root=self.media_root,
                sides_to_run=self.sides_to_run,
            )
           
            self._runtimes_preloaded = True
            self.status_update.emit(" AI models preloaded successfully")
           
        except Exception as e:
            self._runtimes_preloaded = False
            self.status_update.emit(f"  Runtime preload failed: {e}")
   
    def _get_or_load_runtimes(self):
        """Get cached runtimes or load them"""
        if self._runtimes_preloaded and self._runtimes is not None:
            return self._runtimes
        self._preload_runtimes()
        return self._runtimes
    
    def _timing_log(self, msg: str):
        """
        Timing log goes to:
        1. GUI status_update signal
        2. app.log / console through logger
        """
        full_msg = f"[TIMING] {msg}"
        self.status_update.emit(full_msg)
        logger.info(full_msg)

    def _execute_capture(self, capture_count: int, timestamp: str) -> bool:
        if self._stop_event.is_set():
            self.status_update.emit(" Capture cancelled because stop was requested.")
            return False
        """Execute a complete capture + process cycle"""
        try:
            set_live_progress(
                phase="CAPTURING",
                active_zone="All Zones",
                images_captured=0,
                total_images=len(self.capture_sides),
                message="Capturing images from cameras",
            )
 
            cycle_t0 = time.perf_counter()
            cycle_start_wall = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

            self._timing_log(
                f"CYCLE_START | capture_count={capture_count} | "
                f"wall_time={cycle_start_wall} | "
                f"note=timer starts before capture_all call"
            )

            self.status_update.emit(
                f" Capturing images from {len(self.multi_camera_manager.cameras)} cameras..."
            )

            capture_t0 = time.perf_counter()
            self._timing_log(
                f"CAPTURE_CALL_START | capture_count={capture_count} | "
                f"sides={','.join(self.capture_sides)}"
            )

            try:
                images = self.multi_camera_manager.capture_all(
                    sides_to_capture=self.capture_sides,
                )

                capture_sec = time.perf_counter() - capture_t0
                self._timing_log(
                    f"CAPTURE_CALL_DONE | capture_count={capture_count} | "
                    f"time={capture_sec:.3f}s | "
                    f"note=includes trigger wait if capture_all waits internally"
                )

                # Optional: if HARDWARE_TRIGGER.py exposes exact internal timings,
                # this will print them also.
                camera_timing = getattr(self.multi_camera_manager, "last_capture_timing", None)
                if isinstance(camera_timing, dict):
                    for k, v in camera_timing.items():
                        self._timing_log(
                            f"CAMERA_INTERNAL | capture_count={capture_count} | {k}={v}"
                        )

             
                if self._stop_event.is_set():
                    self.status_update.emit(" Capture stopped during camera acquisition.")
                    return False
                missing_capture_sides = [
                    side for side in self.capture_sides
                    if side not in images or images.get(side) is None
                ]

                if missing_capture_sides:
                    raise RuntimeError(
                        "Camera capture failed / missing images for: "
                        + ", ".join(missing_capture_sides)
                    )
                
            except TypeError:
                images = self.multi_camera_manager.capture_all()

                capture_sec = time.perf_counter() - capture_t0
                self._timing_log(
                    f"CAPTURE_CALL_DONE | capture_count={capture_count} | "
                    f"time={capture_sec:.3f}s | "
                    f"mode=old_capture_all_signature"
                )

                camera_timing = getattr(self.multi_camera_manager, "last_capture_timing", None)
                if isinstance(camera_timing, dict):
                    for k, v in camera_timing.items():
                        self._timing_log(
                            f"CAMERA_INTERNAL | capture_count={capture_count} | {k}={v}"
                        )

                if self._stop_event.is_set():
                    self.status_update.emit(" Capture stopped during camera acquisition.")
                    return False
           
            if not images or not any(img is not None for img in images.values()):
                self.status_update.emit(" Capture failed - no images received")
                self.processing_error.emit("No images captured")
                return False
           
            success_count = sum(1 for img in images.values() if img is not None)
 
            set_live_progress(
                phase="CAPTURING",
                active_zone="All Zones",
                images_captured=success_count,
                total_images=len(self.capture_sides),
                message=f"Captured {success_count}/{len(self.capture_sides)} images",
            )
 
            self.status_update.emit(f"   Captured: {success_count}/{len(images)} cameras")
            self.capture_completed.emit(images)
           
            cycle_capture_dir, cycle_id = build_cycle_capture_dir(
                self.media_root,
                sku_name=self.sku_name,
            )
            self.status_update.emit(f" Cycle directory: {cycle_id}")
           
            self.status_update.emit(" Saving images...")

            save_t0 = time.perf_counter()
            image_map = self._save_images_to_cycle(images, cycle_capture_dir)
            save_sec = time.perf_counter() - save_t0

            self._timing_log(
                f"IMAGE_SAVE_DONE | cycle_id={cycle_id} | "
                f"time={save_sec:.3f}s | saved_sides={len(image_map)}"
            )
           
            if not image_map:
                self.status_update.emit(" No images saved to cycle directory")
                self.processing_error.emit("Failed to save images")
                return False
           
            self.images_saved.emit(image_map)
 
            set_live_progress(
                phase="CAPTURING",
                active_zone="All Zones",
                images_captured=len(image_map),
                total_images=len(self.sides_to_run),
                message=f"Saved {len(image_map)}/{len(self.sides_to_run)} AI side images",
            )
 
            self.status_update.emit(f"   Saved {len(image_map)} sides: {', '.join(image_map.keys())}")
           
            self.processing_started.emit(cycle_id)
            self.status_update.emit(f" Starting AI pipeline for {cycle_id}...")
           
            set_live_progress(
                phase="INFERENCE",
                active_zone="All Zones",
                images_captured=len(image_map),
                total_images=len(self.sides_to_run),
                message=f"AI inference started for {cycle_id}",
            )
            if self._stop_event.is_set():
                self.status_update.emit(" AI pipeline skipped because stop was requested.")
                return False
            ai_t0 = time.perf_counter()
            result = self._run_ai_pipeline(image_map, cycle_id, cycle_capture_dir)
            ai_sec = time.perf_counter() - ai_t0

            self._timing_log(
                f"AI_PIPELINE_DONE | cycle_id={cycle_id} | "
                f"time={ai_sec:.3f}s"
            )
           
            if result:
                total_cycle_sec = time.perf_counter() - cycle_t0

                result["timing_capture_call_sec"] = round(capture_sec, 3)
                result["timing_image_save_sec"] = round(save_sec, 3)
                result["timing_ai_pipeline_sec"] = round(ai_sec, 3)
                result["timing_total_from_capture_call_sec"] = round(total_cycle_sec, 3)

                self._timing_log(
                    f"CYCLE_TOTAL | cycle_id={cycle_id} | "
                    f"capture_call={capture_sec:.3f}s | "
                    f"save={save_sec:.3f}s | "
                    f"ai={ai_sec:.3f}s | "
                    f"total={total_cycle_sec:.3f}s"
                )
                set_live_progress(
                    phase="COMPLETED",
                    active_zone="All Zones",
                    images_captured=len(self.sides_to_run),
                    total_images=len(self.sides_to_run),
                    message=f"Cycle completed: {result.get('final_label', 'Unknown')}",
                )
 
                self.processing_completed.emit(result)
                final_label = result.get('final_label', 'Unknown')
                cycle_time = result.get(
                    'timing_total_from_capture_call_sec',
                    result.get('cycle_latency_sec', 0)
                )
               
                self.status_update.emit("")
                self.status_update.emit(f" ═══ CYCLE #{capture_count} COMPLETE ═══")
                self.status_update.emit(f"   Cycle ID: {cycle_id}")
                self.status_update.emit(f"   Result: {final_label}")
                self.status_update.emit(f"   Time: {cycle_time:.2f}s")
                self.status_update.emit("─" * 40)
                self.status_update.emit(" Waiting for next trigger...")
            else:
                self.processing_error.emit("AI pipeline returned no result")
                return False
           
            return True
           
        except Exception as e:
            set_live_progress(
                phase="FAILED",
                active_zone="-",
                message=f"Capture cycle error: {e}",
            )
 
            error_msg = f"Capture cycle error: {e}"
            self.status_update.emit(f" {error_msg}")
            self.processing_error.emit(error_msg)
            traceback.print_exc()
            return False
   
    def _save_images_to_cycle(self, images: Dict[str, np.ndarray], cycle_dir: str) -> Dict[str, str]:
        """
        Save captured numpy arrays to cycle directory.

        Supports both old serial-keyed result:
            {"254901432": image}

        and new role/side-keyed result from updated HARDWARE_TRIGGER.py:
            {"sidewall1": image, "innerwall": image, "bead": image}
        """
        import cv2

        image_map = {}
        known_sides = set(self.side_to_camera.keys()) | set(self.sides_to_run)

        for image_key, img_array in images.items():
            image_key = str(image_key)

            if img_array is None:
                self.status_update.emit(f"     No image from {image_key}")
                continue

            # New camera manager returns side names directly.
            if image_key in known_sides:
                side_name = image_key
            else:
                # Backward compatibility: old camera manager returned serial numbers.
                side_name = self.camera_to_side.get(image_key)

                if side_name is None:
                    for cam_serial, side in self.camera_to_side.items():
                        if image_key in str(cam_serial) or str(cam_serial) in image_key:
                            side_name = side
                            break

            if side_name is None:
                self.status_update.emit(f"     Unknown camera/side key: {image_key}")
                side_name = f"camera_{image_key}"

            side_dir = os.path.join(cycle_dir, side_name)
            os.makedirs(side_dir, exist_ok=True)

            img_path = os.path.join(side_dir, f"{side_name}.png")

            try:
                if img_array.dtype == np.uint16:
                    cv2.imwrite(img_path, img_array)

                elif img_array.dtype == np.uint8:
                    if len(img_array.shape) == 2:
                        cv2.imwrite(img_path, img_array)
                    elif len(img_array.shape) == 3 and img_array.shape[2] == 1:
                        cv2.imwrite(img_path, img_array[:, :, 0])
                    else:
                        cv2.imwrite(img_path, cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY))

                else:
                    img_8bit = img_array.astype(np.uint8)
                    if len(img_8bit.shape) == 2:
                        cv2.imwrite(img_path, img_8bit)
                    elif len(img_8bit.shape) == 3 and img_8bit.shape[2] == 1:
                        cv2.imwrite(img_path, img_8bit[:, :, 0])
                    else:
                        cv2.imwrite(img_path, cv2.cvtColor(img_8bit, cv2.COLOR_RGB2GRAY))

                if os.path.exists(img_path):
                    file_size = os.path.getsize(img_path) / 1024

                    if side_name in self.sides_to_run:
                        image_map[side_name] = img_path
                        self.status_update.emit(
                            f"    {side_name} saved ({img_array.shape}, {file_size:.1f}KB)"
                        )
                    else:
                        self.status_update.emit(
                            f"    {side_name} saved but not selected for AI ({img_array.shape}, {file_size:.1f}KB)"
                        )

            except Exception as e:
                self.status_update.emit(f"    Error saving {side_name}: {e}")

        return image_map
   
    def _run_ai_pipeline(self, image_map: Dict[str, str], cycle_id: str, cycle_capture_dir: str) -> Optional[Dict[str, Any]]:
        """Run the AI pipeline on captured images"""
        try:
            self.status_update.emit("─" * 40)
            self.status_update.emit(f"[AI PIPELINE] Starting for {cycle_id}")
           
            missing_sides = [s for s in self.sides_to_run if s not in image_map]
            if missing_sides:
                error_msg = f"Missing images for sides: {', '.join(missing_sides)}"
                self.processing_error.emit(error_msg)
                return None
           
            runtime_t0 = time.perf_counter()
            runtimes = self._get_or_load_runtimes()
            runtime_sec = time.perf_counter() - runtime_t0

            self._timing_log(
                f"AI_RUNTIME_READY | cycle_id={cycle_id} | "
                f"time={runtime_sec:.3f}s | preloaded={self._runtimes_preloaded}"
            )
            if runtimes is None:
                self.processing_error.emit("Failed to load AI runtimes")
                return None
           
            r_gpu_sem = threading.Semaphore(R_ALIGN_GPU_CONCURRENCY)
            vit_gpu_sem = threading.Semaphore(VIT_GPU_CONCURRENCY)
            yolo_gpu_sem = threading.Semaphore(YOLO_GPU_CONCURRENCY)
           
            date_str = datetime.now().strftime("%d-%m-%Y")
 
            output_root = os.path.join(
                self.media_root,
                "Output",
                self.sku_name,
                date_str,
            )
 
            os.makedirs(output_root, exist_ok=True)
           
            self.status_update.emit("🚀 Running AI inference on all sides...")
            run_cycle_t0 = time.perf_counter()
            result = run_cycle(
                image_map=image_map,
                runtimes=runtimes,
                output_root=output_root,
                cycle_id=cycle_id,
                sides_to_run=self.sides_to_run,
                r_gpu_sem=r_gpu_sem,
                vit_gpu_sem=vit_gpu_sem,
                yolo_gpu_sem=yolo_gpu_sem,
                sku_name=self.sku_name,
                tyre_name=self.tyre_name,
            )
            if not isinstance(result, dict):
                self.processing_error.emit("run_cycle returned invalid result")
                return None
            run_cycle_sec = time.perf_counter() - run_cycle_t0

            self._timing_log(
                f"RUN_CYCLE_DONE | cycle_id={cycle_id} | "
                f"time={run_cycle_sec:.3f}s | sides={','.join(self.sides_to_run)}"
            )

            if isinstance(result, dict):
                result.setdefault("timing", {})
                result["timing"]["runtime_ready_sec"] = round(runtime_sec, 3)
                result["timing"]["run_cycle_sec"] = round(run_cycle_sec, 3)
            try:
                save_cycle_metadata(result)
            except Exception:
                pass
           
            side_results = result.get('side_results', {})
            for side_name in self.sides_to_run:
                side_result = side_results.get(side_name, {})
                label = side_result.get('final_label', 'UNKNOWN')
                self.status_update.emit(f"   {side_name}: {label}")
           
            return result
           
        except Exception as e:
            error_msg = f"AI pipeline error: {e}"
            self.processing_error.emit(error_msg)
            traceback.print_exc()
            return None
   
    def _cleanup(self):
        """Clean shutdown"""
        with self._cleanup_lock:
            if self._cleanup_done:
                return
            self._cleanup_done = True

        self.status_update.emit(" Cleaning up live inspection resources...")

        try:
            if self.multi_camera_manager is not None:
                # This is important: it unblocks PLC wait / camera wait.
                if hasattr(self.multi_camera_manager, "_stop_event"):
                    self.multi_camera_manager._stop_event.set()

                if hasattr(self.multi_camera_manager, "stop_all_streams"):
                    self.multi_camera_manager.stop_all_streams()

        except Exception as e:
            self.status_update.emit(f" Camera cleanup warning: {e}")

        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

        self.status_update.emit(" Live inspection cleanup completed")
   
    def stop(self):
        """Signal the worker to stop and unblock camera/PLC waits."""
        self.status_update.emit(" Stop signal received...")

        # Stop main worker loop
        self._stop_event.set()

        # If worker is waiting for user OK popup confirmation, release that wait also
        try:
            self._ready_confirm_event.set()
        except Exception:
            pass

        # Immediately tell camera manager to stop waiting/capturing
        try:
            if self.multi_camera_manager is not None:
                if hasattr(self.multi_camera_manager, "_stop_event"):
                    self.multi_camera_manager._stop_event.set()
        except Exception:
            pass

        # Run cleanup in background so GUI does not freeze while closing
        try:
            threading.Thread(
                target=self._cleanup,
                daemon=True,
            ).start()
        except Exception:
            self._cleanup()
   
    def is_running(self) -> bool:
        """Check if worker is running"""
        return self._is_running
 
 
# =========================================================
# CONVENIENCE FUNCTION (called from GUI)
# =========================================================
 
def start_continuous_cycle(
    media_root: str,
    sku_name: str,
    tyre_name: str,
    multi_camera_manager,
    min_capture_interval: float = 2.0,
    seg_model_a_path: Optional[str] = None,
    seg_model_b_path: Optional[str] = None,
    vit_checkpoint_path: Optional[str] = None,
    r_detector_path: Optional[str] = None,
    device: str = DEVICE,
    sides_to_run: Optional[List[str]] = None,
    capture_sides: Optional[List[str]] = None,
    side_configs: Optional[Dict[str, Dict[str, Any]]] = None,
    auto_preload: bool = True,
    on_capture_started: Optional[Callable] = None,
    on_capture_completed: Optional[Callable] = None,
    on_images_saved: Optional[Callable] = None,
    on_processing_started: Optional[Callable] = None,
    on_processing_completed: Optional[Callable] = None,
    on_processing_error: Optional[Callable] = None,
    on_status_update: Optional[Callable] = None,
) -> ContinuousCycleWorker:
    """Create a ContinuousCycleWorker"""
   
    sides_to_run = _resolve_sides(sides_to_run)
    media_root = os.path.abspath(media_root)
    device = _normalize_device(device)
    os.makedirs(media_root, exist_ok=True)
   
    worker = ContinuousCycleWorker(
        media_root=media_root,
        sku_name=sku_name,
        tyre_name=tyre_name,
        device=device,
        seg_model_a_path=seg_model_a_path,
        seg_model_b_path=seg_model_b_path,
        vit_checkpoint_path=vit_checkpoint_path,
        r_detector_path=r_detector_path,
        multi_camera_manager=multi_camera_manager,
        min_capture_interval=min_capture_interval,
        sides_to_run=sides_to_run,
        capture_sides=capture_sides,
        side_configs=side_configs,
        auto_preload=auto_preload,
    )
   
    if on_capture_started:
        worker.capture_started.connect(on_capture_started)
    if on_capture_completed:
        worker.capture_completed.connect(on_capture_completed)
    if on_images_saved:
        worker.images_saved.connect(on_images_saved)
    if on_processing_started:
        worker.processing_started.connect(on_processing_started)
    if on_processing_completed:
        worker.processing_completed.connect(on_processing_completed)
    if on_processing_error:
        worker.processing_error.connect(on_processing_error)
    if on_status_update:
        worker.status_update.connect(on_status_update)
   
    return worker
 
 
# =========================================================
# ORIGINAL FUNCTIONS (backward compatibility)
# =========================================================
 
def resolve_cycle_capture_dir(
    media_root: str,
    cycle_id: Optional[str],
    demo_capture_root: Optional[str],
    sku_name: str = "UNKNOWN_SKU",
) -> tuple[str, str]:
    if CAMERA_CAPTURE_ENABLED:
        cycle_capture_dir, cycle_id = build_cycle_capture_dir(
            media_root,
            sku_name=sku_name,
        )
        return cycle_capture_dir, cycle_id
 
    if demo_capture_root:
        cycle_capture_dir = os.path.abspath(demo_capture_root)
        if cycle_id is None:
            cycle_id = f"Cycle_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        return cycle_capture_dir, cycle_id
 
    today_root = _get_today_capture_root(
        media_root,
        sku_name=sku_name,
    )
    existing = [
        d for d in os.listdir(today_root)
        if os.path.isdir(os.path.join(today_root, d)) and d.startswith("Cycle_")
    ]
    if not existing:
        raise FileNotFoundError(f"No Cycle_N folders found under {today_root}.")
    existing.sort(key=lambda d: int(d.split("_", 1)[1]))
    latest_cycle = existing[-1]
    cycle_capture_dir = os.path.join(today_root, latest_cycle)
    cycle_id = cycle_id or latest_cycle
    return cycle_capture_dir, cycle_id
 
 
def build_cycle_image_map(
    cycle_capture_dir: str,
    sides_to_run: List[str],
    multi_camera_manager=None,
) -> Dict[str, str]:
    if CAMERA_CAPTURE_ENABLED:
        if multi_camera_manager is None:
            raise ValueError("CAMERA_CAPTURE_ENABLED=True but multi_camera_manager was not passed.")
        return capture_and_save_images(
            multi_camera_manager=multi_camera_manager,
            cycle_capture_dir=cycle_capture_dir,
            sides_to_run=sides_to_run,
        )
    return build_image_map_from_capture_dir(
        cycle_capture_dir=cycle_capture_dir,
        sides_to_run=sides_to_run,
    )
 
 
def prepare_runtimes_for_cycle(
    sku_name: str, media_root: str, cycle_capture_dir: str,
    device: str, seg_model_a_path: str, seg_model_b_path: str,
    vit_checkpoint_path: str, r_detector_path: str, tyre_name: str,
    side_configs: Optional[Dict[str, Dict[str, Any]]], sides_to_run: List[str],
):
    runtimes = build_all_runtimes(
        sku_name=sku_name, media_root=media_root,
        seg_model_a_path=seg_model_a_path, seg_model_b_path=seg_model_b_path,
        vit_checkpoint_path=vit_checkpoint_path, r_detector_path=r_detector_path,
        device=device, capture_root=cycle_capture_dir,
        tyre_name=tyre_name, side_configs=side_configs, sides_to_run=sides_to_run,
    )
    _apply_tyre_name_to_runtimes(runtimes, tyre_name)
    _maybe_warmup_runtimes(
        runtimes=runtimes, sku_name=sku_name, device=device,
        capture_root=cycle_capture_dir, seg_model_a_path=seg_model_a_path,
        seg_model_b_path=seg_model_b_path, vit_checkpoint_path=vit_checkpoint_path,
        r_detector_path=r_detector_path, tyre_name=tyre_name,
        media_root=media_root, sides_to_run=sides_to_run,
    )
    return runtimes
 
 
def build_gpu_semaphores():
    r_gpu_sem = threading.Semaphore(R_ALIGN_GPU_CONCURRENCY)
    vit_gpu_sem = threading.Semaphore(VIT_GPU_CONCURRENCY)
    yolo_gpu_sem = threading.Semaphore(YOLO_GPU_CONCURRENCY)
    return r_gpu_sem, vit_gpu_sem, yolo_gpu_sem
 
 
def print_cycle_inputs(sku_name, tyre_name, sku_calibration_dir, shared_artifacts_dir,
                       cycle_capture_dir, cycle_id, image_map, sides_to_run):
    print(f"[MAIN] selected sku_name     : {sku_name}")
    print(f"[MAIN] selected tyre_name    : {tyre_name}")
    print(f"[MAIN] sku_calibration_dir   : {sku_calibration_dir}")
    print(f"[MAIN] sku_artifacts_dir     : {shared_artifacts_dir}")
    print(f"[MAIN] cycle_capture_dir     : {cycle_capture_dir}")
    print(f"[MAIN] cycle_id              : {cycle_id}")
    print("[MAIN] image_map:")
    for side_name in sides_to_run:
        print(f"    {side_name}: {image_map.get(side_name, 'MISSING')}")
 
 
def run_capture_folder_cycle(
    media_root: str, sku_name: str = "SKU_001",
    cycle_id: Optional[str] = None, device: str = DEVICE,
    seg_model_a_path: Optional[str] = None, seg_model_b_path: Optional[str] = None,
    vit_checkpoint_path: Optional[str] = None, r_detector_path: Optional[str] = None,
    tyre_name: str = "195_65_R15",
    side_configs: Optional[Dict[str, Dict[str, Any]]] = None,
    sides_to_run: Optional[List[str]] = None,
    multi_camera_manager=None, demo_capture_root: Optional[str] = None,
) -> Dict[str, Any]:
    sides_to_run = _resolve_sides(sides_to_run)
    media_root = os.path.abspath(media_root)
    device = _normalize_device(device)
    os.makedirs(media_root, exist_ok=True)
 
    seg_model_a_path = _required_file(seg_model_a_path, "seg_model_a_path")
    seg_model_b_path = _required_file(seg_model_b_path, "seg_model_b_path")
    vit_checkpoint_path = _required_file(vit_checkpoint_path, "vit_checkpoint_path")
    r_detector_path = _required_file(r_detector_path, "r_detector_path")
 
    sku_calibration_dir = _get_sku_calibration_dir(media_root, sku_name)
    shared_artifacts_dir = _get_sku_artifacts_dir(media_root, sku_name)
 
    cycle_capture_dir, cycle_id = resolve_cycle_capture_dir(
        media_root=media_root,
        cycle_id=cycle_id,
        demo_capture_root=demo_capture_root,
        sku_name=sku_name,
    )
 
    image_map = build_cycle_image_map(
        cycle_capture_dir=cycle_capture_dir, sides_to_run=sides_to_run,
        multi_camera_manager=multi_camera_manager,
    )
   
    set_live_progress(
        phase="CAPTURING",
        active_zone="All Zones",
        images_captured=len(image_map),
        total_images=len(sides_to_run),
        message=f"Images ready: {len(image_map)}/{len(sides_to_run)}",
    )
    print_cycle_inputs(sku_name, tyre_name, sku_calibration_dir, shared_artifacts_dir,
                       cycle_capture_dir, cycle_id, image_map, sides_to_run)
 
    runtimes = prepare_runtimes_for_cycle(
        sku_name=sku_name, media_root=media_root, cycle_capture_dir=cycle_capture_dir,
        device=device, seg_model_a_path=seg_model_a_path, seg_model_b_path=seg_model_b_path,
        vit_checkpoint_path=vit_checkpoint_path, r_detector_path=r_detector_path,
        tyre_name=tyre_name, side_configs=side_configs, sides_to_run=sides_to_run,
    )
 
    r_gpu_sem, vit_gpu_sem, yolo_gpu_sem = build_gpu_semaphores()
 
    date_str = datetime.now().strftime("%d-%m-%Y")
 
    output_root = os.path.join(
        media_root,
        "Output",
        sku_name,
        date_str,
    )
 
    os.makedirs(output_root, exist_ok=True)
    set_live_progress(
        phase="INFERENCE",
        active_zone="All Zones",
        images_captured=len(image_map),
        total_images=len(sides_to_run),
        message="AI inference started",
    )
    result = run_cycle(
        image_map=image_map, runtimes=runtimes, output_root=output_root,
        cycle_id=cycle_id, sides_to_run=sides_to_run,
        r_gpu_sem=r_gpu_sem, vit_gpu_sem=vit_gpu_sem, yolo_gpu_sem=yolo_gpu_sem,
        sku_name=sku_name, tyre_name=tyre_name,
    )
 
    try:
        set_live_progress(
            phase="COMPLETED",
            active_zone="All Zones",
            images_captured=len(sides_to_run),
            total_images=len(sides_to_run),
            message="Inspection completed",
        )
        save_cycle_metadata(result)
    except Exception as e:
        print(f"[DB][ERROR] save failed | error={e}")
 
    return result
 
 
run_cycle_for_gui = run_capture_folder_cycle
 