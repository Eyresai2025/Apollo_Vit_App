import os
import threading
import time
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
)

# Import trigger mode from HARDWARE_TRIGGER
from src.camera.HARDWARE_TRIGGER import TRIGGER_MODE


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
    plc_status = pyqtSignal(bool)
    finished = pyqtSignal()
    error = pyqtSignal(str)
    
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
        plc_interface=None,
        plc_trigger_tag: str = "DB100.DBX0.0",
        min_capture_interval: float = 2.0,
        sides_to_run: Optional[List[str]] = None,
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
        self.plc_interface = plc_interface
        self.plc_trigger_tag = plc_trigger_tag
        self.min_capture_interval = min_capture_interval
        self.sides_to_run = _resolve_sides(sides_to_run)
        self.side_configs = side_configs
        self.auto_preload = auto_preload
        
        self._stop_event = threading.Event()
        self._is_running = False
        self._runtimes_preloaded = False
        self._runtimes = None
        self.is_hardware = (TRIGGER_MODE == "hardware")
        
        # Camera serial to side name mapping
        self.camera_to_side = {
            "244802149": "sidewall1",
            "244802163": "sidewall2",
            "251102086": "innerwall",
            "251401655": "tread",
            "251300826": "bead",
        }
        
        self.side_to_camera = {v: k for k, v in self.camera_to_side.items()}
        os.makedirs(self.media_root, exist_ok=True)
    
    @pyqtSlot()
    def run(self):
        """Main loop - monitors trigger and orchestrates everything"""
        self._is_running = True
        capture_count = 0
        last_capture_time = 0
        trigger_was_high = False
        plc_connection_was_ok = False
        
        self.status_update.emit("=" * 50)
        self.status_update.emit("🚀 Starting Continuous Inspection System")
        self.status_update.emit(f"   Trigger Mode: {TRIGGER_MODE.upper()}")
        self.status_update.emit(f"   SKU: {self.sku_name}")
        self.status_update.emit(f"   Tyre: {self.tyre_name}")
        self.status_update.emit(f"   Device: {self.device}")
        if not self.is_hardware:
            self.status_update.emit(f"   PLC Tag: {self.plc_trigger_tag}")
        self.status_update.emit(f"   Min Interval: {self.min_capture_interval}s")
        self.status_update.emit(f"   Sides: {', '.join(self.sides_to_run)}")
        self.status_update.emit("=" * 50)
        
        # Preload AI runtimes
        if self.auto_preload and not self._runtimes_preloaded:
            self._preload_runtimes()
        
        # Start camera streams
        if hasattr(self.multi_camera_manager, 'start_all_streams'):
            self.multi_camera_manager.start_all_streams()
            if self.is_hardware:
                self.status_update.emit("✅ Camera streams started - waiting for HARDWARE triggers")
            else:
                self.status_update.emit("✅ Camera streams started - waiting for PLC triggers")
        
        if self.is_hardware:
            self.status_update.emit("⏳ Waiting for hardware trigger signal...")
        else:
            self.status_update.emit(f"⏳ Monitoring PLC tag: {self.plc_trigger_tag}")
            self.status_update.emit("   Waiting for rising edge trigger...")
        
        # MAIN LOOP
        while not self._stop_event.is_set():
            try:
                should_capture = False
                
                if self.is_hardware:
                    # HARDWARE MODE: Capture immediately (cameras wait for hardware trigger internally)
                    current_time = time.time()
                    if current_time - last_capture_time >= self.min_capture_interval:
                        should_capture = True
                else:
                    # SOFTWARE MODE: Check PLC tag
                    plc_ok = self._check_plc_connection()
                    if plc_ok != plc_connection_was_ok:
                        plc_connection_was_ok = plc_ok
                        self.plc_status.emit(plc_ok)
                        if plc_ok:
                            self.status_update.emit("✅ PLC connection OK")
                        else:
                            self.status_update.emit("⚠️  PLC not connected - waiting...")
                    
                    trigger_value = self._read_plc_tag()
                    
                    # Rising edge detection
                    if trigger_value and not trigger_was_high:
                        current_time = time.time()
                        if current_time - last_capture_time >= self.min_capture_interval:
                            should_capture = True
                    
                    trigger_was_high = trigger_value
                
                if should_capture:
                    capture_count += 1
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    
                    self.status_update.emit("")
                    trigger_type = "HARDWARE" if self.is_hardware else "PLC"
                    self.status_update.emit(f"⚡ ═══ {trigger_type} TRIGGER #{capture_count} ═══")
                    self.status_update.emit(f"   Time: {timestamp}")
                    self.capture_started.emit(timestamp)
                    
                    capture_success = self._execute_capture(capture_count, timestamp)
                    
                    if capture_success:
                        last_capture_time = time.time()
                    else:
                        self.status_update.emit("⚠️  Capture skipped or failed")
                
                if not self.is_hardware:
                    time.sleep(0.01)
                    
            except Exception as e:
                error_msg = f"Continuous cycle error: {e}"
                self.status_update.emit(f"❌ {error_msg}")
                self.processing_error.emit(error_msg)
                traceback.print_exc()
                time.sleep(1)
        
        self._cleanup()
        self._is_running = False
        self.status_update.emit("✅ Continuous cycle stopped")
        self.finished.emit()
    
    def _check_plc_connection(self) -> bool:
        """Check if PLC is connected"""
        if self.plc_interface is None:
            return False
        try:
            return self.plc_interface.get_connected()
        except Exception:
            return False
    
    def _read_plc_tag(self) -> bool:
        """Read PLC trigger tag value"""
        if self.plc_interface is None:
            return False
        
        try:
            import re
            match = re.match(r'DB(\d+)\.DBX(\d+)\.(\d+)', self.plc_trigger_tag)
            if match:
                db_num = int(match.group(1))
                byte_num = int(match.group(2))
                bit_num = int(match.group(3))
                
                try:
                    result = self.plc_interface.read_area(0x82, db_num, byte_num, 1)
                    if result and len(result) > 0:
                        byte_val = result[0]
                        return bool((byte_val >> bit_num) & 1)
                except Exception:
                    pass
                
                try:
                    result = self.plc_interface.db_read(db_num, byte_num, 1)
                    if result and len(result) > 0:
                        byte_val = result[0]
                        return bool((byte_val >> bit_num) & 1)
                except Exception:
                    pass
            
            try:
                result = self.plc_interface.read_area(0x82, 100, 0, 1)
                return bool(result[0]) if result else False
            except Exception:
                pass
        except Exception:
            pass
        
        return False
    
    def _preload_runtimes(self):
        """Preload AI runtimes"""
        try:
            self.status_update.emit("🔄 Preloading AI models...")
            
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
            self.status_update.emit("✅ AI models preloaded successfully")
            
        except Exception as e:
            self._runtimes_preloaded = False
            self.status_update.emit(f"⚠️  Runtime preload failed: {e}")
    
    def _get_or_load_runtimes(self):
        """Get cached runtimes or load them"""
        if self._runtimes_preloaded and self._runtimes is not None:
            return self._runtimes
        self._preload_runtimes()
        return self._runtimes
    
    def _execute_capture(self, capture_count: int, timestamp: str) -> bool:
        """Execute a complete capture + process cycle"""
        try:
            self.status_update.emit(f"📸 Capturing images from {len(self.multi_camera_manager.cameras)} cameras...")
            
            images = self.multi_camera_manager.capture_all()
            
            if not images or not any(img is not None for img in images.values()):
                self.status_update.emit("❌ Capture failed - no images received")
                self.processing_error.emit("No images captured")
                return False
            
            success_count = sum(1 for img in images.values() if img is not None)
            self.status_update.emit(f"   Captured: {success_count}/{len(images)} cameras")
            self.capture_completed.emit(images)
            
            cycle_capture_dir, cycle_id = build_cycle_capture_dir(self.media_root)
            self.status_update.emit(f"📁 Cycle directory: {cycle_id}")
            
            self.status_update.emit("💾 Saving images...")
            image_map = self._save_images_to_cycle(images, cycle_capture_dir)
            
            if not image_map:
                self.status_update.emit("❌ No images saved to cycle directory")
                self.processing_error.emit("Failed to save images")
                return False
            
            self.images_saved.emit(image_map)
            self.status_update.emit(f"   Saved {len(image_map)} sides: {', '.join(image_map.keys())}")
            
            self.processing_started.emit(cycle_id)
            self.status_update.emit(f"🤖 Starting AI pipeline for {cycle_id}...")
            
            result = self._run_ai_pipeline(image_map, cycle_id, cycle_capture_dir)
            
            if result:
                self.processing_completed.emit(result)
                final_label = result.get('final_label', 'Unknown')
                cycle_time = result.get('cycle_latency_sec', 0)
                
                self.status_update.emit("")
                self.status_update.emit(f"✅ ═══ CYCLE #{capture_count} COMPLETE ═══")
                self.status_update.emit(f"   Cycle ID: {cycle_id}")
                self.status_update.emit(f"   Result: {final_label}")
                self.status_update.emit(f"   Time: {cycle_time:.2f}s")
                self.status_update.emit("─" * 40)
                self.status_update.emit("⏳ Waiting for next trigger...")
            else:
                self.processing_error.emit("AI pipeline returned no result")
                return False
            
            return True
            
        except Exception as e:
            error_msg = f"Capture cycle error: {e}"
            self.status_update.emit(f"❌ {error_msg}")
            self.processing_error.emit(error_msg)
            traceback.print_exc()
            return False
    
    def _save_images_to_cycle(self, images: Dict[str, np.ndarray], cycle_dir: str) -> Dict[str, str]:
        """Save captured numpy arrays to cycle directory"""
        import cv2
        
        image_map = {}
        
        for serial, img_array in images.items():
            if img_array is None:
                self.status_update.emit(f"   ⚠️  No image from camera {serial}")
                continue
            
            side_name = self.camera_to_side.get(str(serial))
            
            if side_name is None:
                for cam_serial, side in self.camera_to_side.items():
                    if str(serial) in cam_serial or cam_serial in str(serial):
                        side_name = side
                        break
            
            if side_name is None or side_name not in self.sides_to_run:
                self.status_update.emit(f"   ⚠️  Unknown camera serial: {serial}")
                side_name = f"camera_{serial}"
            
            side_dir = os.path.join(cycle_dir, side_name)
            os.makedirs(side_dir, exist_ok=True)
            
            img_path = os.path.join(side_dir, f"{side_name}.png")
            
            try:
                if img_array.dtype == np.uint16:
                    img_min = img_array.min()
                    img_max = img_array.max()
                    if img_max > img_min:
                        img_8bit = ((img_array - img_min) / (img_max - img_min) * 255).astype(np.uint8)
                    else:
                        img_8bit = np.zeros_like(img_array, dtype=np.uint8)
                elif img_array.dtype == np.uint8:
                    img_8bit = img_array
                else:
                    img_8bit = img_array.astype(np.uint8)
                
                if len(img_8bit.shape) == 2:
                    cv2.imwrite(img_path, img_8bit)
                elif len(img_8bit.shape) == 3 and img_8bit.shape[2] == 1:
                    cv2.imwrite(img_path, img_8bit[:, :, 0])
                else:
                    cv2.imwrite(img_path, cv2.cvtColor(img_8bit, cv2.COLOR_RGB2GRAY))
                
                if os.path.exists(img_path):
                    image_map[side_name] = img_path
                    file_size = os.path.getsize(img_path) / 1024
                    self.status_update.emit(f"   ✅ {side_name} saved ({img_array.shape}, {file_size:.1f}KB)")
                    
            except Exception as e:
                self.status_update.emit(f"   ❌ Error saving {side_name}: {e}")
        
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
            
            runtimes = self._get_or_load_runtimes()
            if runtimes is None:
                self.processing_error.emit("Failed to load AI runtimes")
                return None
            
            r_gpu_sem = threading.Semaphore(R_ALIGN_GPU_CONCURRENCY)
            vit_gpu_sem = threading.Semaphore(VIT_GPU_CONCURRENCY)
            yolo_gpu_sem = threading.Semaphore(YOLO_GPU_CONCURRENCY)
            
            output_root = os.path.join(self.media_root, "output")
            os.makedirs(output_root, exist_ok=True)
            
            self.status_update.emit("🚀 Running AI inference on all sides...")
            
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
        try:
            if hasattr(self.multi_camera_manager, 'stop_all_streams'):
                self.multi_camera_manager.stop_all_streams()
        except Exception:
            pass
        
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
    
    def stop(self):
        """Signal the worker to stop"""
        self.status_update.emit("🛑 Stop signal received...")
        self._stop_event.set()
    
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
    plc_interface=None,
    plc_trigger_tag: str = "DB100.DBX0.0",
    min_capture_interval: float = 2.0,
    seg_model_a_path: Optional[str] = None,
    seg_model_b_path: Optional[str] = None,
    vit_checkpoint_path: Optional[str] = None,
    r_detector_path: Optional[str] = None,
    device: str = DEVICE,
    sides_to_run: Optional[List[str]] = None,
    side_configs: Optional[Dict[str, Dict[str, Any]]] = None,
    auto_preload: bool = True,
    on_capture_started: Optional[Callable] = None,
    on_capture_completed: Optional[Callable] = None,
    on_images_saved: Optional[Callable] = None,
    on_processing_started: Optional[Callable] = None,
    on_processing_completed: Optional[Callable] = None,
    on_processing_error: Optional[Callable] = None,
    on_status_update: Optional[Callable] = None,
    on_plc_status: Optional[Callable] = None,
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
        plc_interface=plc_interface,
        plc_trigger_tag=plc_trigger_tag,
        min_capture_interval=min_capture_interval,
        sides_to_run=sides_to_run,
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
    if on_plc_status:
        worker.plc_status.connect(on_plc_status)
    
    return worker


# =========================================================
# ORIGINAL FUNCTIONS (backward compatibility)
# =========================================================

def resolve_cycle_capture_dir(
    media_root: str,
    cycle_id: Optional[str],
    demo_capture_root: Optional[str],
) -> tuple[str, str]:
    if CAMERA_CAPTURE_ENABLED:
        cycle_capture_dir, cycle_id = build_cycle_capture_dir(media_root)
        return cycle_capture_dir, cycle_id

    if demo_capture_root:
        cycle_capture_dir = os.path.abspath(demo_capture_root)
        if cycle_id is None:
            cycle_id = f"Cycle_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        return cycle_capture_dir, cycle_id

    today_root = _get_today_capture_root(media_root)
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
        media_root=media_root, cycle_id=cycle_id, demo_capture_root=demo_capture_root,
    )

    image_map = build_cycle_image_map(
        cycle_capture_dir=cycle_capture_dir, sides_to_run=sides_to_run,
        multi_camera_manager=multi_camera_manager,
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

    output_root = os.path.join(media_root, "output")
    os.makedirs(output_root, exist_ok=True)

    result = run_cycle(
        image_map=image_map, runtimes=runtimes, output_root=output_root,
        cycle_id=cycle_id, sides_to_run=sides_to_run,
        r_gpu_sem=r_gpu_sem, vit_gpu_sem=vit_gpu_sem, yolo_gpu_sem=yolo_gpu_sem,
        sku_name=sku_name, tyre_name=tyre_name,
    )

    try:
        save_cycle_metadata(result)
    except Exception as e:
        print(f"[DB][ERROR] save failed | error={e}")

    return result


run_cycle_for_gui = run_capture_folder_cycle