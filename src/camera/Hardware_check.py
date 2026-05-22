"""

=========================================================

ROBUST MULTI-STATION SYSTEM - WITH GRACEFUL CAMERA FAILURE

Handles missing/faulty cameras without breaking production

=========================================================

"""

from arena_api.system import system

from arena_api.buffer import BufferFactory

import ctypes

import numpy as np

import cv2

import threading

import queue

import os

import signal

import time

from datetime import datetime

from collections import defaultdict
 
# =========================================================

# PRODUCTION MODE

# =========================================================

MODE = "AUTO"  # "FREE" for testing, "AUTO" for production
 
# =========================================================

# CAMERA AVAILABILITY STATUS (Set based on actual hardware)

# =========================================================

CAMERA_AVAILABLE = {

    "254901430": True,   # Tread - Working

    "254901432": True,   # Sidewall1 - Working

    "254901428": True,   # Sidewall2 - Working

    "250500042": True,  # Inner -Working

    "220903275": False,  # Bead - Data cable issue (NOT available)

}
 
# How to handle missing cameras: "SKIP", "SIMULATE", "ALERT_ONLY"

MISSING_CAMERA_HANDLING = "SKIP"  # Change to "SIMULATE" for testing with dummy data
 
# =========================================================

# CAMERA ASSIGNMENTS BY STATION (Keep original logic)

# =========================================================

STATION1_CAMERAS = {

    "254901430": "Tread",      # 4K

    "254901432": "Sidewall1",  # 4K

    "254901428": "Sidewall2",  # 4K

    "250500042": "Inner",      # 2K 

}
 
STATION2_CAMERAS = {

    "220903275": "Bead",       # 2K (Data cable issue)

}

# =========================================================
# ALL CAMERA LIST
# =========================================================

ALL_CAMERAS = {
    **STATION1_CAMERAS,
    **STATION2_CAMERAS,
}

ALL_CAMERA_SERIALS = list(ALL_CAMERAS.keys())
# =========================================================

# FILTER AVAILABLE CAMERAS ONLY

# =========================================================

def get_available_cameras(camera_dict):

    """Return only cameras that are marked as available"""

    return {

        serial: role 

        for serial, role in camera_dict.items() 

        if CAMERA_AVAILABLE.get(serial, False)

    }
 
# Get only available cameras for each station

AVAILABLE_STATION1_CAMERAS = get_available_cameras(STATION1_CAMERAS)

AVAILABLE_STATION2_CAMERAS = get_available_cameras(STATION2_CAMERAS)
 
# All available cameras combined

AVAILABLE_CAMERAS = {**AVAILABLE_STATION1_CAMERAS, **AVAILABLE_STATION2_CAMERAS}

CAMERA_SERIALS = list(ALL_CAMERAS.keys())
 
# =========================================================

# CAPTURE SETTINGS

# =========================================================

SAVE_DIR = r"C:\Users\PrajwalSridhar\Desktop\Apollo_share\155_65_R14_AMZ4G\Good"

NUM_TIRES = 50

PATCH_HEIGHT = 14000

NUM_PATCHES = 3

FINAL_HEIGHT = PATCH_HEIGHT * NUM_PATCHES

NUM_STREAM_BUFFERS = 16

SAVE_QUEUE_SIZE = 100

PNG_COMPRESSION = 0
 
# =========================================================

# TIMING SETTINGS

# =========================================================

TRIGGER_TIMEOUT = 30

POST_TRIGGER_DELAY = 0.1
 
# =========================================================

# CAMERA CONFIGURATIONS

# =========================================================

FOUR_K_CONFIG = {

    "name": "4K",

    "width": 4096,

    "patch_height": 14000,

    "final_height": 42000,

    "num_patches": 3,

    "line_rate": 8016.0,

    "line_rate_enable": True,

    "pixel_format": "Mono16",

    "exposure_us": 120.0,

    "gain_db": 24.0,

    "packet_size": 9000,

    "packet_delay": 1000,

}
 
TWO_K_CONFIG = {

    "name": "2K",

    "width": 2048,

    "patch_height": 14000,

    "final_height": 42000,

    "num_patches": 3,

    "line_rate": None,

    "line_rate_enable": False,

    "pixel_format": "Mono16",

    "exposure_us": 120.0,

    "gain_db": 24.0,

    "packet_size": 9000,

    "packet_delay": 1000,

}
 
# =========================================================

# TRIGGER COORDINATOR (Modified for missing cameras)

# =========================================================

class TriggerCoordinator:

    def __init__(self):

        self.lock = threading.RLock()

        self.trigger_count = 0

        self.current_tire = 0

        self.station1_complete = False

        self.station2_complete = False

        self.cameras_ready = defaultdict(bool)

        self.missing_cameras_logged = set()

    def get_trigger_number(self):

        with self.lock:

            if not self.station1_complete:

                return 1

            elif not self.station2_complete:

                return 2

            else:

                return 0

    def should_capture(self, serial):

        """Determine if a camera should capture - only if available"""

        # First check if camera is available

        if not CAMERA_AVAILABLE.get(serial, False):

            return False

        with self.lock:

            trigger_num = self.get_trigger_number()

            if trigger_num == 1:

                return serial in AVAILABLE_STATION1_CAMERAS

            elif trigger_num == 2:

                return serial in AVAILABLE_STATION2_CAMERAS

            else:

                return False

    def mark_trigger_received(self, serial):

        """Mark that a camera completed capture"""

        with self.lock:

            trigger_num = self.get_trigger_number()

            if trigger_num == 1:

                self.cameras_ready[f"station1_{serial}"] = True

                # Check if ALL AVAILABLE station 1 cameras are ready

                expected_cameras = set(AVAILABLE_STATION1_CAMERAS.keys())

                actual_ready = {

                    serial for s in expected_cameras 

                    if self.cameras_ready[f"station1_{s}"]

                }

                if actual_ready == expected_cameras:

                    self.station1_complete = True

                    print(f"\n[COORDINATOR] ✓ Station 1 complete for tire {self.current_tire + 1}")

                    print(f"  Cameras captured: {len(actual_ready)}/{len(expected_cameras)}")

            elif trigger_num == 2:

                self.station2_complete = True

                print(f"\n[COORDINATOR] ✓ Station 2 complete for tire {self.current_tire + 1}")

    def next_tire(self):

        with self.lock:

            self.current_tire += 1

            self.trigger_count = 0

            self.station1_complete = False

            self.station2_complete = False

            self.cameras_ready.clear()

            print(f"\n[COORDINATOR] Starting tire {self.current_tire + 1}")

            return self.current_tire

    def is_cycle_complete(self):

        with self.lock:

            # Consider cycle complete if either:

            # 1. Both stations have available cameras and both are complete, OR

            # 2. Station 2 has no cameras (only station 1 matters)

            station1_done = self.station1_complete or len(AVAILABLE_STATION1_CAMERAS) == 0

            station2_done = self.station2_complete or len(AVAILABLE_STATION2_CAMERAS) == 0

            return station1_done and station2_done

    def get_progress(self):

        with self.lock:

            if not self.station1_complete and len(AVAILABLE_STATION1_CAMERAS) > 0:

                ready_count = sum(1 for s in AVAILABLE_STATION1_CAMERAS.keys() 

                                 if self.cameras_ready[f"station1_{s}"])

                total_count = len(AVAILABLE_STATION1_CAMERAS)

                return f"Tire {self.current_tire + 1}: Station 1 - {ready_count}/{total_count} cameras ready"

            elif not self.station2_complete and len(AVAILABLE_STATION2_CAMERAS) > 0:

                return f"Tire {self.current_tire + 1}: Station 1 done, waiting for Station 2 trigger"

            else:

                return f"Tire {self.current_tire + 1}: Complete"
 
# =========================================================

# SIMULATED CAMERA FOR MISSING HARDWARE

# =========================================================

def create_simulated_image(width, height):

    """Create dummy image for testing when camera is missing"""

    # Create a gradient pattern to indicate simulated image

    img = np.zeros((height, width), dtype=np.uint16)

    # Add a pattern to clearly show it's simulated

    for i in range(height):

        img[i, :] = (i % 65535)  # Gradient pattern

    # Add text overlay (will be visible when normalized)

    return img
 
class SimulatedCameraWorker:

    """Simulates a camera for testing when hardware is missing"""

    def __init__(self, serial, role, camera_config):

        self.serial = serial

        self.role = role

        self.camera_config = camera_config

        self.running = True

    def start(self):

        print(f"[SIMULATED] Camera {self.role} ({self.serial}) - Running in simulation mode")

    def stop(self):

        self.running = False

        print(f"[SIMULATED] Camera {self.role} ({self.serial}) - Stopped")
 
# =========================================================

# GLOBALS

# =========================================================

RUNNING = True

save_queue = queue.Queue(maxsize=SAVE_QUEUE_SIZE)

coordinator = TriggerCoordinator()
 
# =========================================================

# SAFE NODE SET

# =========================================================

def set_node(nodemap, name, value, required=False):

    try:

        node = nodemap.get_node(name)

        if node is None:

            if required:

                raise RuntimeError(f"Node {name} not found")

            print(f"  [SKIP] {name}: node not found")

            return False

        if hasattr(node, 'is_writable') and not node.is_writable:

            print(f"  [SKIP] {name}: not writable")

            return False

        node.value = value

        print(f"  [OK] {name}: {value}")

        return True

    except Exception as e:

        if required:

            raise RuntimeError(f"{name} ERROR: {e}")

        print(f"  [FAIL] {name}: {str(e)[:50]}")

        return False
 
# =========================================================

# BUFFER CONVERSION

# =========================================================

def convert_buffer(buffer):

    copied = BufferFactory.copy(buffer)

    try:

        width = copied.width

        height = copied.height

        total_bytes = len(copied.data)

        c_arr = (ctypes.c_ubyte * total_bytes).from_address(

            ctypes.addressof(copied.pbytes)

        )

        np_arr = np.ctypeslib.as_array(c_arr)

        bytes_per_pixel = total_bytes // (width * height)

        if bytes_per_pixel == 2:

            image = np_arr.view(np.uint16).reshape(height, width)

        else:

            image = np_arr.reshape(height, width)

        return image.copy()

    finally:

        BufferFactory.destroy(copied)
 
# =========================================================

# SAVE WORKER

# =========================================================

def save_worker():

    while RUNNING or not save_queue.empty():

        try:

            item = save_queue.get(timeout=1)

        except queue.Empty:

            continue

        if item is None:

            save_queue.task_done()

            break

        filename, image, is_simulated = item

        try:

            img_8bit = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)

            img_8bit = img_8bit.astype(np.uint8)

            # Add text overlay for simulated images

            if is_simulated:

                h, w = img_8bit.shape

                cv2.putText(img_8bit, "SIMULATED - CAMERA NOT AVAILABLE", 

                           (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 

                           0.7, 255, 2)

            cv2.imwrite(filename, img_8bit, [cv2.IMWRITE_PNG_COMPRESSION, PNG_COMPRESSION])

            sim_tag = " [SIMULATED]" if is_simulated else ""

            print(f"[SAVE]{sim_tag} {os.path.basename(filename)}")

        except Exception as e:

            print(f"[SAVE ERROR] {e}")

        finally:

            save_queue.task_done()
 
# =========================================================

# CAMERA WORKER (Handles missing cameras gracefully)

# =========================================================

def camera_worker(camera, camera_index, camera_config, is_simulated=False, simulated_serial=None, simulated_role=None):

    global RUNNING

    if is_simulated:

        # Handle simulated camera

        serial = simulated_serial

        camera_type = camera_config["name"]

        camera_role = simulated_role

        width = camera_config["width"]

        final_height = camera_config["final_height"]

        num_patches = camera_config["num_patches"]

        print(f"\n[CAM {camera_index}] [SIMULATED] {camera_type} SERIAL: {serial} | ROLE: {camera_role}")

        # Create folder

        serial_dir = os.path.join(SAVE_DIR, f"{camera_role}_{serial}")

        os.makedirs(serial_dir, exist_ok=True)

        tire_count = 0

        while RUNNING and tire_count < NUM_TIRES:

            if not coordinator.should_capture(serial):

                time.sleep(0.05)

                continue

            trigger_num = coordinator.get_trigger_number()

            print(f"\n[CAM {camera_index}] [SIMULATED] {camera_role} - Simulating capture for Trigger {trigger_num}...")

            # Create simulated full image

            full_img = create_simulated_image(width, final_height)

            # Simulate capture time

            time.sleep(0.5)

            # Save simulated image

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

            filename = os.path.join(serial_dir, f"{camera_role}_tire_{tire_count+1}_trigger_{trigger_num}_SIMULATED_{timestamp}.png")

            save_queue.put((filename, full_img, True))

            print(f"[CAM {camera_index}] [SIMULATED] {camera_role} ✓ Simulated capture complete")

            coordinator.mark_trigger_received(serial)

            if coordinator.is_cycle_complete():

                tire_count += 1

                coordinator.next_tire()

        return

    # =========================================================

    # REAL CAMERA WORKER (Original logic)

    # =========================================================

    nodemap = camera.nodemap

    serial = nodemap.get_node("DeviceSerialNumber").value

    camera_type = camera_config["name"]

    camera_role = AVAILABLE_CAMERAS.get(serial, "Unknown")

    width = camera_config["width"]

    patch_height = camera_config["patch_height"]

    final_height = camera_config["final_height"]

    num_patches = camera_config["num_patches"]

    print(f"\n[CAM {camera_index}] {camera_type} SERIAL: {serial} | ROLE: {camera_role}")

    # Create folder

    serial_dir = os.path.join(SAVE_DIR, f"{camera_role}_{serial}")

    os.makedirs(serial_dir, exist_ok=True)

    stream_started = False

    tire_count = 0

    try:

        camera.start_stream(NUM_STREAM_BUFFERS)

        stream_started = True

        print(f"[CAM {camera_index}] STREAM STARTED - Waiting for hardware trigger")

        while RUNNING and tire_count < NUM_TIRES:

            if not coordinator.should_capture(serial):

                time.sleep(0.05)

                continue

            trigger_num = coordinator.get_trigger_number()

            print(f"\n[CAM {camera_index}] [{camera_role}] Waiting for Trigger {trigger_num}...")

            full_img = np.zeros((final_height, width), dtype=np.uint16)

            frames_received = 0

            frames_expected = final_height

            capture_start_time = time.time()

            for patch_num in range(num_patches):

                patch_start_row = patch_num * patch_height

                current_row_in_patch = 0

                while current_row_in_patch < patch_height and RUNNING:

                    try:

                        buffer = camera.get_buffer(timeout=TRIGGER_TIMEOUT * 1000)

                    except Exception as e:

                        print(f"[CAM {camera_index}] Buffer timeout! {e}")

                        if "timeout" in str(e).lower():

                            continue

                        raise

                    try:

                        frame = convert_buffer(buffer)

                        h, w = frame.shape

                        remaining_in_patch = patch_height - current_row_in_patch

                        lines_to_copy = min(h, remaining_in_patch)

                        full_img[patch_start_row + current_row_in_patch:patch_start_row + current_row_in_patch + lines_to_copy, :] = frame[:lines_to_copy, :]

                        current_row_in_patch += lines_to_copy

                        frames_received += lines_to_copy

                        if current_row_in_patch % 1000 == 0 or current_row_in_patch == patch_height:

                            progress = (frames_received / frames_expected) * 100

                            print(f"  [CAM {camera_index}] [{camera_role}] Patch {patch_num+1}: {current_row_in_patch}/{patch_height} lines ({progress:.1f}%)")

                    finally:

                        camera.requeue_buffer(buffer)

            capture_end_time = time.time()

            if frames_received != frames_expected:

                print(f"[WARNING] {camera_role} frame loss: {frames_received}/{frames_expected}")

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

            filename = os.path.join(serial_dir, f"{camera_role}_tire_{tire_count+1}_trigger_{trigger_num}_{timestamp}.png")

            try:

                save_queue.put((filename, full_img, False), timeout=5)

                print(f"[CAM {camera_index}] [{camera_role}] ✓ Captured {frames_received} lines in {capture_end_time - capture_start_time:.2f} sec")

            except queue.Full:

                print(f"[ERROR] Save queue full for {camera_role}")

            coordinator.mark_trigger_received(serial)

            if coordinator.is_cycle_complete():

                tire_count += 1

                coordinator.next_tire()

    except Exception as e:

        print(f"[CAM {camera_index}] [{camera_role}] ERROR: {e}")

        import traceback

        traceback.print_exc()

        RUNNING = False

    finally:

        if stream_started:

            try:

                camera.stop_stream()

                print(f"[CAM {camera_index}] [{camera_role}] STREAM STOPPED")

            except Exception as e:

                print(f"[CAM {camera_index}] STOP ERROR: {e}")
 
# =========================================================

# STATUS MONITOR

# =========================================================

def status_monitor():

    last_progress = ""

    while RUNNING:

        progress = coordinator.get_progress()

        if progress != last_progress:

            print(f"\n[STATUS] {progress}")

            last_progress = progress

        if save_queue.qsize() > 50:

            print(f"[WARNING] Save queue: {save_queue.qsize()}/{SAVE_QUEUE_SIZE}")

        time.sleep(2)
 
# =========================================================

# GET CAMERA CONFIG

# =========================================================

FOUR_K_SERIALS = {"254901430", "254901432", "254901428"}

TWO_K_SERIALS = {"250500042", "220903275"}
 
def get_camera_config(serial):

    if serial in FOUR_K_SERIALS:

        return FOUR_K_CONFIG

    else:

        return TWO_K_CONFIG

def configure_real_camera(cam, serial, config):
    nodemap = cam.nodemap

    print(f"\n[CONFIG] Camera {serial} - {config['name']}")

    set_node(nodemap, "Width", config["width"])
    set_node(nodemap, "Height", config["patch_height"])
    set_node(nodemap, "PixelFormat", config["pixel_format"])

    set_node(nodemap, "ExposureAutoLimitAuto", "Off")
    set_node(nodemap, "ExposureTime", config["exposure_us"])
    set_node(nodemap, "Gain", config["gain_db"])

    if config.get("line_rate_enable") is True:
        set_node(nodemap, "AcquisitionLineRateEnable", True)
        set_node(nodemap, "AcquisitionLineRate", config["line_rate"])
    else:
        print(f"  [SKIP] AcquisitionLineRateEnable / AcquisitionLineRate for {serial}")

    set_node(nodemap, "AcquisitionMode", "Continuous")

    set_node(nodemap, "GevSCPSPacketSize", config["packet_size"])
    set_node(nodemap, "GevSCPD", config["packet_delay"])

    if MODE == "AUTO":
        set_node(nodemap, "TriggerMode", "Off")
        set_node(nodemap, "TriggerSelector", "FrameStart")
        set_node(nodemap, "TriggerSource", "Line0")
        set_node(nodemap, "TriggerActivation", "RisingEdge")
        set_node(nodemap, "TriggerMode", "On")
    else:
        set_node(nodemap, "TriggerMode", "Off")

    print("[CONFIG DONE]")
 
# =========================================================

# MAIN

# =========================================================

def main():

    global RUNNING

    os.makedirs(SAVE_DIR, exist_ok=True)

    print("\n" + "="*70)

    print("MULTI-STATION PRODUCTION SYSTEM - GRACEFUL FAILURE HANDLING")

    print("="*70)

    print(f"MODE: {MODE}")

    print(f"MISSING CAMERA HANDLING: {MISSING_CAMERA_HANDLING}")

    print("\nCAMERA STATUS:")

    # Print camera availability

    for serial, role in STATION1_CAMERAS.items():

        available = CAMERA_AVAILABLE.get(serial, False)

        status = "✓ AVAILABLE" if available else "✗ UNAVAILABLE"

        reason = ""

        if serial == "250500042" and not available:

            reason = "(IO config issue)"

        elif serial == "220903275" and not available:

            reason = "(Data cable issue)"

        print(f"  {role:12} {serial:12} : {status} {reason}")

    for serial, role in STATION2_CAMERAS.items():

        if role not in [r for r in STATION1_CAMERAS.values()]:

            available = CAMERA_AVAILABLE.get(serial, False)

            status = "✓ AVAILABLE" if available else "✗ UNAVAILABLE"

            reason = "(Data cable issue)" if serial == "220903275" and not available else ""

            print(f"  {role:12} {serial:12} : {status} {reason}")

    print(f"\nAvailable Cameras: {len(AVAILABLE_CAMERAS)}/{len(ALL_CAMERAS)}")

    print(f"Station 1: {len(AVAILABLE_STATION1_CAMERAS)}/{len(STATION1_CAMERAS)} cameras")

    print(f"Station 2: {len(AVAILABLE_STATION2_CAMERAS)}/{len(STATION2_CAMERAS)} cameras")

    print(f"Total Tires: {NUM_TIRES}")

    print("="*70)

    if len(AVAILABLE_CAMERAS) == 0:

        print("\nERROR: No cameras available! Please check camera availability settings.")

        return

    # Discover real cameras

    print("\nSearching for physical cameras...")

    devices = system.create_device() if MISSING_CAMERA_HANDLING != "SIMULATE" else []

    cameras_by_serial = {}

    for dev in devices:

        serial = dev.nodemap.get_node("DeviceSerialNumber").value

        cameras_by_serial[serial] = dev

        print(f"  Found physical camera: {serial}")

    # Prepare camera threads

    threads = []

    camera_index = 0

    # Process available cameras

    for serial in CAMERA_SERIALS:

        if not CAMERA_AVAILABLE.get(serial, False):

            if MISSING_CAMERA_HANDLING == "SIMULATE":

                # Create simulated worker

                role = ALL_CAMERAS.get(serial, "Unknown")

                config = get_camera_config(serial)

                t = threading.Thread(

                    target=camera_worker,

                    args=(None, camera_index, config, True, serial, role),

                    daemon=True

                )

                threads.append(t)

                print(f"  [SIMULATED] Camera {role} ({serial}) - Running in simulation mode")

                camera_index += 1

            else:

                print(f"  [SKIPPED] Camera {serial} - Not available ({MISSING_CAMERA_HANDLING} mode)")

            continue

        # Real available camera

        if serial not in cameras_by_serial:

            print(f"  [ERROR] Camera {serial} marked available but not found physically!")

            if MISSING_CAMERA_HANDLING == "SIMULATE":

                role = AVAILABLE_CAMERAS.get(serial, "Unknown")

                config = get_camera_config(serial)

                t = threading.Thread(

                    target=camera_worker,

                    args=(None, camera_index, config, True, serial, role),

                    daemon=True

                )

                threads.append(t)

                print(f"  [SIMULATED] Fallback for {serial}")

            continue

        # Configure real camera

        cam = cameras_by_serial[serial]

        config = get_camera_config(serial)

        # Configure trigger settings

        configure_real_camera(cam, serial, config)

        t = threading.Thread(

            target=camera_worker,

            args=(cam, camera_index, config, False, None, None),

            daemon=True

        )

        threads.append(t)

        print(f"  [REAL] Camera {AVAILABLE_CAMERAS[serial]} ({serial}) - Configured for hardware trigger")

        camera_index += 1

    if len(threads) == 0:

        print("\nERROR: No camera threads created!")

        return

    # Start save thread

    saver_thread = threading.Thread(target=save_worker, daemon=True)

    saver_thread.start()

    # Start status monitor

    monitor_thread = threading.Thread(target=status_monitor, daemon=True)

    monitor_thread.start()

    # Start all camera threads

    for t in threads:

        t.start()

    print("\n" + "="*70)

    print("SYSTEM READY - WAITING FOR PLC TRIGGERS")

    print("="*70)

    print(f"Active cameras: {len([t for t in threads if t.is_alive()])}")

    print("PLC should send:")
    print("  - Pulse 1 on Line0 → Station 1 cameras")

    if len(AVAILABLE_STATION2_CAMERAS) > 0:
        print("  - Pulse 2 on Line0 → Station 2 cameras")
    else:
        print("  - Station 2 skipped because no camera is available")

    print("\nMissing cameras are being handled gracefully")

    print("Press Ctrl+C to stop")

    print("="*70)

    # Wait for completion

    try:

        while RUNNING and coordinator.current_tire < NUM_TIRES:

            time.sleep(1)

    except KeyboardInterrupt:

        print("\n\nUser interrupted")

        RUNNING = False

    # Wait for threads to finish

    for t in threads:

        t.join(timeout=10)

    # Cleanup

    save_queue.join()

    save_queue.put(None)

    saver_thread.join(timeout=5)

    if devices:

        system.destroy_device()

    print("\n" + "="*70)

    print("PRODUCTION COMPLETE")

    print("="*70)

    print(f"Tires Completed: {coordinator.current_tire}/{NUM_TIRES}")

    print(f"Active Cameras: {len(AVAILABLE_CAMERAS)}")

    print(f"Simulated Cameras: {sum(1 for s in CAMERA_SERIALS if not CAMERA_AVAILABLE.get(s, False))}")

    print(f"Output Directory: {SAVE_DIR}")

    print("="*70)
 
if __name__ == "__main__":

    signal.signal(signal.SIGINT, lambda sig, frame: setattr(__import__('__main__'), 'RUNNING', False))

    main()
 