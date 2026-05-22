import os
import time
import ctypes
import threading
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import numpy as np
import snap7
from arena_api.system import system
from arena_api.buffer import BufferFactory


# ================= PLC SETTINGS =================
PLC_IP = "192.168.10.1"
PLC_RACK = 0
PLC_SLOT = 1

TRIGGER_DB = 74
TRIGGER_BYTE = 0
TRIGGER_BIT = 3          # DB74.DBX0.3

RESET_TAG_AFTER_CAPTURE = False
POLL_DELAY_SEC = 0.05


# ================= CAMERA SETTINGS =================
SAVE_DIR = r"C:\Users\PrajwalSridhar\Desktop\Apollo_share\Auto8"

WIDTH = 4096

CHUNK_HEIGHT = 14000
NUM_CHUNKS = 3
FINAL_HEIGHT = CHUNK_HEIGHT * NUM_CHUNKS   # 42000

HEIGHT = CHUNK_HEIGHT

PIXEL_FORMAT = "Mono16"

EXPOSURE_US = 120.0
GAIN_DB = 24.0
LINE_RATE = 8169.178266

PACKET_SIZE = 9000
PACKET_DELAY = 1000
STREAM_BUFFERS = 8

SOFTWARE_TRIGGER_SELECTOR = "AcquisitionStart"

MAX_PARALLEL_CAMERAS = 4


# ==================================================
def get_bool(db_data, byte_index, bit_index):
    return (db_data[byte_index] & (1 << bit_index)) != 0


def set_bool(plc, db_number, byte_index, bit_index, value):
    data = plc.db_read(db_number, byte_index, 1)

    if value:
        data[0] |= (1 << bit_index)
    else:
        data[0] &= ~(1 << bit_index)

    plc.db_write(db_number, byte_index, data)


def connect_plc():
    plc = snap7.client.Client()
    plc.connect(PLC_IP, PLC_RACK, PLC_SLOT)

    if not plc.get_connected():
        raise RuntimeError("PLC not connected")

    print(f"[PLC] Connected: {PLC_IP}")
    return plc


def wait_for_plc_tag_true(plc):
    print(f"[PLC] Waiting for DB{TRIGGER_DB}.DBX{TRIGGER_BYTE}.{TRIGGER_BIT} = TRUE")

    while True:
        data = plc.db_read(TRIGGER_DB, TRIGGER_BYTE, 1)
        trigger = get_bool(data, 0, TRIGGER_BIT)

        if trigger:
            print("[PLC] Trigger tag TRUE. Starting parallel capture...")
            return

        time.sleep(POLL_DELAY_SEC)


def wait_for_plc_tag_false(plc):
    print("[PLC] Waiting for trigger tag to become FALSE...")

    while True:
        data = plc.db_read(TRIGGER_DB, TRIGGER_BYTE, 1)

        if not get_bool(data, 0, TRIGGER_BIT):
            print("[PLC] Trigger tag FALSE. Ready for next capture.")
            return

        time.sleep(POLL_DELAY_SEC)


def set_node(nodemap, name, value):
    try:
        node = nodemap.get_node(name)

        if node and node.is_writable:
            node.value = value
            print(f"[SET OK] {name}: {node.value}")
            return True
        else:
            print(f"[SKIP] {name}: not writable / not found")
            return False

    except Exception as e:
        print(f"[SET FAIL] {name}: {e}")
        return False


def read_node(nodemap, name, default="-"):
    try:
        node = nodemap.get_node(name)

        if node and node.is_readable:
            return node.value

    except Exception:
        pass

    return default


def convert_buffer_to_numpy(buffer):
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
            img = np_arr.view(np.uint16).reshape(height, width)
        else:
            img = np_arr.reshape(height, width)

        return img.copy()

    finally:
        BufferFactory.destroy(copied)


def configure_camera(cam, index):
    nodemap = cam.nodemap

    print("")
    print(f"========== CONFIG CAMERA {index} ==========")

    set_node(nodemap, "Width", WIDTH)
    set_node(nodemap, "Height", HEIGHT)
    set_node(nodemap, "PixelFormat", PIXEL_FORMAT)

    set_node(nodemap, "ExposureAutoLimitAuto", "Off")
    set_node(nodemap, "ExposureTime", EXPOSURE_US)

    set_node(nodemap, "Gain", GAIN_DB)

    set_node(nodemap, "AcquisitionLineRateEnable", True)
    set_node(nodemap, "AcquisitionLineRate", LINE_RATE)

    set_node(nodemap, "AcquisitionMode", "Continuous")

    set_node(nodemap, "GevSCPSPacketSize", PACKET_SIZE)
    set_node(nodemap, "GevSCPD", PACKET_DELAY)

    # Software AcquisitionStart trigger
    set_node(nodemap, "TriggerMode", "Off")
    set_node(nodemap, "TriggerSelector", SOFTWARE_TRIGGER_SELECTOR)
    set_node(nodemap, "TriggerSource", "Software")
    set_node(nodemap, "TriggerMode", "On")

    serial = read_node(nodemap, "DeviceSerialNumber", f"CAM_{index}")

    print("------ FINAL SETTINGS ------")
    for n in [
        "DeviceSerialNumber",
        "Width",
        "Height",
        "PixelFormat",
        "ExposureTime",
        "Gain",
        "AcquisitionLineRate",
        "TriggerMode",
        "TriggerSelector",
        "TriggerSource",
        "GevSCPSPacketSize",
        "GevSCPD",
    ]:
        print(f"{n}: {read_node(nodemap, n)}")

    return serial


def execute_software_trigger(cam):
    nodemap = cam.nodemap

    try:
        trigger_node = nodemap.get_node("TriggerSoftware")
        trigger_node.execute()
        print("[CAM] Software AcquisitionStart trigger executed")

    except Exception as e:
        raise RuntimeError(f"Software trigger failed: {e}")


def capture_42000_stitched_image(cam, serial, start_barrier):
    serial_dir = os.path.join(SAVE_DIR, str(serial))
    os.makedirs(serial_dir, exist_ok=True)

    chunks = []

    print(f"[CAM {serial}] Starting stream...")
    cam.start_stream(STREAM_BUFFERS)

    try:
        # All cameras wait here, then trigger together
        start_barrier.wait()

        execute_software_trigger(cam)

        print(f"[CAM {serial}] Capturing {NUM_CHUNKS} continuous patches...")

        for i in range(NUM_CHUNKS):
            buffer = cam.get_buffer(timeout=20000)

            try:
                img = convert_buffer_to_numpy(buffer)

                print(f"[CAM {serial}] Patch {i + 1}/{NUM_CHUNKS}: {img.shape}")

                if img.shape[0] != CHUNK_HEIGHT:
                    print(
                        f"[WARN] CAM {serial}: patch height {img.shape[0]}, "
                        f"expected {CHUNK_HEIGHT}"
                    )

                chunks.append(img.copy())

            finally:
                cam.requeue_buffer(buffer)

        full_img = np.vstack(chunks)

        print(f"[CAM {serial}] Stitched image shape: {full_img.shape}")

        if full_img.shape[0] != FINAL_HEIGHT:
            print(
                f"[WARN] CAM {serial}: stitched height {full_img.shape[0]}, "
                f"expected {FINAL_HEIGHT}"
            )

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        save_path = os.path.join(
            serial_dir,
            f"cam_{serial}_{timestamp}_stitched_42000.png"
        )

        img_8bit = cv2.normalize(full_img, None, 0, 255, cv2.NORM_MINMAX)
        img_8bit = img_8bit.astype(np.uint8)

        ok = cv2.imwrite(save_path, img_8bit, [cv2.IMWRITE_PNG_COMPRESSION, 0])

        if not ok:
            raise RuntimeError(f"Failed to save image: {save_path}")

        print(f"[SAVE OK] {save_path}")
        return save_path

    finally:
        try:
            cam.stop_stream()
            print(f"[CAM {serial}] Stream stopped")
        except Exception as e:
            print(f"[CAM {serial}] stop_stream warning: {e}")


def capture_all_cameras_parallel(camera_info):
    selected_cameras = camera_info[:MAX_PARALLEL_CAMERAS]

    if len(selected_cameras) == 0:
        raise RuntimeError("No cameras available for capture")

    print("")
    print("========== PARALLEL CAPTURE START ==========")
    print(f"[INFO] Capturing {len(selected_cameras)} cameras in parallel")

    start_barrier = threading.Barrier(len(selected_cameras))

    saved_paths = []

    with ThreadPoolExecutor(max_workers=len(selected_cameras)) as executor:
        futures = []

        for cam, serial in selected_cameras:
            future = executor.submit(
                capture_42000_stitched_image,
                cam,
                serial,
                start_barrier
            )
            futures.append(future)

        for future in as_completed(futures):
            try:
                path = future.result()
                saved_paths.append(path)
                print(f"[DONE] {path}")

            except Exception as e:
                print(f"[CAPTURE ERROR] {e}")

    print("========== PARALLEL CAPTURE DONE ==========")
    print("")

    return saved_paths


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    plc = None
    devices = []

    try:
        plc = connect_plc()

        devices = system.create_device()

        if len(devices) == 0:
            raise RuntimeError("No Lucid cameras found")

        print(f"[CAM] Detected cameras: {len(devices)}")

        camera_info = []

        for idx, cam in enumerate(devices):
            serial = configure_camera(cam, idx)
            camera_info.append((cam, serial))

        while True:
            wait_for_plc_tag_true(plc)

            saved_paths = capture_all_cameras_parallel(camera_info)

            print("[RESULT] Saved files:")
            for p in saved_paths:
                print(p)

            if RESET_TAG_AFTER_CAPTURE:
                set_bool(plc, TRIGGER_DB, TRIGGER_BYTE, TRIGGER_BIT, False)
                print("[PLC] Trigger tag reset to FALSE")

            # This prevents repeated capture if PLC bit stays TRUE
            wait_for_plc_tag_false(plc)

    except KeyboardInterrupt:
        print("Stopped by user")

    except Exception as e:
        print(f"[ERROR] {e}")

    finally:
        try:
            system.destroy_device()
        except Exception:
            pass

        if plc is not None:
            try:
                plc.disconnect()
                print("[PLC] Disconnected")
            except Exception:
                pass


if __name__ == "__main__":
    main()