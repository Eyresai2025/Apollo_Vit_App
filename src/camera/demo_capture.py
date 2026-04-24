import os
import cv2
import time
import ctypes
import numpy as np
import warnings

from arena_api.system import system  # type: ignore
from arena_api.buffer import BufferFactory  # type: ignore

warnings.filterwarnings("ignore")


# Per-camera configuration
CAMERA_CONFIGS = {
    "250500042": {
        "ExposureTime": 250.0,
        "PixelFormat": "Mono16",
    },
    "220903275": {
        "ExposureTime": 400.0,
        "PixelFormat": "Mono16",
    },
}

DEFAULT_HEIGHT = 16384
DEFAULT_WIDTH = 2048
DEFAULT_GAIN = 24.0
DEFAULT_PIXEL_FORMAT = "Mono16"


def create_all_connected_devices(expected_cameras=2, retries=5, delay=1):
    for i in range(retries):
        devices = system.create_device()
        if devices:
            print(f"{len(devices)} camera(s) connected.")
            if len(devices) < expected_cameras:
                print(f"⚠ Expected {expected_cameras} cameras, but found {len(devices)}")
            return devices

        print(f"Retry {i + 1}/{retries}: No cameras found. Retrying in {delay}s...")
        time.sleep(delay)

    raise Exception("❌ No cameras found after retries.")


def setup_software_trigger(device):
    """
    Camera setup for software trigger with per-camera exposure settings.
    """
    nodemap = device.nodemap

    serial = str(nodemap.get_node("DeviceSerialNumber").value)
    model = nodemap.get_node("DeviceModelName").value

    cfg = CAMERA_CONFIGS.get(serial, {})
    exposure_time = float(cfg.get("ExposureTime", 350.0))
    pixel_format = cfg.get("PixelFormat", DEFAULT_PIXEL_FORMAT)

    # Stop trigger before changing trigger params
    nodemap.get_node("TriggerMode").value = "Off"

    # Image settings
    nodemap.get_node("Height").value = DEFAULT_HEIGHT
    nodemap.get_node("Width").value = DEFAULT_WIDTH
    nodemap.get_node("ExposureTime").value = exposure_time
    nodemap.get_node("Gain").value = DEFAULT_GAIN
    nodemap.get_node("PixelFormat").value = pixel_format

    # Software trigger settings
    nodemap.get_node("TriggerSelector").value = "FrameStart"
    nodemap.get_node("TriggerSource").value = "Software"
    nodemap.get_node("TriggerMode").value = "On"
    nodemap.get_node("AcquisitionMode").value = "Continuous"

    # Stream settings
    tl_stream_nodemap = device.tl_stream_nodemap
    tl_stream_nodemap["StreamAutoNegotiatePacketSize"].value = True
    tl_stream_nodemap["StreamPacketResendEnable"].value = True

    print(
        f"Camera ready for software trigger: {model} ({serial}) | "
        f"ExposureTime={exposure_time} | PixelFormat={pixel_format}"
    )


def convert_buffer(buffer):
    """
    Convert Arena buffer to NumPy image array.
    Supports Mono8 and Mono16 correctly.
    """
    item = BufferFactory.copy(buffer)
    bytes_per_pixel = int(len(item.data) / (item.width * item.height))

    if bytes_per_pixel == 1:
        array = (ctypes.c_ubyte * (item.width * item.height)).from_address(
            ctypes.addressof(item.pbytes)
        )
        img = np.ctypeslib.as_array(array).reshape(item.height, item.width).copy()

    elif bytes_per_pixel == 2:
        array = (ctypes.c_uint16 * (item.width * item.height)).from_address(
            ctypes.addressof(item.pbytes)
        )
        img = np.ctypeslib.as_array(array).reshape(item.height, item.width).copy()

    else:
        raise ValueError(
            f"Unsupported bytes_per_pixel={bytes_per_pixel}. "
            f"Expected Mono8 or Mono16."
        )

    return img


def execute_software_trigger(device):
    """
    Fire one software trigger.
    """
    nodemap = device.nodemap
    nodemap.get_node("TriggerSoftware").execute()


def get_one_frame(device):
    """
    Read one frame after trigger.
    """
    buffer = device.get_buffer()
    try:
        img = convert_buffer(buffer)
    finally:
        device.requeue_buffer(buffer)
    return img


def capture_images_from_all_cameras_software(
    save_root,
    expected_cameras=2,
    images_per_camera=5,
    delay_between_triggers=0.05,
    progress_callback=None
):
    """
    Connect cameras, arm them, use software trigger, capture N images per camera,
    and save as:

        save_root/<serial>/<serial>_001.png
        save_root/<serial>/<serial>_002.png
        ...

    progress_callback(current, total, message, serial, save_path)
        current   -> current completed image count
        total     -> total images to capture
        message   -> status text
        serial    -> current camera serial
        save_path -> just-saved image path

    Returns:
        {
            serial1: [img_path1, img_path2, ...],
            serial2: [img_path1, img_path2, ...]
        }
    """
    os.makedirs(save_root, exist_ok=True)

    devices = create_all_connected_devices(expected_cameras=expected_cameras)
    results = {}

    try:
        # Setup all cameras
        for device in devices:
            setup_software_trigger(device)

        # Start stream for all cameras
        for device in devices:
            device.start_stream()

        print("All cameras armed. Starting software-trigger capture...")

        # Create serial folders
        for device in devices:
            serial = str(device.nodemap.get_node("DeviceSerialNumber").value)
            serial_dir = os.path.join(save_root, serial)
            os.makedirs(serial_dir, exist_ok=True)
            results[serial] = []

        total_images = len(devices) * images_per_camera
        completed = 0

        # Capture N images per camera
        for shot_idx in range(images_per_camera):
            print(f"\n[Capture] Shot {shot_idx + 1}/{images_per_camera}")

            # Trigger all cameras
            for device in devices:
                execute_software_trigger(device)

            # Small delay for frame readiness
            time.sleep(delay_between_triggers)

            # Read and save one frame from each camera
            for device in devices:
                serial = str(device.nodemap.get_node("DeviceSerialNumber").value)
                img = get_one_frame(device)

                # For Mono16 preserve data using PNG
                file_name = f"{serial}_{shot_idx + 1:03d}.png"
                save_path = os.path.join(save_root, serial, file_name)

                ok = cv2.imwrite(save_path, img)
                if not ok:
                    raise Exception(f"Failed to save image: {save_path}")

                results[serial].append(save_path)
                completed += 1

                print(f"Saved: {save_path} | dtype={img.dtype} | shape={img.shape}")

                # send live progress update to GUI
                if progress_callback is not None:
                    progress_callback(
                        completed,
                        total_images,
                        f"Capturing images... {completed}/{total_images}",
                        serial,
                        save_path
                    )

    finally:
        for device in devices:
            try:
                device.stop_stream()
            except Exception as e:
                print(f"Error stopping stream: {e}")

        try:
            system.destroy_device()
        except Exception as e:
            print(f"Error destroying devices: {e}")

    return results