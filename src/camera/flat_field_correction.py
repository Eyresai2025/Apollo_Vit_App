"""
Flat Field Correction (FFC) - Permanent Camera Save + Normal Capture
for Lucid 2K / 4K Line Scan Cameras using Arena SDK Python API

Modes:
    1) RUN_MODE = "CALIBRATE_AND_SAVE_FFC"
       - Capture flat image
       - Compute column gain values
       - Preview software corrected image
       - Write gain table into camera FFC memory
       - Execute FlatFieldCorrectionSave
       - Optionally save UserSet1
       - Optionally capture one verification image with camera-side FFC enabled

    2) RUN_MODE = "CAPTURE_WITH_SAVED_FFC"
       - Select saved FFC slot
       - Enable camera-side FFC
       - Capture image directly from camera

Requirements:
    pip install opencv-python==4.10.0.84
    pip install numpy==2.0.0
    pip install matplotlib==3.9.1

Also install:
    Arena SDK + arena_api Python package

Run:
    python flat_field_correction_permanent_camera_ffc.py
"""

import os
import cv2
import time
import ctypes
import datetime
import numpy as np
import matplotlib.pyplot as plt

from arena_api import enums
from arena_api.system import system
from arena_api.enums import PixelFormat
from arena_api.buffer import BufferFactory
from arena_api.__future__.save import Writer


# ============================================================
# RUN MODE
# ============================================================

# First time only: generate FFC table and save permanently into camera.
RUN_MODE = "CALIBRATE_AND_SAVE_FFC"

# After FFC is saved once, comment above and use this for normal capture.
# RUN_MODE = "CAPTURE_WITH_SAVED_FFC"
RUN_FFC_WRITE_TEST = True
# ============================================================
# OUTPUT / DISPLAY OPTIONS
# ============================================================

SAVE_DIR = "images"
os.makedirs(SAVE_DIR, exist_ok=True)

SAVE_PLOTS = True
SHOW_IMAGES = True
CAPTURE_VERIFY_AFTER_SAVE = False


# ============================================================
# CAMERA / ACQUISITION CONFIGURATION
# ============================================================

# Same 16000 height capture for both 2K and 4K.
LINESCAN_HEIGHT = 16000

# Use longer timeout for 16000-height line-scan images.
GET_BUFFER_TIMEOUT = 30000

PIXEL_FORMAT = PixelFormat.Mono16

# FFC slot shown in ArenaView as "Flat Field Correction 1".
FFC_SELECTOR = "FlatFieldCorrection1"

# Save current camera settings to UserSet1 after FFC save.
SAVE_USER_SET_AFTER_FFC = True
USER_SET_NAME = "UserSet1"

# Current gain clamp requested by you.
GAIN_RANGE_MIN = 1.0
GAIN_RANGE_MAX = 15.99

# Your current method used max column value as target.
# If image is getting over-bright/saturated, try "PERCENTILE_95" later.
GAIN_TARGET_MODE = "PERCENTILE_95"          # options: "MAX", "MEAN", "PERCENTILE_95"

# Camera-wise settings. Adjust here only if production settings change.
CAMERA_CONFIGS = {
    "TRI02KA-M": {                 # 2K camera
        "width": 2048,
        "height": LINESCAN_HEIGHT,
        "exposure_us": 120.0,
        "gain": 12.0,
        "line_rate_hz": None,      # keep None if 2K line-rate should not be forced
    },
    "TRT04KG-M": {                 # 4K camera
        "width": 4096,
        "height": LINESCAN_HEIGHT,
        "exposure_us": 122.0,
        "gain": 12.0,
        "line_rate_hz": 8169.0,
    },
}

SUPPORTED_MODELS = list(CAMERA_CONFIGS.keys())

TAB1 = "  "


# ============================================================
# BASIC HELPERS
# ============================================================

def generate_timestamp():
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def safe_stop_stream(device):
    try:
        device.stop_stream()
        print("[OK] Stream stopped")
    except Exception as e:
        print(f"[WARN] Could not stop stream: {e}")


def save_image(buffer, filename):
    writer = Writer()
    writer.pattern = filename
    writer.save(buffer)


def save_plot(fig, filename, title=None):
    if not SAVE_PLOTS:
        return

    if title:
        fig.suptitle(title)

    try:
        fig.tight_layout()
    except Exception:
        pass

    plt.grid(True)
    plt.savefig(filename, dpi=300, bbox_inches="tight")

    if SHOW_IMAGES:
        plt.show()
    else:
        plt.close(fig)


def set_node_if_exists(device, node_name, value):
    try:
        device.nodemap[node_name].value = value
        print(f"[OK] {node_name} set to {value}")
        return True
    except Exception as e:
        print(f"[WARN] Could not set {node_name}: {e}")
        return False


def execute_node_if_exists(device, node_name):
    try:
        device.nodemap[node_name].execute()
        print(f"[OK] {node_name} executed")
        return True
    except Exception as e:
        print(f"[WARN] Could not execute {node_name}: {e}")
        return False


def read_node_if_exists(device, node_name):
    try:
        value = device.nodemap[node_name].value
        print(f"[READ] {node_name}: {value}")
        return value
    except Exception as e:
        print(f"[WARN] Could not read {node_name}: {e}")
        return None


# ============================================================
# DEVICE CONNECTION
# ============================================================

def create_devices_with_tries():
    tries = 0
    tries_max = 6
    sleep_time_secs = 10

    while tries < tries_max:
        devices = system.create_device()

        if not devices:
            print(
                f"{TAB1}Try {tries + 1} of {tries_max}: "
                f"waiting for {sleep_time_secs} secs..."
            )

            for sec_count in range(sleep_time_secs):
                time.sleep(1)
                print(
                    f"{TAB1}{sec_count + 1} seconds passed "
                    + "." * sec_count,
                    end="\r"
                )

            tries += 1
        else:
            print(f"{TAB1}Created {len(devices)} device(s)")
            return devices

    raise Exception("No device found!")


# ============================================================
# CAMERA CONFIGURATION
# ============================================================

def print_camera_info(device):
    print("\n========== CAMERA INFO ==========" )

    read_node_if_exists(device, "DeviceModelName")
    read_node_if_exists(device, "DeviceSerialNumber")

    try:
        print(f"[READ] Width Max : {device.nodemap['Width'].max}")
    except Exception as e:
        print(f"[WARN] Could not read Width.max: {e}")

    try:
        print(f"[READ] Height Max: {device.nodemap['Height'].max}")
    except Exception as e:
        print(f"[WARN] Could not read Height.max: {e}")

    read_node_if_exists(device, "ExposureTime")
    read_node_if_exists(device, "Gain")
    read_node_if_exists(device, "AcquisitionLineRate")

    print("=================================\n")


def configure_linescan_camera(device, model_name):
    print("\n========== CONFIGURING CAMERA ==========" )

    if model_name not in CAMERA_CONFIGS:
        raise Exception(f"Unsupported camera model: {model_name}")

    cfg = CAMERA_CONFIGS[model_name]

    width = int(cfg["width"])
    height = int(cfg["height"])
    exposure_us = float(cfg["exposure_us"])
    gain = float(cfg["gain"])
    line_rate_hz = cfg.get("line_rate_hz")

    # Width / Height
    device.nodemap["Width"].value = width
    print(f"[OK] Width set: {width}")

    device.nodemap["Height"].value = height
    print(f"[OK] Height set: {height}")

    # Pixel Format
    device.nodemap["PixelFormat"].value = PIXEL_FORMAT
    print("[OK] PixelFormat set: Mono16")

    # Acquisition Mode
    device.nodemap["AcquisitionMode"].value = "Continuous"
    print("[OK] AcquisitionMode set: Continuous")

    # Trigger Mode OFF for this FFC test / free-run capture.
    device.nodemap["TriggerMode"].value = "Off"
    print("[OK] TriggerMode set: Off")

    # Gain
    set_node_if_exists(device, "Gain", gain)

    # Line Rate: only force for 4K or where configured.
    if line_rate_hz is not None:
        set_node_if_exists(device, "AcquisitionLineRateEnable", True)
        set_node_if_exists(device, "AcquisitionLineRate", float(line_rate_hz))
    else:
        print("[INFO] Line rate not forced for this camera model")

    # Exposure
    device.nodemap["ExposureTime"].value = exposure_us
    print(f"[OK] ExposureTime set: {exposure_us}")

    # Stream Settings
    tl_stream_nodemap = device.tl_stream_nodemap

    try:
        tl_stream_nodemap["StreamAutoNegotiatePacketSize"].value = True
        print("[OK] StreamAutoNegotiatePacketSize set: True")
    except Exception as e:
        print(f"[WARN] Could not set StreamAutoNegotiatePacketSize: {e}")

    try:
        tl_stream_nodemap["StreamPacketResendEnable"].value = True
        print("[OK] StreamPacketResendEnable set: True")
    except Exception as e:
        print(f"[WARN] Could not set StreamPacketResendEnable: {e}")

    print("\n===== CAMERA SETTINGS READBACK =====")
    read_node_if_exists(device, "Width")
    read_node_if_exists(device, "Height")
    read_node_if_exists(device, "PixelFormat")
    read_node_if_exists(device, "ExposureTime")
    read_node_if_exists(device, "Gain")
    read_node_if_exists(device, "AcquisitionLineRate")

    print("\n========== CONFIG SUCCESS ==========\n")


# ============================================================
# CAMERA FFC HELPERS
# ============================================================

def select_ffc_slot(device):
    """
    Select FlatFieldCorrection1.
    ArenaView may display it as "Flat Field Correction 1",
    while Python enum may accept "FlatFieldCorrection1".
    """
    selector_values = [
        FFC_SELECTOR,
        "Flat Field Correction 1",
    ]

    for value in selector_values:
        try:
            device.nodemap["FlatFieldCorrectionSelector"].value = value
            print(f"[OK] FlatFieldCorrectionSelector set to {value}")
            return True
        except Exception as e:
            print(f"[WARN] Selector value failed: {value} | {e}")

    print("[ERROR] Could not select FFC slot")
    return False


def print_ffc_state(device):
    print("\n========== CAMERA FFC STATE ==========")
    read_node_if_exists(device, "FlatFieldCorrectionEnable")
    read_node_if_exists(device, "FlatFieldCorrectionSelector")
    read_node_if_exists(device, "FlatFieldCorrectionColumnIndex")
    read_node_if_exists(device, "FlatFieldCorrectionRowIndex")
    read_node_if_exists(device, "FlatFieldCorrectionGain")
    read_node_if_exists(device, "FlatFieldCorrectionOffset")
    print("======================================\n")


def clear_existing_camera_ffc(device):
    print("\n========== CLEARING OLD CAMERA FFC ==========")
    select_ffc_slot(device)
    execute_node_if_exists(device, "FlatFieldCorrectionClearAll")
    print("========== OLD CAMERA FFC CLEARED ==========\n")


def enable_camera_ffc(device, enable=True):
    print("\n========== SETTING CAMERA FFC ENABLE ==========")
    select_ffc_slot(device)
    set_node_if_exists(device, "FlatFieldCorrectionEnable", bool(enable))
    read_node_if_exists(device, "FlatFieldCorrectionEnable")
    print(f"Camera FFC Enable requested: {enable}")
    print("==============================================\n")

def test_ffc_write_access(device):
    print("\n========== TESTING FFC WRITE ACCESS ==========\n")

    test_nodes = [
        "FlatFieldCorrectionColumnIndex",
        "FlatFieldCorrectionRowIndex",
        "FlatFieldCorrectionGain",
        "FlatFieldCorrectionOffset",
        "FlatFieldCorrectionSave",
        "FlatFieldCorrectionClearAll",
    ]

    for node_name in test_nodes:
        try:
            node = device.nodemap[node_name]
            print(f"[FOUND] {node_name}")

            try:
                value = node.value
                print(f"       Current value: {value}")
            except Exception as e:
                print(f"       Value read failed: {e}")

        except Exception as e:
            print(f"[NOT FOUND / ERROR] {node_name}: {e}")

    print("\n========== TRYING SMALL FFC WRITE TEST ==========\n")

    try:
        device.nodemap["FlatFieldCorrectionColumnIndex"].value = 0
        print("[OK] ColumnIndex set to 0")
    except Exception as e:
        print(f"[FAIL] ColumnIndex write failed: {e}")

    try:
        device.nodemap["FlatFieldCorrectionRowIndex"].value = 0
        print("[OK] RowIndex set to 0")
    except Exception as e:
        print(f"[FAIL] RowIndex write failed: {e}")

    try:
        device.nodemap["FlatFieldCorrectionGain"].value = 1.0
        print("[OK] Gain set to 1.0")
    except Exception as e:
        print(f"[FAIL] Gain write failed: {e}")

    try:
        device.nodemap["FlatFieldCorrectionOffset"].value = 0.0
        print("[OK] Offset set to 0.0")
    except Exception as e:
        print(f"[FAIL] Offset write failed: {e}")

    print("\n===============================================\n")

def write_ffc_gain_table_to_camera(device, gain_values):
    """
    Writes Python-computed gain values into camera FFC memory.

    For line-scan camera:
    - One gain value per column
    - RowIndex is kept as 0
    """

    print("\n========== WRITING FFC GAIN TABLE TO CAMERA ==========\n")

    width = int(device.nodemap["Width"].value)

    if len(gain_values) != width:
        raise Exception(
            f"Gain length mismatch. "
            f"gain_values={len(gain_values)}, camera width={width}"
        )

    gain_values = np.clip(
        gain_values,
        GAIN_RANGE_MIN,
        GAIN_RANGE_MAX
    )

    # Do not force selector/enable here because this camera is returning
    # SC_ERR_NOT_IMPLEMENTED for Selector and Enable from Python.
    print("[INFO] Skipping FlatFieldCorrectionSelector and Enable setting")
    print("[INFO] Writing directly to current FFC table shown in ArenaView")

    try:
        device.nodemap["FlatFieldCorrectionRowIndex"].value = 0
        print("[OK] FlatFieldCorrectionRowIndex set to 0")
    except Exception as e:
        print(f"[WARN] Could not set RowIndex: {e}")

    for col_idx, gain in enumerate(gain_values):

        try:
            device.nodemap["FlatFieldCorrectionColumnIndex"].value = int(col_idx)
            device.nodemap["FlatFieldCorrectionGain"].value = float(gain)

            try:
                device.nodemap["FlatFieldCorrectionOffset"].value = 0.0
            except Exception:
                pass

        except Exception as e:
            raise Exception(
                f"FFC write failed at column {col_idx}, gain={gain}: {e}"
            )

        if col_idx % 500 == 0:
            print(
                f"Written FFC column {col_idx}/{width}, "
                f"gain={float(gain):.4f}"
            )

    print("\n[INFO] FFC gain table write completed")

    try:
        device.nodemap["FlatFieldCorrectionSave"].execute()
        print("[OK] FlatFieldCorrectionSave executed")
    except Exception as e:
        raise Exception(
            f"FlatFieldCorrectionSave failed: {e}"
        )

    print("\n========== CAMERA FFC SAVE COMPLETED ==========\n")


def save_camera_user_set(device):
    """
    Saves current camera configuration into UserSet1 if supported.
    This helps retain FFC enable state and camera settings after power cycle.
    """
    print("\n========== SAVING CAMERA USER SET ==========")

    try:
        device.nodemap["UserSetSelector"].value = USER_SET_NAME
        print(f"[OK] UserSetSelector set to {USER_SET_NAME}")

        device.nodemap["UserSetSave"].execute()
        print(f"[OK] UserSetSave executed for {USER_SET_NAME}")

    except Exception as e:
        print(f"[WARN] Could not save UserSet: {e}")

    try:
        device.nodemap["UserSetDefault"].value = USER_SET_NAME
        print(f"[OK] UserSetDefault set to {USER_SET_NAME}")

    except Exception as e:
        print(f"[WARN] Could not set UserSetDefault: {e}")

    print("============================================\n")


# ============================================================
# CAPTURE IMAGE HELPERS
# ============================================================

def get_complete_buffer(device):
    buffer = device.get_buffer(timeout=GET_BUFFER_TIMEOUT)
    print("[OK] Buffer acquired")

    retry_count = 0
    retry_count_max = 30

    while buffer.is_incomplete:
        retry_count += 1
        print(f"[WARN] Incomplete buffer. Retry {retry_count}/{retry_count_max}")

        device.requeue_buffer(buffer)
        buffer = device.get_buffer(timeout=GET_BUFFER_TIMEOUT)

        if retry_count > retry_count_max:
            raise Exception("Cannot get valid image data")

    return buffer


def capture_and_save_image(device, suffix="raw"):
    """
    Captures one full image and saves it exactly as received from camera.
    Used for normal camera-side FFC capture.
    """
    buffer = get_complete_buffer(device)

    timestamp = generate_timestamp()
    filename = os.path.join(SAVE_DIR, f"image_{timestamp}_{suffix}.png")

    item = BufferFactory.copy(buffer)

    try:
        save_image(item, filename)
        print(f"[OK] Image saved: {filename}")
        print(f"[OK] Saved image size: width={item.width}, height={item.height}")
    finally:
        BufferFactory.destroy(item)
        device.requeue_buffer(buffer)

    return filename


def capture_mono16_numpy_and_save(device, suffix="flat_calibration"):
    """
    Captures one image, saves it, and also returns a Mono16 numpy image.
    Used for flat calibration capture.
    """
    buffer = get_complete_buffer(device)

    timestamp = generate_timestamp()
    filename = os.path.join(SAVE_DIR, f"image_{timestamp}_{suffix}.png")

    item = BufferFactory.copy(buffer)

    try:
        save_image(item, filename)
        print(f"[OK] Calibration image saved: {filename}")
        print(f"[OK] Calibration image size: width={item.width}, height={item.height}")

        mono_item = item
        converted_created = False

        if mono_item.pixel_format != enums.PixelFormat.Mono16:
            converted = BufferFactory.convert(mono_item, enums.PixelFormat.Mono16)
            mono_item = converted
            converted_created = True

        bits_per_pixel = mono_item.bits_per_pixel

        if bits_per_pixel != 16:
            raise Exception(f"Unsupported bits per pixel: {bits_per_pixel}")

        width = int(mono_item.width)
        height = int(mono_item.height)

        data_ptr = ctypes.cast(
            mono_item.pdata,
            ctypes.POINTER(ctypes.c_uint16)
        )

        full_image = np.ctypeslib.as_array(
            data_ptr,
            shape=(height, width)
        ).copy()

        if converted_created:
            BufferFactory.destroy(mono_item)

    finally:
        BufferFactory.destroy(item)
        device.requeue_buffer(buffer)

    return full_image, filename


def get_calibration_profile(device):
    full_image, filename = capture_mono16_numpy_and_save(
        device,
        suffix="flat_calibration"
    )

    print(f"[INFO] Full calibration image shape: {full_image.shape}")

    # One column profile: average all rows for each column.
    mean_array = np.mean(full_image, axis=0)
    np_array = mean_array.copy()

    return np_array, filename


# ============================================================
# FFC GAIN COMPUTATION / PREVIEW
# ============================================================

def get_target_pixel(np_array):
    if GAIN_TARGET_MODE == "MAX":
        target = np.max(np_array)
    elif GAIN_TARGET_MODE == "MEAN":
        target = np.mean(np_array)
    elif GAIN_TARGET_MODE == "PERCENTILE_95":
        target = np.percentile(np_array, 95)
    else:
        raise Exception(f"Unknown GAIN_TARGET_MODE: {GAIN_TARGET_MODE}")

    print(f"[INFO] Gain target mode : {GAIN_TARGET_MODE}")
    print(f"[INFO] Gain target pixel: {target}")

    return target


def plot_np_array_vs_gain_values(np_array, gain_values, corrected_np_array, filename):
    fig, ax1 = plt.subplots(figsize=(12, 4))

    ax1.set_xlabel("Pixel Index / Column Index")
    ax1.set_ylabel("Pixel Value", color="k")

    ax1.plot(np_array, linewidth=2, label="Original Column Mean")
    ax1.plot(corrected_np_array, linewidth=2, label="Corrected Column Mean")

    ax2 = ax1.twinx()
    ax2.set_ylabel("Gain Value")
    ax2.plot(gain_values, linestyle="dashed", linewidth=2, label="Gain Values")

    ax1.legend(loc=(1.04, 0))

    plot_filename = f"{filename[:-4]}_gain_plot.png"
    save_plot(fig, plot_filename, "FFC Gain Plot")


def compute_ffc_gain(np_array, filename):
    target_pixel_value = get_target_pixel(np_array)

    epsilon = 1e-6

    gain_values = np.where(
        np_array > epsilon,
        target_pixel_value / np_array,
        1.0
    )

    gain_values = np.clip(gain_values, GAIN_RANGE_MIN, GAIN_RANGE_MAX)
    corrected_np_array = np_array * gain_values

    plot_np_array_vs_gain_values(
        np_array,
        gain_values,
        corrected_np_array,
        filename
    )

    print("\n========== FFC COMPUTATION RESULT ==========")
    print("Min pixel :", float(np.min(np_array)))
    print("Max pixel :", float(np.max(np_array)))
    print("Mean pixel:", float(np.mean(np_array)))
    print("Min gain  :", float(np.min(gain_values)))
    print("Max gain  :", float(np.max(gain_values)))
    print("Gain count at max limit:", int(np.sum(gain_values >= GAIN_RANGE_MAX)))
    print("============================================\n")

    return gain_values


def plot_comparison_histogram(before_image, after_image, filename):
    fig = plt.figure(figsize=(10, 4))

    plt.hist(
        before_image.ravel(),
        bins=256,
        range=(0, 65535),
        alpha=0.5,
        label="Before"
    )

    plt.hist(
        after_image.ravel(),
        bins=256,
        range=(0, 65535),
        alpha=0.5,
        label="After"
    )

    plt.xlabel("Pixel Value")
    plt.ylabel("Frequency")
    plt.legend()

    plot_filename = f"{filename[:-4]}_histogram.png"
    save_plot(fig, plot_filename, "Before vs After Histogram")


def apply_ffc_software_preview(gain_values, filename):
    """
    Software preview only.
    This does NOT save anything inside the camera.
    """
    print("\n========== SOFTWARE FFC PREVIEW ==========")

    image = cv2.imread(filename, cv2.IMREAD_UNCHANGED)

    if image is None:
        raise Exception(f"Could not read image for software preview: {filename}")

    image_float = image.astype(np.float64)
    gain_values_2d = gain_values.reshape(1, -1)

    corrected_image = (
        image_float * gain_values_2d
    ).clip(0, 65535).astype(np.uint16)

    timestamp = generate_timestamp()
    corrected_filename = os.path.join(
        SAVE_DIR,
        f"image_{timestamp}_software_preview_corrected.png"
    )

    cv2.imwrite(
        corrected_filename,
        corrected_image,
        [cv2.IMWRITE_PNG_COMPRESSION, 0]
    )

    print(f"[OK] Software preview corrected image saved: {corrected_filename}")
    print("Original image shape  :", image.shape)
    print("Corrected image shape :", corrected_image.shape)
    print("Corrected saturated pixels:", int(np.sum(corrected_image >= 65535)))

    if SHOW_IMAGES:
        plt.figure(figsize=(8, 32))
        plt.title("Original Image - Full Height View")
        plt.imshow(image, cmap="gray", aspect="equal")
        plt.colorbar()
        plt.show()

        plt.figure(figsize=(8, 32))
        plt.title("Software Preview Corrected Image - Full Height View")
        plt.imshow(corrected_image, cmap="gray", aspect="equal")
        plt.colorbar()
        plt.show()

    plot_comparison_histogram(image, corrected_image, filename)

    print("========== SOFTWARE FFC PREVIEW COMPLETE ==========\n")

    return corrected_filename


# ============================================================
# MAIN MODE FLOWS
# ============================================================

def run_calibrate_and_save_ffc(device):
    print("\n========== MODE: CALIBRATE AND SAVE FFC ==========")

    # During flat calibration capture, camera-side FFC must be OFF.
    # enable_camera_ffc(device, False)

    print("\n[INFO] Starting stream for flat calibration image...")
    device.start_stream()

    try:
        np_array, filename = get_calibration_profile(device)
    finally:
        safe_stop_stream(device)

    gain_values = compute_ffc_gain(np_array, filename)

    # Preview only. This is not the permanent camera correction.
    apply_ffc_software_preview(gain_values, filename)

    confirm = input(
        "\nApply and permanently save FFC values to camera? (y/n): "
    ).strip().lower()

    if confirm != "y":
        print("\n[INFO] FFC was not saved to camera.")
        return

    write_ffc_gain_table_to_camera(device, gain_values)

    if SAVE_USER_SET_AFTER_FFC:
        save_camera_user_set(device)

    print(
        "\n[OK] FFC saved into camera. "
        "Next time use RUN_MODE = 'CAPTURE_WITH_SAVED_FFC'."
    )

    if CAPTURE_VERIFY_AFTER_SAVE:
        print("\n[INFO] Capturing verification image using camera-side FFC...")
        enable_camera_ffc(device, True)

        device.start_stream()
        try:
            capture_and_save_image(device, suffix="camera_ffc_verify")
        finally:
            safe_stop_stream(device)


def run_capture_with_saved_ffc(device):
    print("\n========== MODE: CAPTURE WITH SAVED CAMERA FFC ==========")

    # Do not compute gain again. Just enable saved camera FFC and capture.
    enable_camera_ffc(device, True)

    print("\n[INFO] Starting stream with camera-side FFC enabled...")
    device.start_stream()

    try:
        capture_and_save_image(device, suffix="camera_ffc")
    finally:
        safe_stop_stream(device)


# ============================================================
# MAIN
# ============================================================

def main():
    print("\nFlat Field Correction Example Starting...\n")

    devices = create_devices_with_tries()
    device = None

    try:
        device = system.select_device(devices)

        model_name = device.nodemap["DeviceModelName"].value
        print(f"Connected Camera: {model_name}")

        if model_name not in SUPPORTED_MODELS:
            raise Exception(f"Unsupported camera model: {model_name}")

        print_camera_info(device)
        configure_linescan_camera(device, model_name)
        if RUN_FFC_WRITE_TEST:
            test_ffc_write_access(device)
        if RUN_MODE == "CALIBRATE_AND_SAVE_FFC":
            run_calibrate_and_save_ffc(device)

        elif RUN_MODE == "CAPTURE_WITH_SAVED_FFC":
            run_capture_with_saved_ffc(device)

        else:
            raise Exception(f"Unknown RUN_MODE: {RUN_MODE}")

    finally:
        try:
            system.destroy_device()
            print("[OK] Device destroyed")
        except Exception as e:
            print(f"[WARN] Could not destroy device: {e}")

    print("\nFlat Field Correction Complete.")


# ============================================================
# ENTRY
# ============================================================

if __name__ == "__main__":
    main()
