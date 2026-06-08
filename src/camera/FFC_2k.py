
"""
Flat Field Correction (FFC) Example
for Lucid TRI02KA-M Line Scan Camera

Requirements:
    pip install opencv-python==4.10.0.84
    pip install numpy==2.0.0
    pip install matplotlib==3.9.1

Also install:
    Arena SDK + arena_api Python package

Run:
    python flat_field_correction.py
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
# USER CONFIGURATION
# ============================================================

EXPOSURE_TIME = 120.0
ACQUISITION_FRAME_RATE = 500.0
CAMERA_GAIN = 12.0
LINESCAN_WIDTH = 0
LINESCAN_HEIGHT = 14000

GET_BUFFER_TIMEOUT = 2000

FFC_SELECTOR = "FlatFieldCorrection1"

# ============================================================
# CONSTANTS
# ============================================================

TAB1 = "  "

GAIN_RANGE_MIN = 1.0
GAIN_RANGE_MAX = 15.99

PIXEL_FORMAT = PixelFormat.Mono16

os.makedirs("images", exist_ok=True)


# ============================================================
# DEVICE CONNECTION
# ============================================================

def create_devices_with_tries():
    """
    Waits for camera connection.
    """

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

def configure_linescan_camera(device):

    print("\n========== CONFIGURING CAMERA ==========\n")

    try:

        # ==========================================
        # WIDTH
        # ==========================================

        width_node = device.nodemap["Width"]

        width_value = (
            width_node.max
            if LINESCAN_WIDTH < 1
            else LINESCAN_WIDTH
        )

        device.nodemap["Width"].value = width_value

        print(f"Width set: {width_value}")

        # ==========================================
        # HEIGHT
        # ==========================================

        device.nodemap["Height"].value = LINESCAN_HEIGHT

        print(f"Height set: {LINESCAN_HEIGHT}")

        # ==========================================
        # PIXEL FORMAT
        # ==========================================

        device.nodemap["PixelFormat"].value = PIXEL_FORMAT

        print("PixelFormat set: Mono16")

        # ==========================================
        # ACQUISITION MODE
        # ==========================================

        device.nodemap["AcquisitionMode"].value = "Continuous"

        print("AcquisitionMode set: Continuous")

        # ==========================================
        # TRIGGER MODE
        # ==========================================

        device.nodemap["TriggerMode"].value = "Off"

        print("TriggerMode set: Off")
        try:
            device.nodemap["Gain"].value = CAMERA_GAIN
            print(f"Gain set: {CAMERA_GAIN}")
        except Exception as e:
            print(f"Could not set Gain: {e}")
        # ==========================================
        # EXPOSURE
        # ==========================================

        device.nodemap[
            "ExposureTime"
        ].value = EXPOSURE_TIME

        print(f"ExposureTime set: {EXPOSURE_TIME}")

        # ==========================================
        # STREAM SETTINGS
        # ==========================================

        tl_stream_nodemap = device.tl_stream_nodemap

        tl_stream_nodemap[
            "StreamAutoNegotiatePacketSize"
        ].value = True

        tl_stream_nodemap[
            "StreamPacketResendEnable"
        ].value = True

        print("Stream settings configured")

        print("\n========== CONFIG SUCCESS ==========\n")

    except Exception as e:

        print("\n========== CONFIG FAILED ==========\n")

        print(e)

        print("\n===================================\n")

# ============================================================
# SAVE IMAGE
# ============================================================

def save_image(buffer, filename):

    writer = Writer()
    writer.pattern = filename
    writer.save(buffer)


# ============================================================
# SAVE PLOT
# ============================================================

def save_plot(fig, filename, title=None):

    if title:
        fig.suptitle(title)

    fig.tight_layout()

    plt.grid(True)

    plt.savefig(
        filename,
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()


# ============================================================
# TIMESTAMP
# ============================================================

def generate_timestamp():

    return datetime.datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )


# ============================================================
# CAPTURE IMAGE
# ============================================================

def get_calibration_image(device):

    buffer = device.get_buffer(
        timeout=GET_BUFFER_TIMEOUT
    )

    print("Buffer acquired")

    timestamp = generate_timestamp()

    filename = (
        f"images/image_{timestamp}.png"
    )

    retry_count = 0
    retry_count_max = 30

    while buffer.is_incomplete:

        retry_count += 1

        device.requeue_buffer(buffer)

        buffer = device.get_buffer(
            timeout=GET_BUFFER_TIMEOUT
        )

        if retry_count > retry_count_max:

            raise Exception(
                "Cannot get valid image data"
            )

    item = BufferFactory.copy(buffer)

    save_image(item, filename)

    device.requeue_buffer(buffer)

    pixel_format = item.pixel_format

    width = item.width
    height = item.height

    if pixel_format != enums.PixelFormat.Mono16:

        item = BufferFactory.convert(
            item,
            enums.PixelFormat.Mono16
        )

    bits_per_pixel = item.bits_per_pixel

    if bits_per_pixel != 16:

        raise Exception(
            f"Unsupported bits per pixel: "
            f"{bits_per_pixel}"
        )

    data_ptr = ctypes.cast(
        item.pdata,
        ctypes.POINTER(ctypes.c_uint16)
    )

    full_image = np.ctypeslib.as_array(
        data_ptr,
        shape=(height, width)
    )

    mean_array = np.mean(
        full_image,
        axis=0
    )

    np_array = mean_array.copy()

    BufferFactory.destroy(item)

    return np_array, filename


# ============================================================
# PLOT GRAPH
# ============================================================

def plot_np_array_vs_gain_values(
    np_array,
    gain_values,
    corrected_np_array,
    filename
):

    fig, ax1 = plt.subplots(
        figsize=(12, 4)
    )

    color1 = "tab:red"

    ax1.set_xlabel("Pixel Index")

    ax1.set_ylabel(
        "Pixel Value",
        color="k"
    )

    ax1.plot(
        np_array,
        color=color1,
        linewidth=2,
        label="Original Pixel Values"
    )

    ax2 = ax1.twinx()

    color2 = "tab:blue"

    ax2.set_ylabel(
        "Gain Value",
        color=color2
    )

    ax2.plot(
        gain_values,
        color=color2,
        linestyle="dashed",
        linewidth=2,
        label="Gain Values"
    )

    color3 = "tab:green"

    ax1.plot(
        corrected_np_array,
        color=color3,
        linewidth=2,
        label="Corrected Pixel Values"
    )

    ax1.legend(loc=(1.04, 0))

    plot_filename = (
        f"{filename[:-4]}_gain_plot.png"
    )

    save_plot(
        fig,
        plot_filename,
        "FFC Gain Plot"
    )


# ============================================================
# COMPUTE GAIN
# ============================================================

def compute_ffc_gain(np_array, filename):

    max_pixel_value = np.max(np_array)

    epsilon = 1e-6

    gain_values = np.where(
        np_array > epsilon,
        max_pixel_value / np_array,
        1.0
    )

    gain_values = np.clip(
        gain_values,
        GAIN_RANGE_MIN,
        GAIN_RANGE_MAX
    )

    corrected_np_array = (
        np_array * gain_values
    )

    plot_np_array_vs_gain_values(
        np_array,
        gain_values,
        corrected_np_array,
        filename
    )
    print("Min pixel :", np.min(np_array))
    print("Max pixel :", np.max(np_array))
    print("Mean pixel:", np.mean(np_array))

    print("Min gain  :", np.min(gain_values))
    print("Max gain  :", np.max(gain_values))
    return gain_values


# ============================================================
# HISTOGRAM
# ============================================================

def plot_comparison_histogram(
    before_image,
    after_image,
    filename
):

    plt.figure(figsize=(10, 4))

    plt.hist(
        before_image.ravel(),
        bins=256,
        range=(0, 65535),
        color="blue",
        alpha=0.5,
        label="Before"
    )

    plt.hist(
        after_image.ravel(),
        bins=256,
        range=(0, 65535),
        color="red",
        alpha=0.5,
        label="After"
    )

    plt.xlabel("Pixel Value")

    plt.ylabel("Frequency")

    plt.legend()

    plot_filename = (
        f"{filename[:-4]}_histogram.png"
    )

    save_plot(
        plt.gcf(),
        plot_filename,
        "Before vs After Histogram"
    )


# ============================================================
# VALIDATE
# ============================================================

def validate_correction(
    gain_values,
    filename
):

    image = cv2.imread(
        filename,
        cv2.IMREAD_UNCHANGED
    ).astype(np.float64)

    gain_values = gain_values.reshape(1, -1)

    corrected_image = (
        image * gain_values
    ).clip(0, 65535).astype(np.uint16)

    timestamp = generate_timestamp()

    corrected_filename = (
        f"images/image_{timestamp}_corrected.png"
    )

    cv2.imwrite(
        corrected_filename,
        corrected_image
    )

    print(
        f"Corrected image saved: "
        f"{corrected_filename}"
    )

    plot_comparison_histogram(
        image,
        corrected_image,
        filename
    )

    confirm = input(
        "Apply FFC values? (y/n): "
    ).strip().lower()

    return confirm == "y"


# ============================================================
# APPLY FFC
# ============================================================

def apply_ffc_software(gain_values, filename):

    print("\nApplying SOFTWARE flat field correction...\n")

    image = cv2.imread(
        filename,
        cv2.IMREAD_UNCHANGED
    ).astype(np.float64)

    gain_values = gain_values.reshape(1, -1)

    corrected_image = (
        image * gain_values
    ).clip(0, 65535).astype(np.uint16)

    timestamp = generate_timestamp()

    corrected_filename = (
        f"images/image_{timestamp}_corrected.png"
    )

    cv2.imwrite(
        corrected_filename,
        corrected_image
    )

    print(f"Corrected image saved:")
    print(corrected_filename)

    # ==========================================
    # DISPLAY ORIGINAL IMAGE
    # ==========================================

    plt.figure(figsize=(8, 32))

    plt.title("Original Image - Full 16000 Height View")

    plt.imshow(
        image,
        cmap='gray',
        aspect='equal'
    )

    plt.colorbar()

    plt.show()

    # ==========================================
    # DISPLAY CORRECTED IMAGE
    # ==========================================

    print("Original image shape  :", image.shape)
    print("Corrected image shape :", corrected_image.shape)

    plt.figure(figsize=(8, 32))

    plt.title("Corrected Image - Full 16000 Height View")

    plt.imshow(
        corrected_image,
        cmap='gray',
        aspect='equal'
    )

    plt.colorbar()

    plt.show()

    # ==========================================
    # HISTOGRAM
    # ==========================================

    plot_comparison_histogram(
        image,
        corrected_image,
        filename
    )

    print("\nSoftware FFC correction complete.\n")


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "\nFlat Field Correction Example Starting...\n"
    )

    devices = create_devices_with_tries()

    device = system.select_device(devices)

    model_name = device.nodemap[
        "DeviceModelName"
    ].value

    print(f"Connected Camera: {model_name}")

    if model_name != "TRI02KA-M":

        raise Exception(
            f"Unsupported camera model: "
            f"{model_name}"
        )

    configure_linescan_camera(device)

    print("\nStarting stream...")

    device.start_stream()

    np_array, filename = get_calibration_image(
        device
    )

    device.stop_stream()

    gain_values = compute_ffc_gain(
        np_array,
        filename
    )

    # ONLY SOFTWARE CORRECTION
    apply_ffc_software(
        gain_values,
        filename
    )

    system.destroy_device()

    print(
        "\nFlat Field Correction Complete."
    )


# ============================================================
# ENTRY
# ============================================================

if __name__ == "__main__":

    main()
