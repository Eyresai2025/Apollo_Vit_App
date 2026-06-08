import os
import time
import ctypes
from pathlib import Path
from ztrak_save_2d_and_ply import convert_raw_to_outputs
DLL_DIRS = [
    r"C:\Program Files\Teledyne DALSA\Sapera\Bin",
    r"C:\Program Files\Teledyne DALSA\Sapera\Components\NET\Bin",
    r"C:\Program Files\Teledyne DALSA\GenICam 3.20\bin\Win64_x64",
    r"C:\Program Files\Teledyne\Common Components\Bin",
    r"C:\Program Files\Teledyne\GigE Vision Interface\Bin",
]

for d in DLL_DIRS:
    if Path(d).exists():
        os.add_dll_directory(d)
        print("[DLL DIR ADDED]", d)

from pythonnet import load
load("netfx")

import clr
import System

SAPERA_DOTNET_DLL = r"C:\Program Files\Teledyne DALSA\Sapera\Components\NET\Bin\DALSA.SaperaLT.SapClassBasic.dll"
clr.AddReference(SAPERA_DOTNET_DLL)

from DALSA.SaperaLT.SapClassBasic import (
    SapManager,
    SapManagerBase,
    SapLocation,
    SapAcqDevice,
    SapBuffer,
    SapAcqDeviceToBuf,
)

OUT_DIR = Path(__file__).resolve().parent / "ztrak_output"
OUT_DIR.mkdir(exist_ok=True)

NUM_BUFFERS = 4
WAIT_TIMEOUT_MS = 60000

# Production save control
KEEP_RAW_FILE = False      # False for production, True for debugging
KEEP_META_FILE = True

# Configuration source:
#   "PYTHON"   -> Python sets safe features. Locked features optional below.
#   "USERSET1" -> Load UserSet1 saved from Z-Expert, then apply safe overrides.
CONFIG_MODE = "PYTHON"
USERSET_NAME = "UserSet1"
PYTHON_WRITE_LOCKED_FEATURES = False  # False avoids popup from profileRate/ExposureTime/Gain writes.
APPLY_SAFE_OVERRIDES_AFTER_USERSET = True

# Converter / PLY output settings
CONVERTER_FULL_RESOLUTION_PLY = True
CONVERTER_DEBUG_PLY_STEP = 4
CONVERTER_PLY_FORMAT = "ascii"       # "ascii" for large AI-style PLY, "binary" for smaller/faster.
CONVERTER_CENTER_Z = True
CONVERTER_INVALID_C_VALUE = 65535
CONVERTER_X_SCALER_UM = 10.0
CONVERTER_Z_SCALER_UM = 5.0
CONVERTER_Y_STEP_MM = 1.0             # TODO: replace with encoder/conveyor mm-per-profile.

# AI team settings
AI_LASER_POWER = 2047
AI_PROFILE_RATE = 8000.0
AI_EXPOSURE_TIME_US = 100.0
AI_REFLECTANCE_THRESHOLD = 256
AI_GAIN = 4.0
AI_NOISE_REDUCTION_LEVEL = 16

# From your handwritten setting.
# If AI team says 17450 or 17150, change only this value.
AI_PROFILES_PER_SCAN = 17150
AI_TRIGGER_MODE = "Off"


def find_ztrak_acqdevice():
    SapManager.DetectAllServers(SapManagerBase.DetectServerType.GenCP)
    SapManager.DetectAllServers(SapManagerBase.DetectServerType.All)

    server_count = SapManager.GetServerCount()
    print("[INFO] Server count:", server_count)

    for server_idx in range(server_count):
        try:
            server_name = SapManager.GetServerName(server_idx)
            server_type = SapManager.GetServerType(server_idx)
            accessible = SapManager.IsServerAccessible(server_idx)

            print("\n" + "=" * 80)
            print("SERVER INDEX:", server_idx)
            print("Server name:", server_name)
            print("Server type:", server_type)
            print("Is accessible:", accessible)

            if not accessible:
                continue

            acqdev_count = SapManager.GetResourceCount(
                server_idx,
                SapManagerBase.ResourceType.AcqDevice
            )

            print("AcqDevice count:", acqdev_count)

            for res_idx in range(acqdev_count):
                res_name = SapManager.GetResourceName(
                    server_idx,
                    SapManagerBase.ResourceType.AcqDevice,
                    res_idx
                )

                available = SapManager.IsResourceAvailable(
                    server_idx,
                    SapManagerBase.ResourceType.AcqDevice,
                    res_idx
                )

                print(f"  Resource {res_idx}: {res_name} | available={available}")

                if available:
                    return server_name, res_idx, res_name

        except Exception as e:
            print("[WARN] server scan error:", e)

    raise RuntimeError("No available Z-Trak AcqDevice found")


def try_set_feature(acq_device, name, value):
    try:
        if not acq_device.IsFeatureAvailable(name):
            print(f"[SKIP] Feature not available: {name}")
            return False

        ok = acq_device.SetFeatureValue(name, value)
        print(f"[SET] {name} = {value} -> {ok}")
        return ok

    except Exception as e:
        print(f"[WARN] Could not set {name}={value}: {e}")
        return False


def try_execute_command_feature(acq_device, name):
    """Execute GenICam command-like feature safely."""
    for value in (True, 1, "Execute"):
        try:
            if not acq_device.IsFeatureAvailable(name):
                print(f"[SKIP] Command not available: {name}")
                return False

            ok = acq_device.SetFeatureValue(name, value)
            print(f"[COMMAND] {name} using {value!r} -> {ok}")

            if ok:
                return True

        except Exception as e:
            print(f"[WARN] Command attempt failed: {name}={value!r}: {e}")

    return False


def update_features_to_device(acq_device):
    try:
        ok = acq_device.UpdateFeaturesToDevice()
        print("[UPDATE FEATURES TO DEVICE] ->", ok)
        return bool(ok)
    except Exception as e:
        print("[WARN] UpdateFeaturesToDevice failed:", e)
        return False


def update_features_from_device(acq_device):
    try:
        ok = acq_device.UpdateFeaturesFromDevice()
        print("[UPDATE FEATURES FROM DEVICE] ->", ok)
        return bool(ok)
    except Exception as e:
        print("[WARN] UpdateFeaturesFromDevice failed:", e)
        return False


def load_userset(acq_device, user_set_name=USERSET_NAME):
    print(f"\n[LOAD USER SET] {user_set_name}")

    selector_ok = try_set_feature(acq_device, "UserSetSelector", user_set_name)

    if not selector_ok:
        print("[WARN] Could not select user set. Save UserSet1 in Z-Expert first or use CONFIG_MODE='PYTHON'.")
        return False

    load_ok = try_execute_command_feature(acq_device, "UserSetLoad")

    if not load_ok:
        print("[WARN] UserSetLoad did not confirm success. If capture works, settings may already be active.")

    update_features_from_device(acq_device)
    return selector_ok and load_ok


def apply_safe_capture_overrides(acq_device):
    """Safe values that worked on your setup and should not create popup warnings."""
    print("\n[SAFE CAPTURE OVERRIDES]")

    try_set_feature(acq_device, "laserActivation", "On")
    try_set_feature(acq_device, "laserControlMode", "Manual")
    try_set_feature(acq_device, "laserPower", AI_LASER_POWER)
    try_set_feature(acq_device, "peakDetectorReflectanceThreshold", AI_REFLECTANCE_THRESHOLD)
    try_set_feature(acq_device, "noiseReductionLevel", AI_NOISE_REDUCTION_LEVEL)
    try_set_feature(acq_device, "profilesPerScan", AI_PROFILES_PER_SCAN)
    try_set_feature(acq_device, "TriggerMode", AI_TRIGGER_MODE)

    update_features_to_device(acq_device)
    update_features_from_device(acq_device)


def apply_python_configuration(acq_device):
    print("\n[PYTHON CONFIGURATION MODE]")

    apply_safe_capture_overrides(acq_device)

    if PYTHON_WRITE_LOCKED_FEATURES:
        print("\n[OPTIONAL LOCKED FEATURE WRITES]")
        print("[INFO] These may return False / popup depending on current device mode.")
        try_set_feature(acq_device, "profileRate", AI_PROFILE_RATE)
        try_set_feature(acq_device, "ExposureTime", AI_EXPOSURE_TIME_US)
        try_set_feature(acq_device, "Gain", AI_GAIN)
        update_features_to_device(acq_device)
        update_features_from_device(acq_device)
    else:
        print("\n[SKIP LOCKED FEATURES]")
        print("[INFO] profileRate / ExposureTime / Gain should be set from Z-Expert/UserSet.")
        print(f"[INFO] Expected: profileRate={AI_PROFILE_RATE}, ExposureTime={AI_EXPOSURE_TIME_US}, Gain={AI_GAIN}")


def configure_ztrak(acq_device):
    mode = CONFIG_MODE.strip().upper()
    print("\n[CONFIG MODE]", mode)

    if mode == "USERSET1":
        load_userset(acq_device, USERSET_NAME)
        if APPLY_SAFE_OVERRIDES_AFTER_USERSET:
            apply_safe_capture_overrides(acq_device)
        return

    if mode == "PYTHON":
        apply_python_configuration(acq_device)
        return

    raise ValueError(f"Unsupported CONFIG_MODE={CONFIG_MODE!r}. Use 'PYTHON' or 'USERSET1'.")


def get_buffer_param(buffer, prm_name, dummy):
    try:
        prm = getattr(SapBuffer.Prm, prm_name)
        ret = buffer.GetParameter(prm, dummy)

        if isinstance(ret, tuple):
            ok = bool(ret[0])
            val = ret[1]
            return ok, val

        return False, None

    except Exception as e:
        return False, f"<error: {e}>"


def save_buffer_with_sapera(buffer):
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    save_attempts = [
        (OUT_DIR / f"ztrak_scan_{timestamp}.tif", ""),
        (OUT_DIR / f"ztrak_scan_{timestamp}.tiff", ""),
        (OUT_DIR / f"ztrak_scan_{timestamp}.raw", ""),
    ]

    for path, options in save_attempts:
        try:
            ok = buffer.Save(str(path), options)
            print(f"[SAVE TRY] {path} options='{options}' -> {ok}")
            if ok:
                return path
        except Exception as e:
            print(f"[SAVE WARN] {path}: {e}")

    return None


def dump_raw_buffer(buffer):
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    raw_path = OUT_DIR / f"ztrak_scan_{timestamp}_manual_dump.raw"
    meta_path = OUT_DIR / f"ztrak_scan_{timestamp}_manual_dump_meta.txt"

    width = int(buffer.Width)
    height = int(buffer.Height)
    pitch = int(buffer.Pitch)
    bpp = int(buffer.BytesPerPixel)
    fmt = str(buffer.Format)
    pixel_depth = int(buffer.PixelDepth)
    index = int(buffer.Index)

    print("\n[BUFFER INFO]")
    print("Index        :", index)
    print("Width        :", width)
    print("Height       :", height)
    print("Pitch        :", pitch)
    print("BytesPerPixel:", bpp)
    print("PixelDepth   :", pixel_depth)
    print("Format       :", fmt)

    print("\n[3D BUFFER PARAMETERS]")

    params_to_read = [
        ("SCAN3D_COORD_SCALE_A", 0.0),
        ("SCAN3D_COORD_SCALE_B", 0.0),
        ("SCAN3D_COORD_SCALE_C", 0.0),
        ("SCAN3D_COORD_OFFSET_A", 0.0),
        ("SCAN3D_COORD_OFFSET_B", 0.0),
        ("SCAN3D_COORD_OFFSET_C", 0.0),
        ("SCAN3D_INVALID_DATA_VALUE_C", 0),
        ("SCAN3D_DISTANCE_UNIT", 0),
        ("DEVICE_SCAN_TYPE", 0),
        ("SCAN3D_OUTPUT_MODE", 0),
    ]

    param_reads = {}

    for prm_name, dummy in params_to_read:
        ok, val = get_buffer_param(buffer, prm_name, dummy)
        param_reads[prm_name] = val if ok else f"<failed: {val}>"
        print(f"{prm_name}: ok={ok}, value={val}")

    if pitch <= 0:
        pitch = width * max(bpp, 1)

    byte_count = pitch * height
    print("Dump bytes   :", byte_count)

    # pythonnet 3.x ByRef/out parameter style
    ret = buffer.GetAddress(System.IntPtr.Zero)
    print("GetAddress() raw return:", ret)

    if isinstance(ret, tuple):
        ok = bool(ret[0])
        addr_ptr = ret[1]
    else:
        raise RuntimeError("buffer.GetAddress() did not return tuple")

    print("GetAddress() :", ok)

    if not ok:
        raise RuntimeError("buffer.GetAddress() failed")

    addr = addr_ptr.ToInt64()

    if addr == 0:
        raise RuntimeError("buffer address is NULL")

    raw_bytes = ctypes.string_at(addr, byte_count)

    with open(raw_path, "wb") as f:
        f.write(raw_bytes)

    with open(meta_path, "w", encoding="utf-8") as f:
        f.write(f"width={width}\n")
        f.write(f"height={height}\n")
        f.write(f"pitch={pitch}\n")
        f.write(f"bytes_per_pixel={bpp}\n")
        f.write(f"pixel_depth={pixel_depth}\n")
        f.write(f"format={fmt}\n")
        f.write(f"byte_count={byte_count}\n")

        for k, v in param_reads.items():
            f.write(f"{k}={v}\n")

        f.write(f"ai_laser_power={AI_LASER_POWER}\n")
        f.write(f"ai_profile_rate={AI_PROFILE_RATE}\n")
        f.write(f"ai_exposure_time_us={AI_EXPOSURE_TIME_US}\n")
        f.write(f"ai_reflectance_threshold={AI_REFLECTANCE_THRESHOLD}\n")
        f.write(f"ai_gain={AI_GAIN}\n")
        f.write(f"ai_noise_reduction_level={AI_NOISE_REDUCTION_LEVEL}\n")
        f.write(f"ai_profiles_per_scan={AI_PROFILES_PER_SCAN}\n")
        f.write(f"ai_trigger_mode={AI_TRIGGER_MODE}\n")
        f.write(f"config_mode={CONFIG_MODE}\n")
        f.write(f"userset_name={USERSET_NAME}\n")
        f.write(f"python_write_locked_features={PYTHON_WRITE_LOCKED_FEATURES}\n")
        f.write(f"converter_full_resolution_ply={CONVERTER_FULL_RESOLUTION_PLY}\n")
        f.write(f"converter_debug_ply_step={CONVERTER_DEBUG_PLY_STEP}\n")
        f.write(f"converter_ply_format={CONVERTER_PLY_FORMAT}\n")
        f.write(f"converter_invalid_c_value={CONVERTER_INVALID_C_VALUE}\n")
        f.write(f"converter_x_scaler_um={CONVERTER_X_SCALER_UM}\n")
        f.write(f"converter_z_scaler_um={CONVERTER_Z_SCALER_UM}\n")
        f.write(f"converter_y_step_mm={CONVERTER_Y_STEP_MM}\n")

    print("[RAW SAVED]", raw_path)
    print("[META SAVED]", meta_path)

    return raw_path, meta_path


def main():
    print("\n[OK] Sapera SDK loaded")

    server_name, resource_index, resource_name = find_ztrak_acqdevice()

    print("\n[SELECTED]")
    print("Server name   :", server_name)
    print("Resource index:", resource_index)
    print("Resource name :", resource_name)

    location = SapLocation(server_name, resource_index)

    acq_device = None
    buffer = None
    xfer = None

    try:
        print("\n[CREATE ACQ DEVICE]")
        acq_device = SapAcqDevice(location)

        ok = acq_device.Create()
        print("AcqDevice.Create() ->", ok)

        if not ok:
            raise RuntimeError("SapAcqDevice.Create() failed")

        update_features_from_device(acq_device)

        configure_ztrak(acq_device)

        print("\n[CREATE BUFFER]")
        buffer = SapBuffer(
            NUM_BUFFERS,
            acq_device,
            SapBuffer.MemoryType.ScatterGather
        )

        ok = buffer.Create()
        print("Buffer.Create() ->", ok)

        if not ok:
            print("[WARN] ScatterGather buffer failed. Trying Default memory.")

            try:
                buffer.Destroy()
            except Exception:
                pass

            buffer = SapBuffer(
                NUM_BUFFERS,
                acq_device,
                SapBuffer.MemoryType.Default
            )

            ok = buffer.Create()
            print("Buffer.Create(Default) ->", ok)

            if not ok:
                raise RuntimeError("SapBuffer.Create() failed")

        print("Buffer Width :", buffer.Width)
        print("Buffer Height:", buffer.Height)
        print("Buffer Format:", buffer.Format)
        print("Buffer Pitch :", buffer.Pitch)
        print("Buffer BPP   :", buffer.BytesPerPixel)

        print("\n[CREATE TRANSFER]")
        xfer = SapAcqDeviceToBuf(acq_device, buffer)

        ok = xfer.Create()
        print("Xfer.Create() ->", ok)

        if not ok:
            raise RuntimeError("SapAcqDeviceToBuf.Create() failed")

        print("\n[SNAP ONE SCAN]")
        ok = xfer.Snap(1)
        print("Snap(1) ->", ok)

        if not ok:
            raise RuntimeError("Snap(1) failed")

        print(f"[WAIT] timeout={WAIT_TIMEOUT_MS} ms")
        ok = xfer.Wait(WAIT_TIMEOUT_MS)
        print("Wait() ->", ok)

        if not ok:
            print("[WARN] Wait timeout/fail. Calling Abort().")
            try:
                xfer.Abort()
            except Exception:
                pass
            raise RuntimeError("Transfer did not complete")

        print("\n[TRANSFER DONE]")
        print("Buffer Index:", buffer.Index)
        print("Buffer State:", buffer.State)
        print("Space Used  :", buffer.SpaceUsed)

        # print("\n[SAVE USING SAPERA]")
        # saved_path = save_buffer_with_sapera(buffer)

        # if saved_path:
        #     print("[SAPERA SAVE OK]", saved_path)
        # else:
        #     print("[WARN] Sapera Save did not produce a file")

        print("\n[DUMP RAW BUFFER]")
        raw_path, meta_path = dump_raw_buffer(buffer)

        print("\n[CONVERT RAW TO 2D + PLY]")
        output_paths = convert_raw_to_outputs(
            raw_path=raw_path,
            meta_path=meta_path,
            output_dir=OUT_DIR,
            full_resolution_ply=CONVERTER_FULL_RESOLUTION_PLY,
            debug_ply_step=CONVERTER_DEBUG_PLY_STEP,
            ply_format=CONVERTER_PLY_FORMAT,
            center_z=CONVERTER_CENTER_Z,
            invalid_c_value=CONVERTER_INVALID_C_VALUE,
            x_scaler_um=CONVERTER_X_SCALER_UM,
            z_scaler_um=CONVERTER_Z_SCALER_UM,
            y_step_mm=CONVERTER_Y_STEP_MM,
        )

        # Optional raw cleanup for production
        if not KEEP_RAW_FILE:
            try:
                raw_path.unlink(missing_ok=True)
                print("[CLEANUP] Raw file deleted for production:", raw_path)
                output_paths["raw"] = None
            except Exception as e:
                print("[WARN] Could not delete raw file:", e)

        if not KEEP_META_FILE:
            try:
                meta_path.unlink(missing_ok=True)
                print("[CLEANUP] Meta file deleted:", meta_path)
                output_paths["meta"] = None
            except Exception as e:
                print("[WARN] Could not delete meta file:", e)

        print("\n[FINAL OUTPUT PATHS]")
        for k, v in output_paths.items():
            print(f"{k}: {v}")

        print("\n[SUCCESS] One Z-Trak scan captured and converted")

    finally:
        print("\n[CLEANUP]")

        if xfer is not None:
            try:
                if xfer.Grabbing:
                    xfer.Abort()
            except Exception:
                pass

            try:
                xfer.Destroy()
                print("[OK] Xfer destroyed")
            except Exception as e:
                print("[WARN] Xfer destroy failed:", e)

        if buffer is not None:
            try:
                buffer.Destroy()
                print("[OK] Buffer destroyed")
            except Exception as e:
                print("[WARN] Buffer destroy failed:", e)

        if acq_device is not None:
            try:
                acq_device.Destroy()
                print("[OK] AcqDevice destroyed")
            except Exception as e:
                print("[WARN] AcqDevice destroy failed:", e)


if __name__ == "__main__":
    main()