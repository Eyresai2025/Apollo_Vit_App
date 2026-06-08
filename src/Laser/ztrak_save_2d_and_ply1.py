import sys
from pathlib import Path

import numpy as np
import cv2


OUT_DIR = Path(__file__).resolve().parent / "ztrak_output"
OUT_DIR.mkdir(exist_ok=True)

EXPECTED_FORMAT = "Coord3D_CR16"

# PLY mode globals
# FULL_RESOLUTION_PLY=True -> PLY_STEP=1, dense/full PLY for AI/Sherlock validation.
# FULL_RESOLUTION_PLY=False -> DEBUG_PLY_STEP used, smaller/faster debug PLY.
FULL_RESOLUTION_PLY = True
DEBUG_PLY_STEP = 4

# AI team mentioned large PLY files. ASCII full-res will be much larger than binary.
# Use "ascii" for AI/manual-style large files, "binary" for faster/smaller production debug.
PLY_FORMAT = "ascii"  # "ascii" or "binary"
ASCII_PLY_CHUNK_SIZE = 500_000

# From Z-Expert
INVALID_C_VALUE = 65535
X_SCALER_UM = 10.0   # X Scaler from Z-Expert
Z_SCALER_UM = 5.0    # Z Scaler from Z-Expert

# Y depends on movement/encoder/conveyor speed.
# For now use profile index as Y. Later replace with encoder/conveyor mm per profile.
Y_STEP_MM = 1.0

CENTER_Z_FOR_PLY = True
REMOVE_ZERO_REFLECTANCE = True
def read_meta(meta_path: Path):
    meta = {}

    with open(meta_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line or "=" not in line:
                continue

            key, value = line.split("=", 1)
            meta[key.strip()] = value.strip()

    return meta


def is_failed_value(value):
    if value is None:
        return True

    s = str(value).strip()

    if not s:
        return True

    if s.startswith("<failed"):
        return True

    if s.startswith("<error"):
        return True

    return False


def get_float(meta, key, default=None):
    try:
        value = meta.get(key, None)

        if is_failed_value(value):
            return default

        return float(value)

    except Exception:
        return default


def get_int(meta, key, default=None):
    try:
        value = meta.get(key, None)

        if is_failed_value(value):
            return default

        return int(float(value))

    except Exception:
        return default


def find_latest_raw_and_meta():
    raw_files = sorted(
        OUT_DIR.glob("*manual_dump.raw"),
        key=lambda p: p.stat().st_mtime
    )

    if not raw_files:
        raise FileNotFoundError(f"No manual_dump.raw files found in {OUT_DIR}")

    raw_path = raw_files[-1]

    prefix = raw_path.name.replace("_manual_dump.raw", "")
    meta_path = OUT_DIR / f"{prefix}_manual_dump_meta.txt"

    if not meta_path.exists():
        raise FileNotFoundError(f"Meta file not found: {meta_path}")

    return raw_path, meta_path


def resolve_input_paths():
    if len(sys.argv) >= 3:
        raw_path = Path(sys.argv[1])
        meta_path = Path(sys.argv[2])

        if not raw_path.exists():
            raise FileNotFoundError(f"Raw file not found: {raw_path}")

        if not meta_path.exists():
            raise FileNotFoundError(f"Meta file not found: {meta_path}")

        return raw_path, meta_path

    return find_latest_raw_and_meta()


def normalize_to_uint8(img, invalid_mask=None):
    arr = img.astype(np.float32)

    if invalid_mask is not None:
        arr = arr.copy()
        arr[invalid_mask] = np.nan

    valid = arr[np.isfinite(arr)]

    if valid.size == 0:
        return np.zeros(img.shape, dtype=np.uint8)

    lo = np.percentile(valid, 1)
    hi = np.percentile(valid, 99)

    if hi <= lo:
        hi = lo + 1

    out = (arr - lo) / (hi - lo)
    out = np.clip(out, 0, 1)
    out = np.nan_to_num(out, nan=0.0)
    out = (out * 255).astype(np.uint8)

    return out


def decode_coord3d_cr16(raw_path: Path, meta: dict):
    width = int(meta["width"])
    height = int(meta["height"])
    pitch = int(meta["pitch"])
    bpp = int(meta["bytes_per_pixel"])
    fmt = meta["format"]

    print("[META]")
    print("width :", width)
    print("height:", height)
    print("pitch :", pitch)
    print("bpp   :", bpp)
    print("format:", fmt)

    if fmt != EXPECTED_FORMAT:
        print(f"[WARN] This code expects {EXPECTED_FORMAT}, but got {fmt}")

    raw = np.fromfile(raw_path, dtype=np.uint8)

    expected_bytes = pitch * height

    print("\n[RAW SIZE]")
    print("actual bytes  :", raw.size)
    print("expected bytes:", expected_bytes)

    if raw.size < expected_bytes:
        raise RuntimeError("Raw file is smaller than expected from meta")

    raw = raw[:expected_bytes]

    rows = raw.reshape(height, pitch)

    useful_bytes_per_row = width * bpp
    rows = rows[:, :useful_bytes_per_row]

    # Coord3D_CR16 = C uint16 + R uint16
    data = rows.reshape(height, width, 2, 2)

    data_u16 = data[:, :, :, 0].astype(np.uint16) | (
        data[:, :, :, 1].astype(np.uint16) << 8
    )

    c_channel = data_u16[:, :, 0]
    r_channel = data_u16[:, :, 1]

    print("\n[DECODED]")
    print("C shape:", c_channel.shape, c_channel.dtype)
    print("R shape:", r_channel.shape, r_channel.dtype)

    print("\n[C STATS]")
    print("min :", int(c_channel.min()))
    print("max :", int(c_channel.max()))
    print("mean:", float(c_channel.mean()))

    print("\n[R STATS]")
    print("min :", int(r_channel.min()))
    print("max :", int(r_channel.max()))
    print("mean:", float(r_channel.mean()))

    return c_channel, r_channel


def save_2d_images(stem, c_channel, r_channel, invalid_mask, output_dir=OUT_DIR):
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # 8-bit previews for checking
    r_preview_png = output_dir / f"{stem}_2d_reflectance_preview_8bit.png"
    c_preview_png = output_dir / f"{stem}_2d_height_preview_8bit.png"

    r_u8 = normalize_to_uint8(r_channel, invalid_mask)
    c_u8 = normalize_to_uint8(c_channel, invalid_mask)

    cv2.imwrite(str(r_preview_png), r_u8)
    cv2.imwrite(str(c_preview_png), c_u8)

    # 16-bit images for AI / processing
    r16 = r_channel.copy().astype(np.uint16)
    c16 = c_channel.copy().astype(np.uint16)

    r16[invalid_mask] = 0
    c16[invalid_mask] = 0

    r16_png = output_dir / f"{stem}_2d_reflectance_16bit.png"
    c16_png = output_dir / f"{stem}_2d_height_16bit.png"

    cv2.imwrite(str(r16_png), r16)
    cv2.imwrite(str(c16_png), c16)

    print("\n[2D SAVED]")
    print(r_preview_png)
    print(c_preview_png)
    print(r16_png)
    print(c16_png)

    return {
        "reflectance_preview_8bit": r_preview_png,
        "height_preview_8bit": c_preview_png,
        "reflectance_16bit": r16_png,
        "height_16bit": c16_png,
    }


def get_effective_ply_step(full_resolution_ply=None, debug_ply_step=None):
    if full_resolution_ply is None:
        full_resolution_ply = FULL_RESOLUTION_PLY
    if debug_ply_step is None:
        debug_ply_step = DEBUG_PLY_STEP

    return 1 if full_resolution_ply else max(1, int(debug_ply_step))


def _write_ascii_ply(ply_path, x, y, z, gray):
    point_count = x.size
    header = (
        "ply\n"
        "format ascii 1.0\n"
        f"element vertex {point_count}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    )

    with open(ply_path, "w", encoding="ascii", newline="\n") as f:
        f.write(header)
        for start in range(0, point_count, ASCII_PLY_CHUNK_SIZE):
            end = min(start + ASCII_PLY_CHUNK_SIZE, point_count)
            chunk = np.column_stack((
                x[start:end], y[start:end], z[start:end],
                gray[start:end], gray[start:end], gray[start:end],
            ))
            np.savetxt(f, chunk, fmt="%.6f %.6f %.6f %d %d %d")
            print(f"[PLY ASCII WRITE] {end}/{point_count}")


def _write_binary_ply(ply_path, x, y, z, gray):
    point_count = x.size
    vertex = np.empty(
        point_count,
        dtype=[
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ],
    )

    vertex["x"] = x
    vertex["y"] = y
    vertex["z"] = z
    vertex["red"] = gray
    vertex["green"] = gray
    vertex["blue"] = gray

    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {point_count}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    )

    with open(ply_path, "wb") as f:
        f.write(header.encode("ascii"))
        vertex.tofile(f)


def save_ply(
    stem,
    c_channel,
    r_channel,
    invalid_mask,
    meta,
    output_dir=OUT_DIR,
    *,
    full_resolution_ply=None,
    debug_ply_step=None,
    ply_format=None,
    center_z=None,
    invalid_c_value=None,
    x_scaler_um=None,
    z_scaler_um=None,
    y_step_mm=None,
):
    """
    Save PLY point cloud.

    X uses Z-Expert X scaler.
    Z uses Z-Expert Z scaler.
    Y currently uses profile index spacing until encoder/conveyor calibration is added.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    ply_step = get_effective_ply_step(full_resolution_ply, debug_ply_step)
    ply_format = (ply_format or PLY_FORMAT).strip().lower()
    center_z = CENTER_Z_FOR_PLY if center_z is None else bool(center_z)
    x_scaler_um = X_SCALER_UM if x_scaler_um is None else float(x_scaler_um)
    z_scaler_um = Z_SCALER_UM if z_scaler_um is None else float(z_scaler_um)
    y_step_mm = Y_STEP_MM if y_step_mm is None else float(y_step_mm)

    c_ds = c_channel[::ply_step, ::ply_step]
    r_ds = r_channel[::ply_step, ::ply_step]
    invalid_ds = invalid_mask[::ply_step, ::ply_step]

    yy, xx = np.indices(c_ds.shape)

    x = (xx * ply_step * x_scaler_um / 1000.0).astype(np.float32)
    y = (yy * ply_step * y_step_mm).astype(np.float32)
    z = (c_ds.astype(np.float32) * z_scaler_um / 1000.0).astype(np.float32)

    intensity = normalize_to_uint8(r_ds, invalid_ds)
    valid = ~invalid_ds

    x = x[valid]
    y = y[valid]
    z = z[valid]
    gray = intensity[valid].astype(np.uint8)

    if center_z and z.size > 0:
        z_median = np.median(z)
        z = z - z_median
        print("[PLY] Z centered by median:", float(z_median))

    point_count = x.size

    mode_label = "fullres" if ply_step == 1 else f"step{ply_step}"
    format_label = "ascii" if ply_format == "ascii" else "binary"
    ply_path = output_dir / f"{stem}_pointcloud_{mode_label}_{format_label}_xz_mm_y_profile.ply"

    print("\n[PLY INFO]")
    print("PLY format  :", ply_format)
    print("PLY step    :", ply_step)
    print("Point count :", point_count)
    print("X scale     :", x_scaler_um, "um")
    print("Z scale     :", z_scaler_um, "um")
    print("Y step      :", y_step_mm, "mm/profile-step placeholder")
    print("Output      :", ply_path)

    if ply_format == "ascii":
        _write_ascii_ply(ply_path, x, y, z, gray)
    elif ply_format == "binary":
        _write_binary_ply(ply_path, x, y, z, gray)
    else:
        raise ValueError("ply_format must be 'ascii' or 'binary'")

    print("[PLY SAVED]", ply_path)
    return ply_path


def save_summary(stem, meta, output_paths, output_dir=OUT_DIR, extra_settings=None):
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    summary_path = output_dir / f"{stem}_save_summary.txt"

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("[ZTRAK SAVE SUMMARY]\n\n")

        f.write("[META]\n")
        for k, v in meta.items():
            f.write(f"{k}={v}\n")

        if extra_settings:
            f.write("\n[CONVERTER SETTINGS]\n")
            for k, v in extra_settings.items():
                f.write(f"{k}={v}\n")

        f.write("\n[OUTPUT FILES]\n")
        for k, v in output_paths.items():
            f.write(f"{k}={v}\n")

    print("[SUMMARY SAVED]", summary_path)

def convert_raw_to_outputs(
    raw_path,
    meta_path,
    output_dir=None,
    *,
    full_resolution_ply=None,
    debug_ply_step=None,
    ply_format=None,
    center_z=None,
    invalid_c_value=None,
    x_scaler_um=None,
    z_scaler_um=None,
    y_step_mm=None,
):
    """
    Production helper function.
    Converts one captured raw + meta pair into:
      - 8-bit preview images
      - 16-bit processing images
      - PLY point cloud
      - summary txt

    Returns dictionary of output paths.
    """
    raw_path = Path(raw_path)
    meta_path = Path(meta_path)

    if output_dir is None:
        output_dir = OUT_DIR
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)

    print("[RAW ]", raw_path)
    print("[META]", meta_path)

    meta = read_meta(meta_path)

    c_channel, r_channel = decode_coord3d_cr16(raw_path, meta)

    if invalid_c_value is None:
        invalid_c_value = INVALID_C_VALUE

    print("\n[INVALID]")
    print("Using Z-Expert invalid C value:", invalid_c_value)

    invalid_mask = np.zeros(c_channel.shape, dtype=bool)

    invalid_mask |= (c_channel == int(invalid_c_value))

    if REMOVE_ZERO_REFLECTANCE:
        invalid_mask |= (r_channel == 0)

    print("Invalid pixel count:", int(invalid_mask.sum()))
    print("Invalid percentage :", float(invalid_mask.mean() * 100.0))

    stem = raw_path.stem.replace("_manual_dump", "")

    image_paths = save_2d_images(stem, c_channel, r_channel, invalid_mask, output_dir)
    ply_path = save_ply(
        stem,
        c_channel,
        r_channel,
        invalid_mask,
        meta,
        output_dir,
        full_resolution_ply=full_resolution_ply,
        debug_ply_step=debug_ply_step,
        ply_format=ply_format,
        center_z=center_z,
        invalid_c_value=invalid_c_value,
        x_scaler_um=x_scaler_um,
        z_scaler_um=z_scaler_um,
        y_step_mm=y_step_mm,
    )

    output_paths = {}
    output_paths.update(image_paths)
    output_paths["ply"] = ply_path
    output_paths["raw"] = raw_path
    output_paths["meta"] = meta_path

    extra_settings = {
        "full_resolution_ply": full_resolution_ply if full_resolution_ply is not None else FULL_RESOLUTION_PLY,
        "debug_ply_step": debug_ply_step if debug_ply_step is not None else DEBUG_PLY_STEP,
        "effective_ply_step": get_effective_ply_step(full_resolution_ply, debug_ply_step),
        "ply_format": ply_format if ply_format is not None else PLY_FORMAT,
        "center_z": center_z if center_z is not None else CENTER_Z_FOR_PLY,
        "invalid_c_value": invalid_c_value,
        "x_scaler_um": x_scaler_um if x_scaler_um is not None else X_SCALER_UM,
        "z_scaler_um": z_scaler_um if z_scaler_um is not None else Z_SCALER_UM,
        "y_step_mm": y_step_mm if y_step_mm is not None else Y_STEP_MM,
        "remove_zero_reflectance": REMOVE_ZERO_REFLECTANCE,
    }

    summary_path = save_summary(stem, meta, output_paths, output_dir, extra_settings=extra_settings)
    output_paths["summary"] = summary_path

    print("\n[SUCCESS] 2D image and PLY generated")

    return output_paths

def main():
    raw_path, meta_path = resolve_input_paths()
    convert_raw_to_outputs(raw_path, meta_path)


if __name__ == "__main__":
    main()