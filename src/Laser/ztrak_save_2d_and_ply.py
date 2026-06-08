import sys
from pathlib import Path

import numpy as np
import cv2


OUT_DIR = Path(__file__).resolve().parent / "ztrak_output"
OUT_DIR.mkdir(exist_ok=True, parents=True)

EXPECTED_FORMAT = "Coord3D_CR16"


def read_meta(meta_path: Path):
    meta = {}
    current_section = None

    with open(meta_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            if line.startswith("[") and line.endswith("]"):
                current_section = line.strip("[]")
                continue

            if "=" not in line:
                continue

            key, value = line.split("=", 1)

            if current_section:
                meta[f"{current_section}.{key.strip()}"] = value.strip()
            else:
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


def find_latest_raw_and_meta(output_dir=None):
    if output_dir is None:
        output_dir = OUT_DIR
    else:
        output_dir = Path(output_dir)

    raw_files = sorted(
        output_dir.glob("*manual_dump.raw"),
        key=lambda p: p.stat().st_mtime
    )

    if not raw_files:
        raise FileNotFoundError(f"No manual_dump.raw files found in {output_dir}")

    raw_path = raw_files[-1]
    prefix = raw_path.name.replace("_manual_dump.raw", "")
    meta_path = output_dir / f"{prefix}_manual_dump_meta.txt"

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


def save_2d_images(output_dir, stem, c_channel, r_channel, invalid_mask):
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    r_preview_png = output_dir / f"{stem}_2d_reflectance_preview_8bit.png"
    c_preview_png = output_dir / f"{stem}_2d_height_preview_8bit.png"

    r_u8 = normalize_to_uint8(r_channel, invalid_mask)
    c_u8 = normalize_to_uint8(c_channel, invalid_mask)

    cv2.imwrite(str(r_preview_png), r_u8)
    cv2.imwrite(str(c_preview_png), c_u8)

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


def _prepare_points(
    c_channel,
    r_channel,
    invalid_mask,
    full_resolution_ply,
    debug_ply_step,
    center_z,
    x_scaler_um,
    z_scaler_um,
    y_step_mm,
):
    ply_step = 1 if full_resolution_ply else int(debug_ply_step)

    if ply_step < 1:
        ply_step = 1

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
    gray = intensity[valid]

    if center_z and z.size > 0:
        z_median = np.median(z)
        z = z - z_median
        print("[PLY] Z centered by median:", float(z_median))

    return x, y, z, gray, ply_step


def save_binary_ply(ply_path, x, y, z, gray):
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


def save_ascii_ply(ply_path, x, y, z, gray, chunk_size=500000):
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

        for start in range(0, point_count, chunk_size):
            end = min(start + chunk_size, point_count)
            lines = [
                f"{x[i]:.6f} {y[i]:.6f} {z[i]:.6f} {int(gray[i])} {int(gray[i])} {int(gray[i])}\n"
                for i in range(start, end)
            ]
            f.writelines(lines)
            print(f"[PLY ASCII WRITE] {end}/{point_count}")


def save_ply(
    output_dir,
    stem,
    c_channel,
    r_channel,
    invalid_mask,
    full_resolution_ply=False,
    debug_ply_step=4,
    ply_format="binary",
    center_z=True,
    x_scaler_um=10.0,
    z_scaler_um=5.0,
    y_step_mm=1.0,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    x, y, z, gray, ply_step = _prepare_points(
        c_channel=c_channel,
        r_channel=r_channel,
        invalid_mask=invalid_mask,
        full_resolution_ply=full_resolution_ply,
        debug_ply_step=debug_ply_step,
        center_z=center_z,
        x_scaler_um=x_scaler_um,
        z_scaler_um=z_scaler_um,
        y_step_mm=y_step_mm,
    )

    point_count = x.size
    ply_format = str(ply_format).strip().lower()

    if ply_format not in ("binary", "ascii"):
        raise ValueError("ply_format must be 'binary' or 'ascii'.")

    density_tag = "fullres" if full_resolution_ply else f"step{ply_step}"
    ply_path = output_dir / f"{stem}_pointcloud_{density_tag}_{ply_format}_xz_mm_y_profile.ply"

    print("\n[PLY INFO]")
    print("PLY format  :", ply_format)
    print("PLY step    :", ply_step)
    print("Point count :", point_count)
    print("X scale     :", x_scaler_um, "um")
    print("Z scale     :", z_scaler_um, "um")
    print("Y step      :", y_step_mm, "mm/profile-step placeholder")
    print("Output      :", ply_path)

    if ply_format == "binary":
        save_binary_ply(ply_path, x, y, z, gray)
    else:
        save_ascii_ply(ply_path, x, y, z, gray)

    print("[PLY SAVED]", ply_path)
    return ply_path


def save_summary(output_dir, stem, meta, output_paths, converter_config):
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    summary_path = output_dir / f"{stem}_save_summary.txt"

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("[ZTRAK SAVE SUMMARY]\n\n")

        f.write("[META]\n")
        for k, v in meta.items():
            f.write(f"{k}={v}\n")

        f.write("\n[CONVERTER CONFIG]\n")
        for k, v in converter_config.items():
            f.write(f"{k}={v}\n")

        f.write("\n[OUTPUT FILES]\n")
        for k, v in output_paths.items():
            f.write(f"{k}={v}\n")

    print("[SUMMARY SAVED]", summary_path)
    return summary_path


def convert_raw_to_outputs(
    raw_path,
    meta_path,
    output_dir=None,
    full_resolution_ply=False,
    debug_ply_step=4,
    ply_format="binary",
    center_z=True,
    invalid_c_value=65535,
    x_scaler_um=10.0,
    z_scaler_um=5.0,
    y_step_mm=1.0,
):
    raw_path = Path(raw_path)
    meta_path = Path(meta_path)

    if output_dir is None:
        output_dir = raw_path.parent
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)

    print("[RAW ]", raw_path)
    print("[META]", meta_path)

    meta = read_meta(meta_path)
    c_channel, r_channel = decode_coord3d_cr16(raw_path, meta)

    print("\n[INVALID]")
    print("Using invalid C value:", invalid_c_value)

    invalid_mask = np.zeros(c_channel.shape, dtype=bool)
    invalid_mask |= (c_channel == int(invalid_c_value))
    invalid_mask |= (r_channel == 0)

    print("Invalid pixel count:", int(invalid_mask.sum()))
    print("Invalid percentage :", float(invalid_mask.mean() * 100.0))

    stem = raw_path.stem.replace("_manual_dump", "")

    image_paths = save_2d_images(output_dir, stem, c_channel, r_channel, invalid_mask)
    ply_path = save_ply(
        output_dir=output_dir,
        stem=stem,
        c_channel=c_channel,
        r_channel=r_channel,
        invalid_mask=invalid_mask,
        full_resolution_ply=full_resolution_ply,
        debug_ply_step=debug_ply_step,
        ply_format=ply_format,
        center_z=center_z,
        x_scaler_um=x_scaler_um,
        z_scaler_um=z_scaler_um,
        y_step_mm=y_step_mm,
    )

    output_paths = {}
    output_paths.update(image_paths)
    output_paths["ply"] = ply_path
    output_paths["raw"] = raw_path
    output_paths["meta"] = meta_path

    converter_config = {
        "full_resolution_ply": full_resolution_ply,
        "debug_ply_step": debug_ply_step,
        "ply_format": ply_format,
        "center_z": center_z,
        "invalid_c_value": invalid_c_value,
        "x_scaler_um": x_scaler_um,
        "z_scaler_um": z_scaler_um,
        "y_step_mm": y_step_mm,
    }

    summary_path = save_summary(output_dir, stem, meta, output_paths, converter_config)
    output_paths["summary"] = summary_path

    print("\n[SUCCESS] 2D image and PLY generated")
    return output_paths


def main():
    raw_path, meta_path = resolve_input_paths()
    convert_raw_to_outputs(raw_path, meta_path)


if __name__ == "__main__":
    main()
