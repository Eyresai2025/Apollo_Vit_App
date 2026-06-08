import re
from pathlib import Path

import numpy as np
import cv2


OUT_DIR = Path(__file__).resolve().parent / "ztrak_output"


def read_meta(meta_path: Path):
    meta = {}

    with open(meta_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or "=" not in line:
                continue

            k, v = line.split("=", 1)
            meta[k.strip()] = v.strip()

    return meta


def normalize_to_uint8(img):
    img = img.astype(np.float32)

    valid = img[np.isfinite(img)]

    if valid.size == 0:
        return np.zeros(img.shape, dtype=np.uint8)

    lo = np.percentile(valid, 1)
    hi = np.percentile(valid, 99)

    if hi <= lo:
        hi = lo + 1

    out = (img - lo) / (hi - lo)
    out = np.clip(out, 0, 1)
    out = (out * 255).astype(np.uint8)

    return out


def find_latest_raw_and_meta():
    raw_files = sorted(OUT_DIR.glob("*manual_dump.raw"), key=lambda p: p.stat().st_mtime)

    if not raw_files:
        raise FileNotFoundError(f"No raw files found in {OUT_DIR}")

    raw_path = raw_files[-1]

    # Your meta file follows same timestamp prefix
    prefix = raw_path.name.replace("_manual_dump.raw", "")
    meta_path = OUT_DIR / f"{prefix}_manual_dump_meta.txt"

    if not meta_path.exists():
        raise FileNotFoundError(f"Meta file not found: {meta_path}")

    return raw_path, meta_path


def main():
    raw_path, meta_path = find_latest_raw_and_meta()

    print("[RAW ]", raw_path)
    print("[META]", meta_path)

    meta = read_meta(meta_path)

    width = int(meta["width"])
    height = int(meta["height"])
    pitch = int(meta["pitch"])
    bpp = int(meta["bytes_per_pixel"])
    fmt = meta["format"]

    print("\n[META INFO]")
    print("width :", width)
    print("height:", height)
    print("pitch :", pitch)
    print("bpp   :", bpp)
    print("format:", fmt)

    if fmt != "Coord3D_CR16":
        print("[WARN] This decoder is currently written for Coord3D_CR16")

    raw = np.fromfile(raw_path, dtype=np.uint8)

    expected_bytes = pitch * height
    print("\n[RAW SIZE]")
    print("actual bytes  :", raw.size)
    print("expected bytes:", expected_bytes)

    if raw.size < expected_bytes:
        raise RuntimeError("Raw file smaller than expected")

    raw = raw[:expected_bytes]

    # Reshape row-wise using pitch
    rows = raw.reshape(height, pitch)

    # Actual useful data per row = width * bytes_per_pixel
    useful_bytes_per_row = width * bpp
    rows = rows[:, :useful_bytes_per_row]

    # Coord3D_CR16 = two uint16 channels per pixel: C + R
    data_u16 = rows.reshape(height, width, 2, 2)

    # Convert 2 bytes to uint16, little-endian
    data_u16 = data_u16[:, :, :, 0].astype(np.uint16) | (
        data_u16[:, :, :, 1].astype(np.uint16) << 8
    )

    c_channel = data_u16[:, :, 0]
    r_channel = data_u16[:, :, 1]

    print("\n[DECODED]")
    print("C shape:", c_channel.shape, "dtype:", c_channel.dtype)
    print("R shape:", r_channel.shape, "dtype:", r_channel.dtype)

    print("\n[C STATS]")
    print("min:", int(c_channel.min()))
    print("max:", int(c_channel.max()))
    print("mean:", float(c_channel.mean()))

    print("\n[R STATS]")
    print("min:", int(r_channel.min()))
    print("max:", int(r_channel.max()))
    print("mean:", float(r_channel.mean()))

    stem = raw_path.stem.replace("_manual_dump", "")

    c_npy = OUT_DIR / f"{stem}_C_coord_uint16.npy"
    r_npy = OUT_DIR / f"{stem}_R_reflectance_uint16.npy"

    np.save(c_npy, c_channel)
    np.save(r_npy, r_channel)

    c_png = OUT_DIR / f"{stem}_C_coord_preview.png"
    r_png = OUT_DIR / f"{stem}_R_reflectance_preview.png"

    cv2.imwrite(str(c_png), normalize_to_uint8(c_channel))
    cv2.imwrite(str(r_png), normalize_to_uint8(r_channel))

    # Save small CSV preview only, full CSV will be huge
    preview_csv = OUT_DIR / f"{stem}_preview_200x200.csv"
    preview = c_channel[:200, :200]
    np.savetxt(preview_csv, preview, delimiter=",", fmt="%d")

    print("\n[SAVED]")
    print(c_npy)
    print(r_npy)
    print(c_png)
    print(r_png)
    print(preview_csv)

    print("\n[SUCCESS] Raw Z-Trak scan decoded")


if __name__ == "__main__":
    main()