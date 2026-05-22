# src/COMMON/live_sku_resolver.py

import os
import struct
from src.COMMON.common import load_env


def _read_s7_int(plc_client, db_number: int, byte_offset: int) -> int:
    """
    Reads Siemens INT from DBx.DBW byte_offset.
    Siemens INT is signed 16-bit big-endian.
    """
    if plc_client is None:
        raise RuntimeError("PLC client is not available. Run Test Mode hardware check first.")

    data = plc_client.db_read(db_number, byte_offset, 2)

    if data is None or len(data) < 2:
        raise RuntimeError(f"PLC read failed for DB{db_number}.DBW{byte_offset}")

    return int(struct.unpack(">h", bytes(data[:2]))[0])


def recipe_number_to_sku(recipe_number: int, prefix: str = "SKU", digits: int = 3) -> str:
    if recipe_number <= 0:
        raise RuntimeError(f"Invalid PLC active recipe number: {recipe_number}")

    return f"{prefix}_{int(recipe_number):0{digits}d}"


def resolve_live_sku_from_plc(plc_client, media_path: str, env_path: str):
    """
    Reads active recipe from PLC and maps it to AI calibration SKU folder.

    Example:
        DB74.DBW78 = 1
        -> SKU_001
        -> media/AI_Calibration_Files/SKU_001
    """
    env = load_env(env_path)

    db_number = int(env.get("PLC_ACTIVE_RECIPE_DB", 74))
    byte_offset = int(env.get("PLC_ACTIVE_RECIPE_BYTE", 78))

    sku_prefix = env.get("LIVE_SKU_PREFIX", "SKU")
    sku_digits = int(env.get("LIVE_SKU_DIGITS", 3))

    recipe_number = _read_s7_int(
        plc_client=plc_client,
        db_number=db_number,
        byte_offset=byte_offset,
    )

    sku_name = recipe_number_to_sku(
        recipe_number=recipe_number,
        prefix=sku_prefix,
        digits=sku_digits,
    )

    sku_dir = os.path.join(
        media_path,
        "AI_Calibration_Files",
        sku_name,
    )

    if not os.path.isdir(sku_dir):
        raise FileNotFoundError(
            f"PLC active recipe is {recipe_number}, mapped SKU is {sku_name}, "
            f"but folder not found:\n{sku_dir}"
        )

    return {
        "recipe_number": recipe_number,
        "sku_name": sku_name,
        "sku_dir": sku_dir,
        "tag": f"DB{db_number}.DBW{byte_offset}",
    }