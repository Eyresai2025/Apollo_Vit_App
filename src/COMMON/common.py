import math
import os
import sys
import re
import logging
from datetime import datetime
from typing import Dict, Any, Optional, Tuple

import cv2  # type: ignore

from src.COMMON.exceptions import ValidationError
from src.COMMON.config import TireConstants

logger = logging.getLogger(__name__)


def _parse_tyre_name(tyrename: str) -> Tuple[int, int, int, str]:
    """
    Parse and normalize tyre name.

    Supported formats:
    - 195/65R15
    - 195_65_R15
    - 195-65-R15
    - 195/65r15

    Returns:
        section_width, aspect_ratio, rim_inch, normalized_name
    """
    if not tyrename:
        raise ValidationError("Tire name cannot be empty", code="EMPTY_TYRE_NAME")

    raw = str(tyrename).strip().upper()

    match = re.fullmatch(r"(\d{3})[/_-](\d{2})[/_-]?R(\d{2,3})", raw)
    if not match:
        raise ValidationError(
            f"Invalid tire name format: {tyrename}",
            code="INVALID_TYRE_FORMAT",
            details={
                "expected_format": "NNN/NNRNN",
                "received": raw,
            },
        )

    section_width = int(match.group(1))
    aspect_ratio = int(match.group(2))
    rim_inch = int(match.group(3))

    normalized = f"{section_width}/{aspect_ratio:02d}R{rim_inch}"

    return section_width, aspect_ratio, rim_inch, normalized


def validate_tyre_name(tyrename: str) -> str:
    """
    Validate and normalize tyre name to format NNN/NNRNN.

    Example:
        195_65_R15 -> 195/65R15
        195-65-R15 -> 195/65R15
        195/65r15  -> 195/65R15
    """
    _, _, _, normalized = _parse_tyre_name(tyrename)
    return normalized


def tyre_basics(cycle_no: int, tyrename: str) -> Dict[str, Any]:
    """
    Calculate tire basic parameters from tire name and cycle number.
    """
    if not isinstance(cycle_no, int) or cycle_no < 0:
        raise ValidationError(
            "Cycle number must be non-negative integer",
            code="INVALID_CYCLE_NO",
        )

    try:
        section_width, aspect_ratio, id_inch, tyrename = _parse_tyre_name(tyrename)

        inner_dia = id_inch * TireConstants.MM_PER_INCH
        section_height = section_width * aspect_ratio / 100
        outer_dia = inner_dia + section_height * 2

        now = datetime.now()
        date_t = now.strftime("%Y-%m-%d_%H-%M-%S")
        date = now.strftime("%d-%m-%Y")

        tyre_dict = {
            "tirename": tyrename,
            "cycle_no": cycle_no,
            "defect": False,
            "numberOfDefects": 0,
            "sectionHeight": int(section_height),
            "sectionWidth": section_width,
            "aspectRatio": aspect_ratio,
            "radius": id_inch,
            "od": int(outer_dia),
            "rollerDiameter": TireConstants.ROLLER_DIAMETER_MM,
            "rollerDistance": TireConstants.ROLLER_DISTANCE_MM,
            "inspectionDateTime": date_t,
            "inspectionDate": date,
        }

        return tyre_dict

    except ValidationError:
        raise

    except ValueError as e:
        raise ValidationError(
            f"Failed to parse tire dimensions: {e}",
            code="TIRE_PARSE_ERROR",
            details={"tyrename": tyrename},
        )


def sidewall_dimensions(tyrename: str) -> tuple:
    """
    Calculate sidewall dimensions from tire name.

    Returns:
        width_mm, height_mm, area_mm2
    """
    section_width, aspect_ratio, id_inch, tyrename = _parse_tyre_name(tyrename)

    inner_dia = id_inch * TireConstants.MM_PER_INCH
    section_height = section_width * aspect_ratio / 100

    sidewall_dia = inner_dia + section_height
    sidewall_width = section_height
    sidewall_height = sidewall_dia * math.pi
    area_of_sidewall = sidewall_width * sidewall_height

    return int(sidewall_width), int(sidewall_height), int(area_of_sidewall)


def tread_dimensions(tyrename: str) -> tuple:
    """
    Calculate tread dimensions from tire name.

    Returns:
        width_mm, height_mm, area_mm2
    """
    section_width, aspect_ratio, id_inch, tyrename = _parse_tyre_name(tyrename)

    inner_dia = id_inch * TireConstants.MM_PER_INCH
    section_height = section_width * aspect_ratio / 100
    outer_dia = inner_dia + section_height * 2

    tread_width = section_width
    tread_height = outer_dia * math.pi
    area_of_tread = tread_width * tread_height

    return int(tread_width), int(tread_height), int(area_of_tread)


def innerwall_dimensions(tyrename: str) -> tuple:
    """
    Calculate innerwall dimensions from tire name.

    Returns:
        width_mm, height_mm, area_mm2
    """
    section_width, aspect_ratio, id_inch, tyrename = _parse_tyre_name(tyrename)

    inner_dia = id_inch * TireConstants.MM_PER_INCH
    section_height = section_width * aspect_ratio / 100

    innerwall_width = section_height
    innerwall_ref_dia = inner_dia + section_height
    innerwall_height = innerwall_ref_dia * math.pi
    area_of_innerwall = innerwall_width * innerwall_height

    return int(innerwall_width), int(innerwall_height), int(area_of_innerwall)


def bead_dimensions(
    tyrename: str,
    bead_width_mm: Optional[int] = None,
    bead_center_offset_mm: Optional[int] = None,
) -> tuple:
    """
    Calculate bead dimensions from tire name.

    Returns:
        width_mm, height_mm, area_mm2
    """
    _, _, id_inch, tyrename = _parse_tyre_name(tyrename)

    if bead_width_mm is None:
        bead_width_mm = TireConstants.BEAD_WIDTH_MM

    if bead_center_offset_mm is None:
        bead_center_offset_mm = TireConstants.BEAD_CENTER_OFFSET_MM

    inner_dia = id_inch * TireConstants.MM_PER_INCH

    bead_width = bead_width_mm
    bead_ref_dia = inner_dia + bead_width + 2 * bead_center_offset_mm
    bead_height = bead_ref_dia * math.pi
    area_of_bead = bead_width * bead_height

    return int(bead_width), int(bead_height), int(area_of_bead)


def tyre_bboxes(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    assert img is not None, f"File could not be read: {img_path}"

    img = cv2.medianBlur(img, 5)
    _, th1 = cv2.threshold(img, 7, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(th1, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise ValueError(f"No contours found in image: {img_path}")

    cnt = sorted(contours, key=cv2.contourArea, reverse=True)
    x, y, w, h = cv2.boundingRect(cnt[0])
    area = w * h

    return x, y, w, h, area


def defect_dimension(bbox):
    x = int(bbox[0])
    y = int(bbox[1])
    defect_width = int(bbox[2])
    defect_height = int(bbox[3])

    xmin = int(x)
    ymin = int(y)
    xmax = int(defect_width + x)
    ymax = int(defect_height + y)

    defect_area = int(defect_width * defect_height)

    # kept xmin/ymin/xmax/ymax calculation in case you use it later
    _ = (xmin, ymin, xmax, ymax)

    return defect_height, defect_width, defect_area


def resource_path(relative_path: str) -> str:
    """
    Get correct resource path for normal Python run and PyInstaller exe run.
    """
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        return os.path.join(sys._MEIPASS, relative_path)

    return os.path.join(os.path.abspath("."), relative_path)


def load_env(root_dir=None):
    """
    Backward-compatible configuration dictionary.

    New code should use ``src.COMMON.config.get_config()`` for typed values.
    Existing modules may continue using this function while they are migrated.
    """
    from src.COMMON.config import load_legacy_env

    return load_legacy_env(root_dir)

