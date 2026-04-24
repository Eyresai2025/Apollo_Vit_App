import math
import os
import sys
from datetime import datetime

import cv2  # type: ignore


def tyre_basics(cycle_no, tyrename):
    tyrename = str(tyrename).upper()
    st = tyrename.find("R") + 3
    detail = tyrename[:st]

    section_width = tyrename[0:3]
    id_inch = detail[-2:]

    if len(detail) != 8:
        aspect_ratio = 80
    else:
        aspect_ratio = detail[3:5]

    inner_dia = int(id_inch) * 25.4
    section_height = int(section_width) * int(aspect_ratio) / 100
    outer_dia = int(inner_dia) + int(section_height) * 2

    roller_dia = 100
    roller_dist = 350

    date_t = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    date = datetime.now().strftime("%d-%m-%Y")

    tyre_dict = {
        "tirename": tyrename,
        "cycle_no": cycle_no,
        "defect": False,
        "numberOfDefects": 0,
        "sectionHeight": int(section_height),
        "sectionWidth": int(section_width),
        "aspectRatio": int(aspect_ratio),
        "radius": int(id_inch),
        "od": int(outer_dia),
        "rollerDiameter": roller_dia,
        "rollerDistance": roller_dist,
        "inspectionDateTime": date_t,
        "inspectionDate": date,
    }
    return tyre_dict


def sidewall_dimensions(tyrename):
    str1 = str(tyrename).upper()
    st = str1.find("R") + 3
    detail = str1[:st]

    section_width = str1[0:3]
    id_inch = detail[-2:]

    if len(detail) != 8:
        aspect_ratio = 80
    else:
        aspect_ratio = detail[3:5]

    inner_dia = int(id_inch) * 25.4
    section_height = int(section_width) * int(aspect_ratio) / 100

    sidewall_dia = int(inner_dia) + int(section_height)
    sidewall_width = section_height
    sidewall_height = sidewall_dia * math.pi
    area_of_sidewall = sidewall_width * sidewall_height

    return int(sidewall_width), int(sidewall_height), int(area_of_sidewall)


def tread_dimensions(tyrename):
    str1 = str(tyrename).upper()
    st = str1.find("R") + 3
    detail = str1[:st]

    section_width = str1[0:3]
    id_inch = detail[-2:]

    if len(detail) != 8:
        aspect_ratio = 80
    else:
        aspect_ratio = detail[3:5]

    inner_dia = int(id_inch) * 25.4
    section_height = int(section_width) * int(aspect_ratio) / 100
    outer_dia = int(inner_dia) + int(section_height) * 2

    tread_width = int(section_width)
    tread_height = outer_dia * math.pi
    area_of_tread = tread_width * tread_height

    return int(tread_width), int(tread_height), int(area_of_tread)


def innerwall_dimensions(tyrename):
    str1 = str(tyrename).upper()
    st = str1.find("R") + 3
    detail = str1[:st]

    section_width = str1[0:3]
    id_inch = detail[-2:]

    if len(detail) != 8:
        aspect_ratio = 80
    else:
        aspect_ratio = detail[3:5]

    inner_dia = int(id_inch) * 25.4
    section_height = int(section_width) * int(aspect_ratio) / 100

    innerwall_width = section_height
    innerwall_ref_dia = inner_dia + section_height
    innerwall_height = innerwall_ref_dia * math.pi
    area_of_innerwall = innerwall_width * innerwall_height

    return int(innerwall_width), int(innerwall_height), int(area_of_innerwall)


def bead_dimensions(tyrename, beadWidth_mm=20, beadCenterOffset_mm=0):
    str1 = str(tyrename).upper()
    st = str1.find("R") + 3
    detail = str1[:st]

    id_inch = detail[-2:]
    inner_dia = int(id_inch) * 25.4

    bead_width = beadWidth_mm
    bead_ref_dia = inner_dia + bead_width + 2 * beadCenterOffset_mm
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
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)


def load_env(root_dir=None):
    env_vars = {}

    env_path = resource_path(".env")
    with open(env_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line or line.startswith("#"):
                continue

            if "=" not in line:
                continue

            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")

            env_vars[key] = value

    return env_vars