# src/COMMON/recipe_tag_map.py

"""
Common PLC recipe tag map.

Used by:
- Axis Status Page
- New SKU / Axis Teaching
- Recipe Management
- PLC recipe write/read verification

DB75:
    Active/running recipe values from PLC/HMI.
    Axis Status only READS DB75.

DB53:
    Recipe entry/write DB.
    New SKU / Recipe Management writes DB53.

DB74.DBW78:
    Active running recipe number.

DB75.DBW288:
    Recipe number entry/write tag.
    Do NOT use as active recipe number.
"""

# =========================================================
# PLACEHOLDER / CONFIRMATION AREA
# =========================================================
# PLC/HMI currently shows SD12 HOME value as Tyre Diameter.
# If PLC team gives a separate tyre diameter tag later,
# change only these two constants.
TYRE_ROTATION_DIAMETER_DB75_BYTE = 264
TYRE_ROTATION_DIAMETER_DB53_BYTE = 520 


def t(
    key,
    legacy_key,
    axis_id,
    sd,
    description,
    group,
    position,
    db75_byte,
    db53_byte,
):
    return {
        "key": key,
        "legacy_key": legacy_key,

        "axis_id": axis_id,
        "sd": sd,
        "description": description,
        "group": group,
        "position": position,

        "db75_db": 75,
        "db75_byte": db75_byte,
        "db75_type": "REAL",

        "db53_db": 53,
        "db53_byte": db53_byte,
        "db53_type": "REAL",
    }


RECIPE_TARGETS = [
    # =====================================================
    # SD1 - BEAD LOCKING LH
    # DB75: 0,4,8
    # DB53: 256,260,264
    # Existing app axis 1 name is BED LOCKING RH.
    # PLC team confirmed SD number mapping is correct.
    # =====================================================
    t("sd1_bead_locking_lh_home", None, 1, "SD1", "BEAD LOCKING LH", "MACHINE", "HOME", 0, 256),
    t("sd1_bead_locking_lh_work1", "bed_locking_rh", 1, "SD1", "BEAD LOCKING LH", "MACHINE", "WORK 1", 4, 260),
    t("sd1_bead_locking_lh_work2", None, 1, "SD1", "BEAD LOCKING LH", "MACHINE", "WORK 2", 8, 264),

    # =====================================================
    # SD2 - BEAD LOCKING RH
    # =====================================================
    t("sd2_bead_locking_rh_home", None, 2, "SD2", "BEAD LOCKING RH", "MACHINE", "HOME", 24, 280),
    t("sd2_bead_locking_rh_work1", "bed_locking_lh", 2, "SD2", "BEAD LOCKING RH", "MACHINE", "WORK 1", 28, 284),
    t("sd2_bead_locking_rh_work2", None, 2, "SD2", "BEAD LOCKING RH", "MACHINE", "WORK 2", 32, 288),

    # =====================================================
    # SD3 - SW2 CAMERA UP/DOWN
    # =====================================================
    t("sd3_sw2_camera_up_down_home", None, 3, "SD3", "SW2 CAMERA UP/DOWN", "CAMERA", "HOME", 48, 304),
    t("sd3_sw2_camera_up_down_work1", "sidewall2_camera_up_down", 3, "SD3", "SW2 CAMERA UP/DOWN", "CAMERA", "WORK 1", 52, 308),
    t("sd3_sw2_camera_up_down_work2", "sidewall2_laser_up_down", 3, "SD3", "SW2 CAMERA UP/DOWN", "LASER", "WORK 2", 56, 312),

    # =====================================================
    # SD4 - INNER CAMERA FWD/REV
    # Needs HOME, WORK1, WORK2, WORK3
    # =====================================================
    t("sd4_inner_camera_fwd_rev_home", None, 4, "SD4", "INNER CAMERA FWD/REV", "CAMERA", "HOME", 72, 328),
    t("sd4_inner_camera_fwd_rev_work1", "inner_camera_front_back", 4, "SD4", "INNER CAMERA FWD/REV", "CAMERA", "WORK 1", 76, 332),
    t("sd4_inner_camera_fwd_rev_work2", None, 4, "SD4", "INNER CAMERA FWD/REV", "CAMERA", "WORK 2", 80, 336),
    t("sd4_inner_camera_fwd_rev_work3", None, 4, "SD4", "INNER CAMERA FWD/REV", "CAMERA", "WORK 3", 84, 340),

    # =====================================================
    # SD5 - SW1 CAMERA IN/OUT
    # =====================================================
    t("sd5_sw1_camera_in_out_home", None, 5, "SD5", "SW1 CAMERA IN/OUT", "CAMERA", "HOME", 96, 352),
    t("sd5_sw1_camera_in_out_work1", "sidewall1_camera_fwd_rev", 5, "SD5", "SW1 CAMERA IN/OUT", "CAMERA", "WORK 1", 100, 356),
    t("sd5_sw1_camera_in_out_work2", "sidewall1_laser_fwd_rev", 5, "SD5", "SW1 CAMERA IN/OUT", "LASER", "WORK 2", 104, 360),

    # =====================================================
    # SD6 - SW1 CAMERA UP/DOWN
    # =====================================================
    t("sd6_sw1_camera_up_down_home", None, 6, "SD6", "SW1 CAMERA UP/DOWN", "CAMERA", "HOME", 120, 376),
    t("sd6_sw1_camera_up_down_work1", "sidewall1_camera_up_down", 6, "SD6", "SW1 CAMERA UP/DOWN", "CAMERA", "WORK 1", 124, 380),
    t("sd6_sw1_camera_up_down_work2", "sidewall1_laser_up_down", 6, "SD6", "SW1 CAMERA UP/DOWN", "LASER", "WORK 2", 128, 384),

    # =====================================================
    # SD7 - SW2 CAMERA IN/OUT
    # =====================================================
    t("sd7_sw2_camera_in_out_home", None, 7, "SD7", "SW2 CAMERA IN/OUT", "CAMERA", "HOME", 144, 400),
    t("sd7_sw2_camera_in_out_work1", "sidewall2_camera_fwd_rev", 7, "SD7", "SW2 CAMERA IN/OUT", "CAMERA", "WORK 1", 148, 404),
    t("sd7_sw2_camera_in_out_work2", "sidewall2_laser_fwd_rev", 7, "SD7", "SW2 CAMERA IN/OUT", "LASER", "WORK 2", 152, 408),

    # =====================================================
    # SD8 - INNER CAMERA IN/OUT
    # =====================================================
    t("sd8_inner_camera_in_out_home", None, 8, "SD8", "INNER CAMERA IN/OUT", "CAMERA", "HOME", 168, 424),
    t("sd8_inner_camera_in_out_work1", "inner_camera_in_out", 8, "SD8", "INNER CAMERA IN/OUT", "CAMERA", "WORK 1", 172, 428),
    t("sd8_inner_camera_in_out_work2", None, 8, "SD8", "INNER CAMERA IN/OUT", "CAMERA", "WORK 2", 176, 432),

    # =====================================================
    # SD9 - TREAD CAMERA UP/DOWN
    # =====================================================
    t("sd9_tread_camera_up_down_home", None, 9, "SD9", "TREAD CAMERA UP/DOWN", "CAMERA", "HOME", 192, 448),
    t("sd9_tread_camera_up_down_work1", "tread_camera_up_down", 9, "SD9", "TREAD CAMERA UP/DOWN", "CAMERA", "WORK 1", 196, 452),
    t("sd9_tread_camera_up_down_work2", "tread_laser_up_down", 9, "SD9", "TREAD CAMERA UP/DOWN", "LASER", "WORK 2", 200, 456),

    # =====================================================
    # SD10 - TYRE CENTERING
    # =====================================================
    t("sd10_tyre_centering_home", None, 10, "SD10", "TYRE CENTERING", "MACHINE", "HOME", 216, 472),
    t("sd10_tyre_centering_work1", "tyre_centring", 10, "SD10", "TYRE CENTERING", "MACHINE", "WORK 1", 220, 476),
    t("sd10_tyre_centering_work2", None, 10, "SD10", "TYRE CENTERING", "MACHINE", "WORK 2", 224, 480),

    # =====================================================
    # SD11 - INNER CAMERA UP/DOWN
    # =====================================================
    t("sd11_inner_camera_up_down_home", None, 11, "SD11", "INNER CAMERA UP/DOWN", "CAMERA", "HOME", 240, 496),
    t("sd11_inner_camera_up_down_work1", "inner_camera_up_down", 11, "SD11", "INNER CAMERA UP/DOWN", "CAMERA", "WORK 1", 244, 500),
    t("sd11_inner_camera_up_down_work2", None, 11, "SD11", "INNER CAMERA UP/DOWN", "CAMERA", "WORK 2", 248, 504),

    # =====================================================
    # SD12 - TYRE ROTATION
    # PLC confirmed:
    #   TYRE ROTATION DIAMETER = DB75.DBD264
    #   TYRE ROTATION          = DB75.DBD268
    #   WORK 2                 = DB75.DBD276
    # =====================================================
    t(
        "sd12_tyre_rotation_diameter",
        None,
        12,
        "SD12",
        "TYRE ROTATION",
        "MACHINE",
        "TYRE ROTATION DIAMETER",
        TYRE_ROTATION_DIAMETER_DB75_BYTE,
        TYRE_ROTATION_DIAMETER_DB53_BYTE,
    ),
    t(
        "sd12_tyre_rotation",
        "tyre_rotation",
        12,
        "SD12",
        "TYRE ROTATION",
        "MACHINE",
        "TYRE ROTATION",
        268,
        524,
    ),
    t(
        "sd12_tyre_rotation_work2",
        None,
        12,
        "SD12",
        "TYRE ROTATION",
        "MACHINE",
        "WORK 2",
        276,
        528,
    ),
]