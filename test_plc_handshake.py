#!/usr/bin/env python3
"""
test_plc_handshake.py

PLC handshake + recipe + tyre RPM test.

Tests:
1) Read machine mode       DB74.DBW74 INT
2) Read active recipe      DB74.DBW78 INT
3) Read recipe list        DB74.DBW80 INT
4) Read tyre RPM           DB74.DBD82 REAL
5) Write APP_OK            DB74.DBX0.0 BOOL
6) Write ACCEPT            DB74.DBX0.2 BOOL
7) Write REJECT            DB74.DBX0.3 BOOL
8) Write recipe entry no   DB75.DBW288 INT
"""

import time
from datetime import datetime

try:
    import snap7
    from snap7.util import get_int, get_word, get_real, set_int
except Exception as e:
    raise SystemExit(
        "snap7 is not installed.\n"
        "Install using:\n"
        "    pip install python-snap7\n\n"
        f"Original error: {e}"
    )


PLC_IP = "192.168.10.1"
PLC_RACK = 0
PLC_SLOT = 1


# =========================================================
# TEST FLAGS
# =========================================================
TEST_APP_OK = True
TEST_ACCEPT_REJECT = True
TEST_RECIPE_ENTRY_WRITE = False
TEST_RECIPE_LIST_READ = True
TEST_TYRE_RPM_READ = True


# =========================================================
# PLC READ TAGS
# =========================================================
PLC_MODE_DB = 74
PLC_MODE_BYTE = 74
PLC_MODE_TYPE = "INT"

ACTIVE_RECIPE_DB = 74
ACTIVE_RECIPE_BYTE = 78
ACTIVE_RECIPE_TYPE = "INT"

RECIPE_LIST_DB = 74
RECIPE_LIST_BYTE = 80
RECIPE_LIST_TYPE = "INT"

TYRE_RPM_DB = 74
TYRE_RPM_BYTE = 82
TYRE_RPM_TYPE = "REAL"


# =========================================================
# PLC WRITE BOOL TAGS
# =========================================================
APP_OK_DB = 74
APP_OK_BYTE = 0
APP_OK_BIT = 0

PLC_ACCEPT_DB = 74
PLC_ACCEPT_BYTE = 0
PLC_ACCEPT_BIT = 2

PLC_REJECT_DB = 74
PLC_REJECT_BYTE = 0
PLC_REJECT_BIT = 3


# =========================================================
# PLC WRITE INT TAG
# =========================================================
RECIPE_ENTRY_DB = 75
RECIPE_ENTRY_BYTE = 288
RECIPE_ENTRY_TYPE = "INT"


MODE_MAP = {
    0: "UNKNOWN",
    1: "MANUAL",
    2: "TEACHING",
    3: "AUTO",
    4: "FAULT",
}


def now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def bit_address(db, byte, bit):
    return f"DB{db}.DBX{byte}.{bit}"


def word_address(db, byte):
    return f"DB{db}.DBW{byte}"


def real_address(db, byte):
    return f"DB{db}.DBD{byte}"


# =========================================================
# BOOL HELPERS
# =========================================================
def read_db_bit(client, db_number, byte_index, bit_index):
    data = client.db_read(db_number, byte_index, 1)
    return bool(data[0] & (1 << bit_index))


def write_db_bit(client, db_number, byte_index, bit_index, value):
    data = client.db_read(db_number, byte_index, 1)
    byte_val = data[0]

    if value:
        byte_val = byte_val | (1 << bit_index)
    else:
        byte_val = byte_val & ~(1 << bit_index)

    client.db_write(db_number, byte_index, bytes([byte_val]))


def set_and_verify_bit(client, db, byte, bit, value, label):
    addr = bit_address(db, byte, bit)

    write_db_bit(client, db, byte, bit, value)
    time.sleep(0.2)

    read_back = read_db_bit(client, db, byte, bit)

    print(f"[{now()}] Wrote {label} = {value} at {addr}")
    print(f"[{now()}] {label} read-back: {read_back}")

    if read_back == value:
        print(f"[{now()}] {label} write/read-back OK")
        return True

    print(f"[{now()}] {label} write/read-back FAILED")
    return False


def test_bit_with_plc_confirmation(client, db, byte, bit, label):
    addr = bit_address(db, byte, bit)

    print("\n" + "-" * 80)
    print(f"TEST: {label}")
    print("-" * 80)

    set_and_verify_bit(client, db, byte, bit, False, label)

    print(f"\nAbout to set {label} TRUE at {addr}.")
    input("Press ENTER when PLC team is ready to observe this bit...")

    ok_true = set_and_verify_bit(client, db, byte, bit, True, label)

    print(
        f"\nAsk PLC team to confirm {addr} is TRUE now.\n"
        f"Press ENTER after PLC team confirms. Then this script will reset {label} to FALSE."
    )
    input()

    ok_false = set_and_verify_bit(client, db, byte, bit, False, label)

    return ok_true and ok_false


# =========================================================
# INT / WORD / REAL HELPERS
# =========================================================
def read_db_int(client, db_number, byte_index):
    data = client.db_read(db_number, byte_index, 2)
    return int(get_int(data, 0))


def write_db_int(client, db_number, byte_index, value):
    data = bytearray(2)
    set_int(data, 0, int(value))
    client.db_write(db_number, byte_index, data)


def read_db_real(client, db_number, byte_index):
    data = client.db_read(db_number, byte_index, 4)
    return float(get_real(data, 0))


def read_plc_int_or_word(client, db, byte, data_type):
    data = client.db_read(db, byte, 2)

    if data_type.upper() == "INT":
        return int(get_int(data, 0))

    if data_type.upper() == "WORD":
        return int(get_word(data, 0))

    raise ValueError(f"Unsupported type={data_type}. Use INT or WORD.")


def read_mode_value(client):
    return read_plc_int_or_word(client, PLC_MODE_DB, PLC_MODE_BYTE, PLC_MODE_TYPE)


def read_active_recipe_value(client):
    return read_plc_int_or_word(
        client,
        ACTIVE_RECIPE_DB,
        ACTIVE_RECIPE_BYTE,
        ACTIVE_RECIPE_TYPE,
    )


def read_recipe_list_value(client):
    return read_plc_int_or_word(
        client,
        RECIPE_LIST_DB,
        RECIPE_LIST_BYTE,
        RECIPE_LIST_TYPE,
    )


def read_tyre_rpm_value(client):
    return read_db_real(client, TYRE_RPM_DB, TYRE_RPM_BYTE)


# =========================================================
# RECIPE ENTRY WRITE TEST
# =========================================================
def test_recipe_entry_write(client):
    addr = word_address(RECIPE_ENTRY_DB, RECIPE_ENTRY_BYTE)

    print("\n" + "=" * 80)
    print("TEST: WRITE RECIPE ENTRY NUMBER TO PLC")
    print("=" * 80)
    print(f"Address   : {addr}")
    print(f"Data Type : {RECIPE_ENTRY_TYPE}")
    print("=" * 80)

    try:
        current_value = read_db_int(client, RECIPE_ENTRY_DB, RECIPE_ENTRY_BYTE)
        print(f"[{now()}] Current value at {addr}: {current_value}")
    except Exception as e:
        print(f"[{now()}] WARNING: Could not read current value at {addr}: {e}")

    recipe_no_text = input("Enter recipe number to write to PLC DB75.DBW288: ").strip()

    if not recipe_no_text:
        print("No recipe number entered. Skipping recipe entry write.")
        return False

    try:
        recipe_no = int(recipe_no_text)
    except ValueError:
        print(f"Invalid recipe number: {recipe_no_text}")
        return False

    print(f"\nAbout to write recipe number {recipe_no} to {addr}.")
    input("Press ENTER when PLC team is ready to observe DB75.DBW288...")

    write_db_int(client, RECIPE_ENTRY_DB, RECIPE_ENTRY_BYTE, recipe_no)
    time.sleep(0.2)

    read_back = read_db_int(client, RECIPE_ENTRY_DB, RECIPE_ENTRY_BYTE)

    print(f"[{now()}] Wrote Recipe Entry Number = {recipe_no} at {addr}")
    print(f"[{now()}] Recipe Entry read-back   = {read_back}")

    if read_back == recipe_no:
        print(f"[{now()}] Recipe Entry write/read-back OK")
        return True

    print(f"[{now()}] Recipe Entry write/read-back FAILED")
    return False


def main():
    print("=" * 80)
    print("PLC HANDSHAKE + RECIPE + TYRE RPM TEST")
    print("=" * 80)
    print(f"PLC IP             : {PLC_IP}")
    print(f"PLC Rack/Slot      : {PLC_RACK}/{PLC_SLOT}")
    print(f"Mode Address       : {word_address(PLC_MODE_DB, PLC_MODE_BYTE)} ({PLC_MODE_TYPE})")
    print(f"Active Recipe      : {word_address(ACTIVE_RECIPE_DB, ACTIVE_RECIPE_BYTE)} ({ACTIVE_RECIPE_TYPE})")
    print(f"Recipe List        : {word_address(RECIPE_LIST_DB, RECIPE_LIST_BYTE)} ({RECIPE_LIST_TYPE})")
    print(f"Tyre RPM           : {real_address(TYRE_RPM_DB, TYRE_RPM_BYTE)} ({TYRE_RPM_TYPE})")
    print(f"App OK Address     : {bit_address(APP_OK_DB, APP_OK_BYTE, APP_OK_BIT)}")
    print(f"Accept Address     : {bit_address(PLC_ACCEPT_DB, PLC_ACCEPT_BYTE, PLC_ACCEPT_BIT)}")
    print(f"Reject Address     : {bit_address(PLC_REJECT_DB, PLC_REJECT_BYTE, PLC_REJECT_BIT)}")
    print(f"Recipe Entry Write : {word_address(RECIPE_ENTRY_DB, RECIPE_ENTRY_BYTE)} ({RECIPE_ENTRY_TYPE})")
    print("=" * 80)

    client = snap7.client.Client()

    app_ok_result = None
    accept_ok = None
    reject_ok = None
    recipe_entry_ok = None

    active_recipe_value = None
    recipe_list_value = None
    tyre_rpm_value = None

    recipe_list_ok = False
    tyre_rpm_ok = False

    try:
        print(f"[{now()}] Connecting to PLC...")
        client.connect(PLC_IP, PLC_RACK, PLC_SLOT)

        if not client.get_connected():
            raise RuntimeError("snap7 client is not connected after connect()")

        print(f"[{now()}] PLC connected successfully")

        print("\n--- TEST 1: READ MACHINE MODE FROM PLC ---")
        mode_value = read_mode_value(client)
        mode_text = MODE_MAP.get(mode_value, f"UNKNOWN_VALUE_{mode_value}")

        print(f"[{now()}] Mode raw value : {mode_value}")
        print(f"[{now()}] Mode decoded   : {mode_text}")

        print("\n--- TEST 1B: READ ACTIVE RECIPE FROM PLC ---")
        active_recipe_value = read_active_recipe_value(client)
        print(f"[{now()}] Active Recipe raw value : {active_recipe_value}")

        if active_recipe_value <= 0:
            print(f"[{now()}] WARNING: Active recipe value is {active_recipe_value}")
        else:
            print(f"[{now()}] Active recipe read OK")

        if TEST_RECIPE_LIST_READ:
            print("\n--- TEST 1C: READ RECIPE LIST FROM PLC ---")
            recipe_list_value = read_recipe_list_value(client)

            print(f"[{now()}] Recipe List raw value : {recipe_list_value}")

            if recipe_list_value <= 0:
                print(
                    f"[{now()}] WARNING: Recipe list value is {recipe_list_value}. "
                    "Check PLC recipe list mapping."
                )
                recipe_list_ok = False
            else:
                print(f"[{now()}] Recipe list read OK")
                recipe_list_ok = True

        if TEST_TYRE_RPM_READ:
            print("\n--- TEST 1D: READ TYRE RPM FROM PLC ---")
            tyre_rpm_value = read_tyre_rpm_value(client)

            print(f"[{now()}] Tyre RPM raw value : {tyre_rpm_value:.6f}")

            if tyre_rpm_value < 0:
                print(f"[{now()}] WARNING: Tyre RPM is negative. Check PLC scaling.")
                tyre_rpm_ok = False
            else:
                print(f"[{now()}] Tyre RPM read OK")
                tyre_rpm_ok = True

        if TEST_APP_OK:
            print("\n--- TEST 2: WRITE APP_OK TRUE TO PLC ---")

            set_and_verify_bit(client, APP_OK_DB, APP_OK_BYTE, APP_OK_BIT, False, "APP_OK")

            input("Press ENTER when PLC team is ready to observe APP_OK...")

            app_ok_true = set_and_verify_bit(
                client,
                APP_OK_DB,
                APP_OK_BYTE,
                APP_OK_BIT,
                True,
                "APP_OK",
            )

            print(
                f"\nAsk PLC team to check that {bit_address(APP_OK_DB, APP_OK_BYTE, APP_OK_BIT)} is TRUE now.\n"
                "Press ENTER after PLC team confirms. Then this script will reset APP_OK to FALSE."
            )
            input()

            app_ok_false = set_and_verify_bit(
                client,
                APP_OK_DB,
                APP_OK_BYTE,
                APP_OK_BIT,
                False,
                "APP_OK",
            )

            app_ok_result = app_ok_true and app_ok_false

        if TEST_ACCEPT_REJECT:
            print("\n--- SAFETY RESET BEFORE ACCEPT / REJECT ---")
            set_and_verify_bit(client, PLC_ACCEPT_DB, PLC_ACCEPT_BYTE, PLC_ACCEPT_BIT, False, "ACCEPT")
            set_and_verify_bit(client, PLC_REJECT_DB, PLC_REJECT_BYTE, PLC_REJECT_BIT, False, "REJECT")

            accept_ok = test_bit_with_plc_confirmation(
                client,
                PLC_ACCEPT_DB,
                PLC_ACCEPT_BYTE,
                PLC_ACCEPT_BIT,
                "ACCEPT",
            )

            reject_ok = test_bit_with_plc_confirmation(
                client,
                PLC_REJECT_DB,
                PLC_REJECT_BYTE,
                PLC_REJECT_BIT,
                "REJECT",
            )

            print("\n--- FINAL ACCEPT / REJECT SAFETY RESET ---")
            set_and_verify_bit(client, PLC_ACCEPT_DB, PLC_ACCEPT_BYTE, PLC_ACCEPT_BIT, False, "ACCEPT")
            set_and_verify_bit(client, PLC_REJECT_DB, PLC_REJECT_BYTE, PLC_REJECT_BIT, False, "REJECT")

        if TEST_RECIPE_ENTRY_WRITE:
            recipe_entry_ok = test_recipe_entry_write(client)

        print("\n" + "=" * 80)
        print("PLC TEST SUMMARY")
        print("=" * 80)
        print("Machine Mode Read     : OK")
        print(f"Active Recipe Read    : {'OK' if active_recipe_value and active_recipe_value > 0 else 'FAILED / ZERO'}")
        print(f"Active Recipe No      : {active_recipe_value}")

        if TEST_RECIPE_LIST_READ:
            print(f"Recipe List Read      : {'OK' if recipe_list_ok else 'FAILED / ZERO'}")
            print(f"Recipe List Value     : {recipe_list_value}")
        else:
            print("Recipe List Read      : SKIPPED")

        if TEST_TYRE_RPM_READ:
            print(f"Tyre RPM Read         : {'OK' if tyre_rpm_ok else 'FAILED'}")
            print(f"Tyre RPM Value        : {tyre_rpm_value}")
        else:
            print("Tyre RPM Read         : SKIPPED")

        if TEST_APP_OK:
            print(f"APP_OK Test           : {'OK' if app_ok_result else 'FAILED'}")
        else:
            print("APP_OK Test           : SKIPPED")

        if TEST_ACCEPT_REJECT:
            print(f"ACCEPT Test           : {'OK' if accept_ok else 'FAILED'}")
            print(f"REJECT Test           : {'OK' if reject_ok else 'FAILED'}")
        else:
            print("ACCEPT Test           : SKIPPED")
            print("REJECT Test           : SKIPPED")

        if TEST_RECIPE_ENTRY_WRITE:
            print(f"Recipe Entry Write    : {'OK' if recipe_entry_ok else 'FAILED'}")
        else:
            print("Recipe Entry Write    : SKIPPED")

        print("=" * 80)

    except Exception as e:
        print("\n" + "=" * 80)
        print("PLC TEST FAILED")
        print("=" * 80)
        print(f"Error: {e}")

    finally:
        try:
            if client.get_connected():
                try:
                    write_db_bit(client, APP_OK_DB, APP_OK_BYTE, APP_OK_BIT, False)
                    write_db_bit(client, PLC_ACCEPT_DB, PLC_ACCEPT_BYTE, PLC_ACCEPT_BIT, False)
                    write_db_bit(client, PLC_REJECT_DB, PLC_REJECT_BYTE, PLC_REJECT_BIT, False)
                except Exception:
                    pass

                client.disconnect()
                print(f"[{now()}] PLC disconnected")
        except Exception:
            pass


if __name__ == "__main__":
    main()