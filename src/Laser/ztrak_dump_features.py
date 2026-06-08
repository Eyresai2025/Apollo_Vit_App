import os
import csv
from pathlib import Path

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
)

OUT_DIR = Path(__file__).resolve().parent / "ztrak_output"
OUT_DIR.mkdir(exist_ok=True)

FEATURE_CSV = OUT_DIR / "ztrak_features.csv"
COMMAND_TXT = OUT_DIR / "ztrak_commands.txt"
CATEGORY_TXT = OUT_DIR / "ztrak_categories.txt"
FEATURE_SAVE = OUT_DIR / "ztrak_saved_features.ccf"


def get_prop(obj, prop_name):
    try:
        return getattr(obj, prop_name)
    except Exception:
        return getattr(obj, f"get_{prop_name}")()


def find_first_acqdevice():
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
            print("[WARN] Server scan error:", e)

    raise RuntimeError("No available Z-Trak AcqDevice found.")


def try_get_feature_value(acq_device, feature_name):
    """
    Try multiple Sapera GetFeatureValue overloads.
    Returns: value, type_name, ok
    """

    attempts = [
        ("String", System.String),
        ("Double", System.Double),
        ("Single", System.Single),
        ("Int32", System.Int32),
        ("UInt32", System.UInt32),
        ("Int64", System.Int64),
        ("UInt64", System.UInt64),
        ("Boolean", System.Boolean),
    ]

    for type_name, dotnet_type in attempts:
        try:
            ref_val = clr.Reference[dotnet_type]()
            ok = acq_device.GetFeatureValue(feature_name, ref_val)
            if ok:
                return str(ref_val.Value), type_name, True
        except Exception:
            pass

    return "", "Unreadable", False


def main():
    print("\n[OK] Sapera SDK loaded")

    server_name, resource_index, resource_name = find_first_acqdevice()

    print("\n[SELECTED]")
    print("Server name   :", server_name)
    print("Resource index:", resource_index)
    print("Resource name :", resource_name)

    location = SapLocation(server_name, resource_index)
    acq_device = SapAcqDevice(location)

    try:
        created = acq_device.Create()
        print("\nCreate() returned:", created)

        if not created:
            raise RuntimeError("SapAcqDevice.Create() failed")

        print("\n[UPDATE FEATURES FROM DEVICE]")
        try:
            ok = acq_device.UpdateFeaturesFromDevice()
            print("UpdateFeaturesFromDevice() returned:", ok)
        except Exception as e:
            print("[WARN] UpdateFeaturesFromDevice failed:", e)

        # Save current feature configuration if supported
        try:
            ok = acq_device.SaveFeatures(str(FEATURE_SAVE))
            print("[SAVE FEATURES]", FEATURE_SAVE, "ok=", ok)
        except Exception as e:
            print("[WARN] SaveFeatures failed:", e)

        # Dump feature names and values
        print("\n[DUMP FEATURES]")
        feature_names = list(get_prop(acq_device, "FeatureNames"))
        print("Feature count:", len(feature_names))

        rows = []

        for i, feature_name in enumerate(feature_names):
            value, value_type, ok = try_get_feature_value(acq_device, feature_name)

            available = False
            try:
                available = acq_device.IsFeatureAvailable(feature_name)
            except Exception:
                pass

            rows.append({
                "index": i,
                "feature_name": feature_name,
                "available": available,
                "read_ok": ok,
                "value_type": value_type,
                "value": value,
            })

            print(f"{i:04d} | {feature_name} | available={available} | {value_type} | {value}")

        with open(FEATURE_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["index", "feature_name", "available", "read_ok", "value_type", "value"]
            )
            writer.writeheader()
            writer.writerows(rows)

        print("\n[SAVED]", FEATURE_CSV)

        # Dump command names
        print("\n[DUMP COMMANDS]")
        try:
            command_names = list(get_prop(acq_device, "CommandNames"))
            with open(COMMAND_TXT, "w", encoding="utf-8") as f:
                for cmd in command_names:
                    print("COMMAND:", cmd)
                    f.write(str(cmd) + "\n")
            print("[SAVED]", COMMAND_TXT)
        except Exception as e:
            print("[WARN] Command dump failed:", e)

        # Dump categories
        print("\n[DUMP CATEGORIES]")
        try:
            category_names = list(get_prop(acq_device, "CategoryPathNames"))
            with open(CATEGORY_TXT, "w", encoding="utf-8") as f:
                for cat in category_names:
                    print("CATEGORY:", cat)
                    f.write(str(cat) + "\n")
            print("[SAVED]", CATEGORY_TXT)
        except Exception as e:
            print("[WARN] Category dump failed:", e)

        print("\n[SUCCESS] Feature dump completed")

    finally:
        try:
            acq_device.Destroy()
            print("[OK] AcqDevice destroyed/closed")
        except Exception as e:
            print("[WARN] Destroy failed:", e)


if __name__ == "__main__":
    main()