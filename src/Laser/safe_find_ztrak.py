import os
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

from DALSA.SaperaLT.SapClassBasic import SapManager, SapManagerBase

print("\n[OK] Sapera SDK loaded")

# Detect servers
try:
    ok1 = SapManager.DetectAllServers(SapManagerBase.DetectServerType.GenCP)
    print("DetectAllServers(GenCP) =", ok1)
except Exception as e:
    print("[WARN] GenCP detect failed:", e)

try:
    ok2 = SapManager.DetectAllServers(SapManagerBase.DetectServerType.All)
    print("DetectAllServers(All) =", ok2)
except Exception as e:
    print("[WARN] All detect failed:", e)

server_count = SapManager.GetServerCount()
print("\n[SERVER LIST]")
print("Server count:", server_count)

DEVICE_INFO_KEYS = [
    "DeviceVendorName",
    "DeviceModelName",
    "DeviceID",
    "DeviceUserID",
    "DeviceSerialNumber",
    "DeviceVersion",
    "DeviceManufacturerInfo",
    "GevCurrentIPAddress",
    "GevCurrentSubnetMask",
    "GevCurrentDefaultGateway",
]

for server_idx in range(server_count):
    print("\n" + "=" * 80)
    print("SERVER INDEX:", server_idx)

    server_name = None
    server_type = None
    accessible = False

    try:
        server_name = SapManager.GetServerName(server_idx)
        print("Server name:", server_name)
    except Exception as e:
        print("Server name error:", e)

    try:
        server_type = SapManager.GetServerType(server_idx)
        print("Server type:", server_type)
    except Exception as e:
        print("Server type error:", e)

    try:
        accessible = SapManager.IsServerAccessible(server_idx)
        print("Is accessible:", accessible)
    except Exception as e:
        print("Accessibility check error:", e)

    # Skip fake/system/non-device server
    if not accessible or str(server_name).lower() == "system" or str(server_type).lower() == "none":
        print("[SKIP] Not a real accessible device server")
        continue

    print("\n[DEVICE INFO]")
    for key in DEVICE_INFO_KEYS:
        try:
            val = SapManager.ReadDeviceInfoValue(server_idx, key)
            if val is not None and str(val).strip():
                print(f"{key}: {val}")
        except Exception as e:
            print(f"{key}: <read failed: {e}>")

    print("\n[RESOURCES]")
    try:
        for res_name in System.Enum.GetNames(SapManagerBase.ResourceType):
            try:
                res_type = System.Enum.Parse(SapManagerBase.ResourceType, res_name)
                res_count = SapManager.GetResourceCount(server_idx, res_type)

                if res_count > 0:
                    print(f"ResourceType {res_name} count = {res_count}")

                    for res_idx in range(res_count):
                        try:
                            rname = SapManager.GetResourceName(server_idx, res_type, res_idx)
                        except Exception as e:
                            rname = f"<name read error: {e}>"

                        try:
                            available = SapManager.IsResourceAvailable(server_idx, res_type, res_idx)
                        except Exception as e:
                            available = f"<availability error: {e}>"

                        print(f"  Resource index {res_idx}: {rname} | available={available}")

            except Exception:
                pass
    except Exception as e:
        print("Resource listing error:", e)

print("\n[DONE] Safe Z-Trak server/resource listing completed")