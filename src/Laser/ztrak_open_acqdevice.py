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

from DALSA.SaperaLT.SapClassBasic import (
    SapManager,
    SapManagerBase,
    SapLocation,
    SapAcqDevice,
)

print("\n[OK] Sapera SDK loaded")

# Detect servers
SapManager.DetectAllServers(SapManagerBase.DetectServerType.GenCP)
SapManager.DetectAllServers(SapManagerBase.DetectServerType.All)

server_count = SapManager.GetServerCount()
print("[INFO] Server count:", server_count)

target_server_index = None
target_server_name = None
target_resource_index = None
target_resource_name = None

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
            print("[SKIP] Not accessible")
            continue

        acqdev_count = SapManager.GetResourceCount(
            server_idx,
            SapManagerBase.ResourceType.AcqDevice
        )

        print("AcqDevice count:", acqdev_count)

        if acqdev_count > 0:
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
                    target_server_index = server_idx
                    target_server_name = server_name
                    target_resource_index = res_idx
                    target_resource_name = res_name
                    break

        if target_server_name is not None:
            break

    except Exception as e:
        print("[WARN] Server scan error:", e)

if target_server_name is None:
    raise RuntimeError("No available Z-Trak AcqDevice resource found.")

print("\n[SELECTED DEVICE]")
print("Server index   :", target_server_index)
print("Server name    :", target_server_name)
print("Resource index :", target_resource_index)
print("Resource name  :", target_resource_name)

# Create SapLocation
location = SapLocation(target_server_name, target_resource_index)
print("\n[OK] SapLocation created")

# Try creating SapAcqDevice
acq_device = None

try:
    acq_device = SapAcqDevice(location)
    print("[OK] SapAcqDevice object created using SapAcqDevice(location)")
except Exception as e1:
    print("[WARN] SapAcqDevice(location) failed:", e1)

    try:
        acq_device = SapAcqDevice(location, "")
        print("[OK] SapAcqDevice object created using SapAcqDevice(location, '')")
    except Exception as e2:
        print("[ERROR] SapAcqDevice creation failed:", e2)

        print("\n[AVAILABLE SapAcqDevice CONSTRUCTORS]")
        try:
            t = clr.GetClrType(SapAcqDevice)
            for c in t.GetConstructors():
                print(c)
        except Exception as e3:
            print("Could not list constructors:", e3)

        raise

# Create hardware object
print("\n[CREATE DEVICE]")
created = acq_device.Create()
print("Create() returned:", created)

if not created:
    raise RuntimeError("SapAcqDevice.Create() failed")

print("\n[SapAcqDevice PUBLIC METHODS - useful names]")
try:
    for m in acq_device.GetType().GetMethods():
        name = m.Name
        if any(k in name.lower() for k in [
            "feature", "parameter", "value", "create", "destroy",
            "get", "set", "acq", "grab", "start", "stop"
        ]):
            print(m)
except Exception as e:
    print("[WARN] Method listing failed:", e)

print("\n[SUCCESS] Z-Trak AcqDevice opened successfully from Python")

# Clean shutdown
try:
    acq_device.Destroy()
    print("[OK] AcqDevice destroyed/closed")
except Exception as e:
    print("[WARN] Destroy failed:", e)