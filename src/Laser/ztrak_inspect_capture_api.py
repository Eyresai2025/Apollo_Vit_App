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

asm = System.Reflection.Assembly.LoadFile(SAPERA_DOTNET_DLL)

OUT_DIR = Path(__file__).resolve().parent / "ztrak_output"
OUT_DIR.mkdir(exist_ok=True)
OUT_TXT = OUT_DIR / "sapera_capture_api.txt"

KEYWORDS = [
    "SapBuffer",
    "SapAcqDeviceToBuf",
    "SapTransfer",
    "SapXfer",
    "SapView",
    "SapDisplay",
    "SapFormat",
    "SapLocation",
    "SapAcqDevice",
]

METHOD_FILTERS = [
    "Create",
    "Destroy",
    "Grab",
    "Snap",
    "Freeze",
    "Abort",
    "Wait",
    "Get",
    "Set",
    "Save",
    "Copy",
    "Address",
    "Data",
    "Buffer",
    "Xfer",
    "Event",
]

def write_line(f, text=""):
    print(text)
    f.write(text + "\n")

with open(OUT_TXT, "w", encoding="utf-8") as f:
    write_line(f, "[SAPERA CAPTURE API INSPECTION]")
    write_line(f, f"DLL: {SAPERA_DOTNET_DLL}")
    write_line(f)

    types = list(asm.GetTypes())

    for t in types:
        full = str(t.FullName)

        if not any(k.lower() in full.lower() for k in KEYWORDS):
            continue

        write_line(f, "=" * 120)
        write_line(f, "TYPE: " + full)
        write_line(f, "BASE: " + str(t.BaseType))
        write_line(f, "IS ENUM: " + str(t.IsEnum))
        write_line(f)

        if t.IsEnum:
            try:
                write_line(f, "[ENUM NAMES]")
                for name in System.Enum.GetNames(t):
                    write_line(f, f"  {name}")
            except Exception as e:
                write_line(f, f"  <enum read failed: {e}>")
            write_line(f)
            continue

        write_line(f, "[CONSTRUCTORS]")
        try:
            for c in t.GetConstructors():
                write_line(f, "  " + str(c))
        except Exception as e:
            write_line(f, f"  <constructor read failed: {e}>")

        write_line(f)
        write_line(f, "[USEFUL METHODS]")
        try:
            for m in t.GetMethods():
                name = str(m.Name)
                if any(k.lower() in name.lower() for k in METHOD_FILTERS):
                    write_line(f, "  " + str(m))
        except Exception as e:
            write_line(f, f"  <method read failed: {e}>")

        write_line(f)

write_line(open(OUT_TXT, "a", encoding="utf-8"), "\n[DONE]")
print("\n[SAVED]", OUT_TXT)