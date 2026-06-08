from arena_api.system import system
import socket
import struct
import time

SERIAL = "254901428"
NEW_IP = "192.168.3.20"
NEW_SUBNET = "255.255.255.0"
NEW_GATEWAY = "0.0.0.0"


def ip_to_int(ip):
    return struct.unpack("!I", socket.inet_aton(ip))[0]


target = None

for info in system.device_infos:
    print(info)
    if str(info.get("serial")) == SERIAL:
        target = info

if target is None:
    print(f"Target camera not found: {SERIAL}")
    exit()

cam = None

try:
    devices = system.create_device([target])
    cam = devices[0]
    nm = cam.nodemap

    print("Camera opened successfully")
    print("Serial:", nm["DeviceSerialNumber"].value)
    print("Model :", nm["DeviceModelName"].value)

    print(f"Setting persistent IP: {SERIAL} -> {NEW_IP}")

    nm["GevPersistentIPAddress"].value = ip_to_int(NEW_IP)
    nm["GevPersistentSubnetMask"].value = ip_to_int(NEW_SUBNET)
    nm["GevPersistentDefaultGateway"].value = ip_to_int(NEW_GATEWAY)

    nm["GevCurrentIPConfigurationDHCP"].value = False
    nm["GevCurrentIPConfigurationPersistentIP"].value = True

    time.sleep(0.5)

    print("Persistent IP set successfully.")
    print("Now unplug/plug camera cable or power cycle the camera.")

except Exception as e:
    print("FAILED to set persistent IP")
    print("Reason:", e)

finally:
    if cam is not None:
        system.destroy_device(cam)