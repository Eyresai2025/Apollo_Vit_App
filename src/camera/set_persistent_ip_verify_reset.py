from arena_api.system import system
import socket
import struct
import time

SERIAL = "250500042"
NEW_IP = "192.168.1.22"
NEW_SUBNET = "255.255.255.0"
NEW_GATEWAY = "0.0.0.0"


def ip_to_int(ip):
    return struct.unpack("!I", socket.inet_aton(ip))[0]


def int_to_ip(value):
    try:
        return socket.inet_ntoa(struct.pack("!I", int(value)))
    except Exception:
        return str(value)


def get_ip_node(nm, node_name):
    try:
        return int_to_ip(nm[node_name].value)
    except Exception as e:
        return f"READ_FAILED: {e}"


def set_bool_node(nm, node_name, value):
    try:
        nm[node_name].value = value
        print(f"[OK] {node_name} = {value}")
    except Exception as e:
        print(f"[SKIP/FAILED] {node_name}: {e}")


target = None

print("=" * 100)
print("Searching camera...")
print("=" * 100)

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

    print("\nCamera opened successfully")
    print("Serial:", nm["DeviceSerialNumber"].value)
    print("Model :", nm["DeviceModelName"].value)

    print("\nBEFORE SETTING:")
    print("Current IP     :", get_ip_node(nm, "GevCurrentIPAddress"))
    print("Persistent IP  :", get_ip_node(nm, "GevPersistentIPAddress"))
    print("Persistent Mask:", get_ip_node(nm, "GevPersistentSubnetMask"))
    print("Persistent GW  :", get_ip_node(nm, "GevPersistentDefaultGateway"))

    print("\nSETTING NEW PERSISTENT IP...")
    nm["GevPersistentIPAddress"].value = ip_to_int(NEW_IP)
    nm["GevPersistentSubnetMask"].value = ip_to_int(NEW_SUBNET)
    nm["GevPersistentDefaultGateway"].value = ip_to_int(NEW_GATEWAY)

    set_bool_node(nm, "GevCurrentIPConfigurationDHCP", False)
    set_bool_node(nm, "GevCurrentIPConfigurationLLA", False)
    set_bool_node(nm, "GevCurrentIPConfigurationPersistentIP", True)

    time.sleep(1)

    print("\nAFTER SETTING / READBACK:")
    print("Persistent IP  :", get_ip_node(nm, "GevPersistentIPAddress"))
    print("Persistent Mask:", get_ip_node(nm, "GevPersistentSubnetMask"))
    print("Persistent GW  :", get_ip_node(nm, "GevPersistentDefaultGateway"))

    readback_ip = get_ip_node(nm, "GevPersistentIPAddress")

    if readback_ip != NEW_IP:
        print("\n[FAILED] Persistent IP readback is not matching.")
        print(f"Expected: {NEW_IP}")
        print(f"Readback: {readback_ip}")
        print("Do not power cycle yet. This means node write is not taking correctly.")
    else:
        print("\n[OK] Persistent IP readback matched.")

        print("\nTrying camera DeviceReset...")
        try:
            nm["DeviceReset"].execute()
            print("[OK] DeviceReset command sent.")
            print("Camera will disconnect/reboot now.")
        except Exception as e:
            print("[WARNING] DeviceReset failed or not available:", e)
            print("Manually power cycle the camera PoE cable.")

except Exception as e:
    print("\nFAILED")
    print("Reason:", e)

finally:
    if cam is not None:
        try:
            system.destroy_device(cam)
        except Exception:
            pass

print("\nWait 30 seconds, then run:")
print("ping 192.168.3.20")
print("python camera_check.py")