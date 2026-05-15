##Enabler code

# from arena_api.system import system

# SERIAL = "254901432"

# NEW_IP = "192.168.3.22"
# NEW_SUBNET = "255.255.255.0"
# NEW_GATEWAY = "0.0.0.0"

# target = None

# for info in system.device_infos:
#     print(info)
#     if str(info.get("serial")) == SERIAL:
#         target = info.copy()

# if target is None:
#     print("Target camera not found")
#     exit()

# target["ip"] = NEW_IP
# target["subnetmask"] = NEW_SUBNET
# target["defaultgateway"] = NEW_GATEWAY
# target["dhcp"] = False
# target["persistentip"] = True

# print("Force IP target:")
# print(target)

# system.force_ip(target)

# print("Done. Now unplug/plug camera cable and check again.")


####Presisnet ip set
from arena_api.system import system
import socket
import struct

SERIAL = "254901432"
NEW_IP = "192.168.3.22"
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
    print("Target camera not found")
    exit()

devices = system.create_device([target])
cam = devices[0]
nm = cam.nodemap

nm["GevPersistentIPAddress"].value = ip_to_int(NEW_IP)
nm["GevPersistentSubnetMask"].value = ip_to_int(NEW_SUBNET)
nm["GevPersistentDefaultGateway"].value = ip_to_int(NEW_GATEWAY)

nm["GevCurrentIPConfigurationDHCP"].value = False
nm["GevCurrentIPConfigurationPersistentIP"].value = True

print(f"Persistent IP set: {SERIAL} -> {NEW_IP}")

system.destroy_device(cam)
print("Unplug/plug camera cable and check again.")



####Froce ip set
# from arena_api.system import system

# SERIAL = "254901432"

# NEW_IP = "192.168.3.22"
# NEW_SUBNET = "255.255.255.0"
# NEW_GATEWAY = "0.0.0.0"

# target = None

# for info in system.device_infos:
#     print(info)
#     if str(info.get("serial")) == SERIAL:
#         target = info.copy()

# if target is None:
#     print("Target camera not found")
#     exit()

# target["ip"] = NEW_IP
# target["subnetmask"] = NEW_SUBNET
# target["defaultgateway"] = NEW_GATEWAY

# print("Force IP target:")
# print(target)

# system.force_ip(target)

# print("Force IP done. Unplug/plug camera cable and check again.")