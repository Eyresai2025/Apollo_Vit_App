from arena_api.system import system

SERIAL = "250500042"
NEW_IP = "192.168.1.22"
NEW_SUBNET = "255.255.255.0"
NEW_GATEWAY = "0.0.0.0"

target = None

for info in system.device_infos:
    print(info)
    if str(info.get("serial")) == SERIAL:
        target = info.copy()

if target is None:
    print(f"Target camera not found: {SERIAL}")
    exit()

target["ip"] = NEW_IP
target["subnetmask"] = NEW_SUBNET
target["defaultgateway"] = NEW_GATEWAY
target["dhcp"] = False
target["persistentip"] = True

print("\nForce IP target:")
print(target)

system.force_ip(target)

print(f"\nForce IP done: {SERIAL} -> {NEW_IP}")
print("Now wait 5 seconds, then ping the camera.")