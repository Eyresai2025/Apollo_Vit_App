from arena_api.system import system


def short_error(e):
    lines = str(e).splitlines()
    for line in lines:
        line = line.strip()
        if line:
            return line
    return str(e)


def main():
    print("=" * 100)
    print("AVAILABLE ARENA / LUCID CAMERAS - CHECK ONLY")
    print("=" * 100)

    infos = system.device_infos

    if not infos:
        print("No cameras detected.")
        print("=" * 100)
        return

    print(f"Total detected entries: {len(infos)}")

    seen_serials = {}

    for i, info in enumerate(infos, start=1):
        serial = str(info.get("serial"))
        seen_serials[serial] = seen_serials.get(serial, 0) + 1

        print("\n" + "-" * 100)
        print(f"Camera Entry {i}")
        print("-" * 100)

        print(f"Model       : {info.get('model')}")
        print(f"Vendor      : {info.get('vendor')}")
        print(f"Serial      : {info.get('serial')}")
        print(f"IP          : {info.get('ip')}")
        print(f"Subnet      : {info.get('subnetmask')}")
        print(f"Gateway     : {info.get('defaultgateway')}")
        print(f"MAC         : {info.get('mac')}")
        print(f"Version     : {info.get('version')}")
        print(f"DHCP        : {info.get('dhcp')}")
        print(f"Persistent  : {info.get('persistentip')}")
        print(f"LLA         : {info.get('lla')}")

        cam = None

        try:
            devices = system.create_device([info])
            cam = devices[0]

            nodemap = cam.nodemap

            device_serial = nodemap["DeviceSerialNumber"].value
            device_model = nodemap["DeviceModelName"].value

            print("Open Status : OK")
            print(f"Opened As   : {device_model} / {device_serial}")

        except Exception as e:
            print("Open Status : FAILED")
            print("Reason      :", short_error(e))

        finally:
            if cam is not None:
                try:
                    system.destroy_device(cam)
                except Exception:
                    pass

    print("\n" + "=" * 100)
    print("DUPLICATE SERIAL CHECK")
    print("=" * 100)

    duplicates_found = False

    for serial, count in seen_serials.items():
        if count > 1:
            duplicates_found = True
            print(f"Serial {serial} appears {count} times")

    if not duplicates_found:
        print("No duplicate serial entries found.")

    print("\n" + "=" * 100)
    print("Camera check completed. No IP settings were changed.")
    print("=" * 100)


if __name__ == "__main__":
    main()