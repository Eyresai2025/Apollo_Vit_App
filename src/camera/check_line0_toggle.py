from arena_api.system import system
import time


LINE_NAME = "Line0"
CHECK_INTERVAL_SEC = 0.05   # 50 ms
RUN_SECONDS = 60            # monitor for 60 seconds
NUM_CAMERAS_TO_CHECK = 1


def get_node(nodemap, name):
    try:
        return nodemap.get_node(name)
    except Exception:
        return None


def set_node(nodemap, name, value):
    node = get_node(nodemap, name)
    if node is None:
        print(f"[WARN] Node not found: {name}")
        return False

    try:
        if hasattr(node, "is_writable") and not node.is_writable:
            print(f"[WARN] Node not writable: {name}")
            return False

        node.value = value
        print(f"[OK] {name} = {node.value}")
        return True

    except Exception as e:
        print(f"[WARN] Could not set {name} = {value} | {e}")
        return False


def read_node(nodemap, name):
    node = get_node(nodemap, name)
    if node is None:
        return None

    try:
        return node.value
    except Exception:
        return None


def get_camera_name(device, index):
    nodemap = device.nodemap

    serial = read_node(nodemap, "DeviceSerialNumber")
    model = read_node(nodemap, "DeviceModelName")

    if serial or model:
        return f"CAM{index} | Model={model} | Serial={serial}"

    return f"CAM{index}"


def configure_line0_input(device, cam_name):
    nodemap = device.nodemap

    print(f"\n--- Configuring {cam_name} ---")

    set_node(nodemap, "LineSelector", LINE_NAME)
    set_node(nodemap, "LineMode", "Input")

    # For input line, LineSource should normally be Off.
    # Some cameras may not allow writing LineSource for input lines.
    set_node(nodemap, "LineSource", "Off")

    # Do not force LineActivationVoltage here because some cameras show only Low.
    # You can manually test Low/High in ArenaView if available.

    status = read_node(nodemap, "LineStatus")
    print(f"[INIT] {cam_name} {LINE_NAME} LineStatus = {status}")

    return status


def main():
    print("Searching for cameras...")
    devices = system.create_device()

    if not devices:
        print("No cameras found.")
        return

    print(f"Found {len(devices)} camera(s).")

    devices = devices[:NUM_CAMERAS_TO_CHECK]

    cam_infos = []

    for i, device in enumerate(devices):
        cam_name = get_camera_name(device, i)
        initial_status = configure_line0_input(device, cam_name)

        cam_infos.append({
            "device": device,
            "name": cam_name,
            "last_status": initial_status,
            "toggle_count": 0,
        })

    print("\n======================================")
    print("Now press/release the HMI trigger.")
    print("Watching Line0 status changes...")
    print("======================================\n")

    start_time = time.time()

    try:
        while time.time() - start_time < RUN_SECONDS:
            for cam in cam_infos:
                device = cam["device"]
                nodemap = device.nodemap

                # Make sure we are still reading Line0
                try:
                    read_node(nodemap, "LineSelector")
                except Exception:
                    pass

                current_status = read_node(nodemap, "LineStatus")

                if current_status != cam["last_status"]:
                    cam["toggle_count"] += 1

                    now = time.strftime("%H:%M:%S")
                    print(
                        f"[{now}] TOGGLE | {cam['name']} | "
                        f"{cam['last_status']} -> {current_status} | "
                        f"count={cam['toggle_count']}"
                    )

                    cam["last_status"] = current_status

            time.sleep(CHECK_INTERVAL_SEC)

    except KeyboardInterrupt:
        print("\nStopped by user.")

    finally:
        print("\n========== SUMMARY ==========")
        for cam in cam_infos:
            print(
                f"{cam['name']} | Final LineStatus={cam['last_status']} | "
                f"Toggle Count={cam['toggle_count']}"
            )

        system.destroy_device()
        print("Camera devices released.")


if __name__ == "__main__":
    main()