from arena_api.system import system  # type: ignore
from arena_api.buffer import BufferFactory  # type: ignore
import ctypes
import numpy as np
import time
import warnings
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
 
warnings.filterwarnings("ignore")
 
# ---------------- CAMERA SETTINGS ----------------
HEIGHT = 16384
WIDTH = 4096
EXPOSURE_US = 200.0
GAIN_DB = 24.0
DEFAULT_GAMMA = 1.0
DEFAULT_LINE_RATE_HZ = 4096.178266
PIXEL_FORMAT = "Mono16"
# -------------------------------------------------
 
# Global camera system instance
_camera_system = None
 
def _set_numeric_node(nodemap, node_name, value):
    try:
        node = nodemap.get_node(node_name)
        if node is None:
            print(f"Node '{node_name}' not found")
            return
 
        if isinstance(value, int):
            final_value = int(max(node.min, min(value, node.max)))
        else:
            final_value = float(max(node.min, min(value, node.max)))
 
        node.value = final_value
        print(f"{node_name}: {final_value}")
    except Exception as e:
        print(f"Failed to set {node_name}: {e}")
 
def _set_enum_node(nodemap, node_name, value):
    try:
        node = nodemap.get_node(node_name)
        if node is None:
            print(f"Node '{node_name}' not found")
            return
        node.value = value
        print(f"{node_name}: {value}")
    except Exception as e:
        print(f"Failed to set {node_name}: {e}")
 
def _try_set_line_rate(nodemap, value):
    for node_name in ["AcquisitionLineRate", "LineRate", "LineRateHz"]:
        try:
            node = nodemap.get_node(node_name)
            if node is not None:
                final_value = float(max(node.min, min(value, node.max)))
                node.value = final_value
                print(f"{node_name}: {final_value}")
                return
        except Exception:
            pass
    print("Line rate node not supported / not available")
 
def convert_buffer(buffer):
    """Convert Arena buffer to NumPy array. Supports Mono8 / Mono16."""
    copied_buffer = BufferFactory.copy(buffer)
    try:
        width = copied_buffer.width
        height = copied_buffer.height
        total_bytes = len(copied_buffer.data)
        bytes_per_pixel = total_bytes // (width * height)
 
        c_arr_type = ctypes.c_ubyte * total_bytes
        c_arr = c_arr_type.from_address(ctypes.addressof(copied_buffer.pbytes))
        np_view = np.ctypeslib.as_array(c_arr)
 
        if bytes_per_pixel == 1:
            img = np_view.reshape(height, width).astype(np.uint8)
        elif bytes_per_pixel == 2:
            img = np_view.view(np.uint16).reshape(height, width)
        else:
            img = np_view.reshape(height, width, bytes_per_pixel)
 
        return np.ascontiguousarray(img.copy())
    finally:
        BufferFactory.destroy(copied_buffer)
 
class MultiCameraSystem:
    """Singleton camera system for managing multiple cameras."""
   
    _instance = None
    _lock = threading.Lock()
   
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
            return cls._instance
   
    def __init__(self):
        if not hasattr(self, 'initialized'):
            self.devices = []
            self.camera_map = {}
            self.initialized = False
            self.stream_started = False
            self.expected_cameras = 3
   
    def create_all_connected_devices(self, retries=5, delay=1):
        """Discover all connected cameras."""
        for i in range(retries):
            devices = system.create_device()
            if devices:
                print(f"{len(devices)} camera(s) connected.")
                if len(devices) < self.expected_cameras:
                    print(f"⚠ Expected {self.expected_cameras} cameras, but found {len(devices)}")
                return devices
 
            print(f"Retry {i + 1}/{retries}: No cameras found. Retrying in {delay}s...")
            time.sleep(delay)
 
        raise Exception("❌ No cameras found after retries.")
   
    def setup_camera(self, device):
        """Configure a single camera."""
        nodemap = device.nodemap
        serial = nodemap.get_node("DeviceSerialNumber").value
 
        print(f"\nConfiguring camera {serial}...")
 
        # Turn trigger OFF before changing settings
        try:
            nodemap.get_node("TriggerMode").value = "Off"
            print("TriggerMode: Off")
        except Exception as e:
            print(f"Failed to set TriggerMode Off: {e}")
 
        # Image settings
        _set_numeric_node(nodemap, "Height", HEIGHT)
        _set_numeric_node(nodemap, "Width", WIDTH)
        _set_numeric_node(nodemap, "ExposureTime", EXPOSURE_US)
        _set_numeric_node(nodemap, "Gain", GAIN_DB)
 
        # Gamma
        try:
            gamma_enable = nodemap.get_node("GammaEnable")
            if gamma_enable is not None:
                gamma_enable.value = True
                print("GammaEnable: True")
        except Exception as e:
            print(f"GammaEnable not supported: {e}")
 
        _set_numeric_node(nodemap, "Gamma", DEFAULT_GAMMA)
 
        # Line rate
        _try_set_line_rate(nodemap, DEFAULT_LINE_RATE_HZ)
 
        # Pixel format
        _set_enum_node(nodemap, "PixelFormat", PIXEL_FORMAT)
 
        # Software trigger settings
        _set_enum_node(nodemap, "TriggerSelector", "FrameStart")
        _set_enum_node(nodemap, "TriggerSource", "Software")
        _set_enum_node(nodemap, "AcquisitionMode", "Continuous")
        _set_enum_node(nodemap, "TriggerMode", "On")
 
        # Stream settings
        try:
            tl_stream_nodemap = device.tl_stream_nodemap
            tl_stream_nodemap["StreamAutoNegotiatePacketSize"].value = True
            tl_stream_nodemap["StreamPacketResendEnable"].value = True
            print("Stream settings configured")
        except Exception as e:
            print(f"Failed to set stream settings: {e}")
 
        model = nodemap.get_node("DeviceModelName").value
        print(f"✅ Camera ready: {model} ({serial})")
   
    def initialize(self):
        """Initialize all cameras - called ONCE at app startup."""
        if self.initialized:
            print("Cameras already initialized.")
            return
       
        print("\n🔌 Initializing camera system...")
        self.devices = self.create_all_connected_devices()
       
        print("\n⚙️ Configuring cameras...")
        for device in self.devices:
            self.setup_camera(device)
       
        print("\n▶️ Starting streams...")
        for device in self.devices:
            device.start_stream()
            serial = device.nodemap.get_node("DeviceSerialNumber").value
            self.camera_map[serial] = device
       
        self.stream_started = True
        self.initialized = True
       
        # Flush initial buffers
        self._flush_buffers()
       
        print(f"\n✅ Camera system ready! {len(self.devices)} cameras active.")
        print(f"📸 Camera serials: {list(self.camera_map.keys())}")
   
    def _flush_buffers(self):
        """Clear any stale buffers."""
        for device in self.devices:
            try:
                while True:
                    buffer = device.get_buffer(timeout=100)
                    device.requeue_buffer(buffer)
            except Exception:
                pass
        print("Buffer queues flushed.")
   
    def fire_software_trigger(self, device):
        """Send software trigger to camera."""
        nodemap = device.nodemap
        trigger_node = nodemap.get_node("TriggerSoftware")
        trigger_node.execute()
   
    def capture_one_image(self, device, timeout=2000):
        """Capture single image from camera."""
        buffer = device.get_buffer(timeout=timeout)
        try:
            img = convert_buffer(buffer)
        finally:
            device.requeue_buffer(buffer)
        return img
   
    def capture_all_images(self, timeout=2000):
        """
        Capture synchronized images from all cameras.
        This is the main function to call from main.py.
       
        Returns:
            dict: {serial_number: image_array}
        """
        if not self.initialized or not self.stream_started:
            raise RuntimeError("Camera system not initialized!")
       
        # Fire triggers on all cameras quickly
        for device in self.devices:
            self.fire_software_trigger(device)
       
        # Collect images
        results = {}
        for device in self.devices:
            serial = device.nodemap.get_node("DeviceSerialNumber").value
            try:
                image = self.capture_one_image(device, timeout=timeout)
                results[serial] = image
            except Exception as e:
                print(f"❌ Failed to capture from {serial}: {e}")
                results[serial] = None
       
        return results
   
    def get_images_as_list(self, timeout=2000):
        """
        Capture images and return as a list (ordered by serial number).
        Convenient for AI inference.
       
        Returns:
            list: List of image arrays in consistent order
        """
        results_dict = self.capture_all_images(timeout)
        serials = sorted(results_dict.keys())
        return [results_dict[s] for s in serials if results_dict[s] is not None]
   
    def close(self):
        """Shutdown camera system."""
        print("\n🛑 Shutting down camera system...")
       
        for device in self.devices:
            try:
                device.stop_stream()
            except Exception as e:
                print(f"Error stopping stream: {e}")
       
        try:
            system.destroy_device()
        except Exception as e:
            print(f"Error destroying devices: {e}")
       
        self.devices = []
        self.camera_map = {}
        self.initialized = False
        self.stream_started = False
       
        print("Camera system closed.")
 
# --- Public API functions for backward compatibility ---
 
def create_all_connected_devices():
    """Legacy function - returns camera system instance."""
    global _camera_system
    if _camera_system is None:
        _camera_system = MultiCameraSystem()
        _camera_system.initialize()
    return _camera_system.devices
 
def setup(device):
    """Legacy function - now handled internally."""
    pass  # Configuration is now done in initialize()
 
def capture_images_from_all_cameras():
    """
    Returns:
        dict: {serial_number: image_array}
    """
    global _camera_system
    if _camera_system is None:
        raise RuntimeError("Camera system not initialized! Call create_all_connected_devices() first.")
 
    return _camera_system.capture_all_images()
 
def get_camera_system():
    """Get the global camera system instance."""
    global _camera_system
    return _camera_system
 
def shutdown_cameras():
    """Clean shutdown of camera system."""
    global _camera_system
    if _camera_system:
        _camera_system.close()
        _camera_system = None
 