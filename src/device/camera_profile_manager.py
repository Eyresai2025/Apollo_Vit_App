import json
from pathlib import Path
from copy import deepcopy


ZONE_KEYS = {
    "Sidewall 1": "sidewall1",
    "Sidewall 2": "sidewall2",
    "Tread": "tread",
    "Inner": "inner",
    "Bead": "bead",
}

ZONE_NAMES = list(ZONE_KEYS.keys())


DEFAULT_CAMERA_SETTINGS = {
    "serial": "",
    "enabled": True,

    # Mode
    "use_hardware_trigger": True,

    # Geometry
    "width": 4096,
    "height": 6000,
    "pixel_format": "Mono16",

    # Exposure / gain
    "exposure_auto": "Off",
    "exposure_time": 150.0,
    "gain_auto": "Off",
    "gain": 0.0,

    # Line rate
    "acquisition_line_rate_enable": True,
    "acquisition_line_rate": 4096.0,

    # Acquisition
    "acquisition_mode": "Continuous",

    # Hardware trigger nodes
    "line_selector": "Line0",
    "line_mode": "Input",
    "line_source": "Off",
    "trigger_selector": "AcquisitionStart",
    "trigger_source": "Line0",
    "trigger_activation": "RisingEdge",
    "trigger_mode": "On",

    # Network
    "packet_size": 9000,
}


class CameraProfileManager:
    def __init__(self, profile_dir=None):
        """
        Saves profiles inside:
            media/camera_profiles/

        Example:
            media/camera_profiles/100_camera_config.json
        """

        if profile_dir is None:
            self.profile_dir = Path("media") / "camera_profiles"
        else:
            self.profile_dir = Path(profile_dir)

        self.profile_dir.mkdir(parents=True, exist_ok=True)

    def profile_path(self, sku_name: str) -> Path:
        sku_name = str(sku_name).strip().replace(" ", "_")

        if not sku_name:
            sku_name = "default"

        return self.profile_dir / f"{sku_name}_camera_config.json"

    def default_profile(self, sku_name: str) -> dict:
        profile = {
            "sku": sku_name,
            "cameras": {}
        }

        for zone_name, zone_key in ZONE_KEYS.items():
            profile["cameras"][zone_key] = deepcopy(DEFAULT_CAMERA_SETTINGS)

        return profile

    def save_profile(self, sku_name: str, profile_data: dict) -> Path:
        path = self.profile_path(sku_name)

        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(profile_data, f, indent=4)

        return path

    def load_profile(self, sku_name: str) -> dict:
        path = self.profile_path(sku_name)

        if not path.exists():
            return self.default_profile(sku_name)

        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)