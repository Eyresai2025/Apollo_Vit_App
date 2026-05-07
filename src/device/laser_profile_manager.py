import json
from pathlib import Path
from copy import deepcopy


LASER_ZONE_KEYS = {
    "Sidewall 1": "sidewall1",
    "Sidewall 2": "sidewall2",
    "Tread": "tread",
}

LASER_ZONE_NAMES = list(LASER_ZONE_KEYS.keys())


DEFAULT_LASER_SETTINGS = {
    "laser_id": "",
    "laser_name": "",
    "enabled": True,

    # Direct GUI configuration mode
    # False = configure all nodes directly from GUI
    # True  = load UserSet first, then apply GUI values
    "use_user_set": False,
    "user_set": "UserSet1",

    # Z-Trak / 3D output settings
    "device_output_type": "Linescan3D",
    "scan3d_data_type": "UniformX Z",
    "profiles_per_scan": 1,

    # Acquisition / profile settings
    "scan_rate": 4000.0,
    "exposure": 100.0,
    "range_mode": "Mid",
    "resolution": "High",

    # ROI
    "roi_x_start": 0,
    "roi_width": 4096,
    "roi_z_start": 0,
    "roi_height": 2048,

    # Filtering / quality
    "profile_averaging": 1,
    "threshold": 50.0,

    # Trigger / network
    "trigger_mode": "Off",
    "trigger_source": "Software",
    "trigger_activation": "RisingEdge",
    "packet_size": 9000,

    # Display scaling
    "x_scale": 1.0,
    "z_scale": 1.0,
    "aspect_lock": True,

    # Output
    "output_format": "Profile",

    # Raw missing point handling
    # Leave empty unless you know invalid raw value from Sapera/Z-Trak
    "invalid_value": "",
}


class LaserProfileManager:
    def __init__(self, profile_dir=None):
        """
        Saves laser profiles inside:

            media/laser_profiles/

        Example:

            media/laser_profiles/100_laser_config.json
        """

        if profile_dir is None:
            self.profile_dir = Path("media") / "laser_profiles"
        else:
            self.profile_dir = Path(profile_dir)

        self.profile_dir.mkdir(parents=True, exist_ok=True)

    def profile_path(self, sku_name: str) -> Path:
        sku_name = str(sku_name).strip().replace(" ", "_")

        if not sku_name:
            sku_name = "default"

        return self.profile_dir / f"{sku_name}_laser_config.json"

    def default_profile(self, sku_name: str) -> dict:
        profile = {
            "sku": sku_name,
            "lasers": {}
        }

        for zone_name, zone_key in LASER_ZONE_KEYS.items():
            profile["lasers"][zone_key] = deepcopy(DEFAULT_LASER_SETTINGS)

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