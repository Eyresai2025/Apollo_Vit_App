from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from src.COMMON.config import ConfigManager, DeviceType


class CentralConfigTests(unittest.TestCase):
    def make_env(self, content: str) -> Path:
        directory = Path(tempfile.mkdtemp(prefix="apollo_config_test_"))
        path = directory / ".env"
        path.write_text(content, encoding="utf-8")
        return path

    def test_typed_loading(self) -> None:
        env_path = self.make_env(
            "\n".join(
                [
                    "DATABASE_URL=mongodb://localhost:27017/",
                    "DATABASE_NAME=Apollo_Test",
                    "DEPLOYMENT=False",
                    "PLC_IP=192.168.10.1",
                    "DB_POOL_SIZE=20",
                    "DB_MIN_POOL_SIZE=5",
                    "ENABLE_WARMUP=no",
                    "INFERENCE_DEVICE=cpu",
                    "CAM_SIDEWALL1_ENABLED=False",
                    "CAM_SIDEWALL2_ENABLED=False",
                    "CAM_INNERWALL_ENABLED=False",
                    "CAM_TREAD_ENABLED=False",
                    "CAM_BEAD_ENABLED=False",
                ]
            )
        )
        manager = ConfigManager(env_path)
        cfg = manager.config
        self.assertEqual(cfg.database.name, "Apollo_Test")
        self.assertEqual(cfg.database.pool_size, 20)
        self.assertFalse(cfg.inference.enable_warmup)
        self.assertEqual(cfg.inference.device, DeviceType.CPU)
        self.assertTrue(manager.validation_report.is_valid)

    def test_os_environment_overrides_file(self) -> None:
        env_path = self.make_env(
            "DATABASE_URL=mongodb://localhost:27017/\n"
            "DATABASE_NAME=FromFile\n"
            "DEPLOYMENT=False\n"
            "PLC_IP=192.168.10.1\n"
        )
        os.environ["DATABASE_NAME"] = "FromOS"
        try:
            manager = ConfigManager(env_path)
            self.assertEqual(manager.config.database.name, "FromOS")
            self.assertEqual(manager.source_for("DATABASE_NAME"), "OS environment")
        finally:
            os.environ.pop("DATABASE_NAME", None)

    def test_invalid_types_are_reported(self) -> None:
        env_path = self.make_env(
            "DATABASE_URL=mongodb://localhost:27017/\n"
            "DATABASE_NAME=Apollo_Test\n"
            "DEPLOYMENT=not-a-bool\n"
            "PLC_IP=192.168.10.1\n"
        )
        manager = ConfigManager(env_path)
        self.assertFalse(manager.validation_report.is_valid)
        codes = {issue.code for issue in manager.validation_report.errors}
        self.assertIn("INVALID_CONFIG_TYPE", codes)

    def test_secrets_are_masked(self) -> None:
        env_path = self.make_env(
            "DATABASE_URL=mongodb://user:password@localhost:27017/\n"
            "DATABASE_NAME=Apollo_Test\n"
            "VALID_PASSWORD=hello\n"
            "DEPLOYMENT=False\n"
            "PLC_IP=192.168.10.1\n"
        )
        manager = ConfigManager(env_path)
        masked = manager.masked_raw_dict()
        self.assertEqual(masked["VALID_PASSWORD"], "***")
        self.assertNotIn("user:password", masked["DATABASE_URL"])


if __name__ == "__main__":
    unittest.main()
