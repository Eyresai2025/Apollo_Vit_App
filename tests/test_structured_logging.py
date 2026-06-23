from __future__ import annotations

import json
import logging
import tempfile
import unittest
from pathlib import Path

from src.COMMON.config import ConfigManager
from src.COMMON.structured_logging import (
    ContextFilter,
    JsonLineFormatter,
    RepeatSuppressionFilter,
    configure_logging,
    get_logger,
    log_context,
    shutdown_logging,
)


class StructuredLoggingTests(unittest.TestCase):
    def tearDown(self) -> None:
        shutdown_logging()

    def make_manager(self) -> ConfigManager:
        root = Path(tempfile.mkdtemp(prefix="apollo_logging_test_"))
        env = root / ".env"
        env.write_text(
            "\n".join(
                [
                    "DATABASE_URL=mongodb://localhost:27017/",
                    "DATABASE_NAME=Apollo_Test",
                    "DEPLOYMENT=False",
                    "PLC_IP=192.168.10.1",
                    "INFERENCE_DEVICE=cpu",
                    "CAM_SIDEWALL1_ENABLED=False",
                    "CAM_SIDEWALL2_ENABLED=False",
                    "CAM_INNERWALL_ENABLED=False",
                    "CAM_TREAD_ENABLED=False",
                    "CAM_BEAD_ENABLED=False",
                    "LOG_DIR=logs",
                    "LOG_LEVEL=DEBUG",
                    "LOG_CONSOLE_ENABLED=False",
                    "LOG_TEXT_ENABLED=True",
                    "LOG_JSON_ENABLED=True",
                    "LOG_ERROR_ENABLED=True",
                    "LOG_MAX_BYTES=1048576",
                    "LOG_BACKUP_COUNT=2",
                    "LOG_REPEAT_WINDOW_SEC=0",
                ]
            ),
            encoding="utf-8",
        )
        return ConfigManager(env)

    def test_logging_config_is_typed(self) -> None:
        manager = self.make_manager()
        self.assertEqual(manager.config.logging.level, "DEBUG")
        self.assertEqual(manager.config.logging.max_bytes, 1048576)
        self.assertFalse(manager.config.logging.console_enabled)
        self.assertTrue(manager.validation_report.is_valid)

    def test_json_formatter_contains_context(self) -> None:
        record = logging.LogRecord(
            name="apollo.test",
            level=logging.INFO,
            pathname=__file__,
            lineno=10,
            msg="Cycle complete",
            args=(),
            exc_info=None,
        )
        with log_context(cycle_id="Cycle_12", zone="tread", tyre_id="TYRE-9"):
            ContextFilter().filter(record)
        payload = json.loads(JsonLineFormatter().format(record))
        self.assertEqual(payload["cycle_id"], "Cycle_12")
        self.assertEqual(payload["zone"], "tread")
        self.assertEqual(payload["tyre_id"], "TYRE-9")

    def test_repeat_suppression(self) -> None:
        filter_ = RepeatSuppressionFilter(window_seconds=60)
        first = logging.LogRecord("x", logging.WARNING, __file__, 1, "same", (), None)
        second = logging.LogRecord("x", logging.WARNING, __file__, 2, "same", (), None)
        ContextFilter().filter(first)
        ContextFilter().filter(second)
        self.assertTrue(filter_.filter(first))
        self.assertFalse(filter_.filter(second))

    def test_configured_files_receive_logs(self) -> None:
        manager = self.make_manager()
        paths = configure_logging(manager.config, force=True)
        logger = get_logger("apollo.test", component="TEST")
        logger.info(
            "Structured test event",
            extra={
                "event_code": "TEST_EVENT",
                "cycle_id": "Cycle_99",
                "zone": "bead",
            },
        )
        logger.error(
            "Structured test error",
            extra={"event_code": "TEST_ERROR", "error_code": "TEST-001"},
        )
        shutdown_logging()

        text = Path(paths["text"]).read_text(encoding="utf-8")
        json_lines = Path(paths["json"]).read_text(encoding="utf-8").splitlines()
        error = Path(paths["error"]).read_text(encoding="utf-8")

        self.assertIn("Structured test event", text)
        self.assertIn("Structured test error", error)
        payloads = [json.loads(line) for line in json_lines if line.strip()]
        selected = [item for item in payloads if item.get("event_code") == "TEST_EVENT"]
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0]["cycle_id"], "Cycle_99")
        self.assertEqual(selected[0]["zone"], "bead")


if __name__ == "__main__":
    unittest.main()
