"""Write and verify Apollo VIT structured log files without hardware."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.COMMON.config import get_config
from src.COMMON.structured_logging import (
    configure_logging,
    get_logger,
    log_event,
    shutdown_logging,
    timed_operation,
)


def main() -> int:
    config = get_config()
    paths = configure_logging(config, force=True)
    logger = get_logger(__name__, component="LOGGING_TEST")

    log_event(
        logger,
        logging.INFO,
        "Structured logging validation started",
        event_code="LOG_VALIDATION_STARTED",
        cycle_id="VALIDATION-CYCLE",
        zone="sidewall1",
        status="STARTED",
    )

    with timed_operation(
        logger,
        "logging validation operation",
        component="LOGGING_TEST",
        event_code="LOG_VALIDATION_OPERATION",
        cycle_id="VALIDATION-CYCLE",
    ):
        sum(range(1000))

    log_event(
        logger,
        logging.WARNING,
        "This is a validation warning",
        event_code="LOG_VALIDATION_WARNING",
        cycle_id="VALIDATION-CYCLE",
        error_code="LOG-WARN-001",
    )
    log_event(
        logger,
        logging.ERROR,
        "This is a validation error record",
        event_code="LOG_VALIDATION_ERROR",
        cycle_id="VALIDATION-CYCLE",
        error_code="LOG-TEST-001",
    )
    shutdown_logging()

    print("=" * 78)
    print("APOLLO VIT STRUCTURED LOGGING VALIDATION")
    print("=" * 78)
    failures = []
    for kind in ("text", "json", "error"):
        raw = paths.get(kind)
        if not raw:
            print(f"{kind.upper():8s}: DISABLED")
            continue
        path = Path(raw)
        ok = path.exists() and path.stat().st_size > 0
        print(f"{kind.upper():8s}: {'OK' if ok else 'FAILED'} | {path}")
        if not ok:
            failures.append(f"{kind} file was not created")

    json_path = Path(paths["json"]) if paths.get("json") else None
    if json_path and json_path.exists():
        parsed = []
        for line in json_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                parsed.append(json.loads(line))
            except json.JSONDecodeError as exc:
                failures.append(f"invalid JSONL line: {exc}")
        matching = [
            item for item in parsed
            if item.get("event_code") == "LOG_VALIDATION_ERROR"
            and item.get("cycle_id") == "VALIDATION-CYCLE"
        ]
        print(f"JSONL   : {'OK' if matching else 'FAILED'} | parsed records={len(parsed)}")
        if not matching:
            failures.append("structured validation event not found in JSONL")

    if failures:
        print("-" * 78)
        for failure in failures:
            print(f"[ERROR] {failure}")
        return 2

    print("-" * 78)
    print("Status  : PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
