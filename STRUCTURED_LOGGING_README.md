# Apollo VIT Structured Logging V1

## Purpose

Structured Logging V1 replaces the single root `app.log` configuration with a centralized, thread-safe logging service. It remains compatible with existing Python `logging.getLogger(...)` calls and adds searchable industrial context.

## Output files

By default the application creates:

```text
logs/
├── app.log      # readable application log
├── app.jsonl    # one JSON object per line
└── error.log    # ERROR and CRITICAL records only
```

Files rotate automatically when `LOG_MAX_BYTES` is reached. Old versions are retained according to `LOG_BACKUP_COUNT`.

## Configuration

Add these values to `.env` only when the defaults need to change:

```env
LOG_DIR=logs
LOG_LEVEL=INFO
LOG_CONSOLE_ENABLED=True
LOG_TEXT_ENABLED=True
LOG_JSON_ENABLED=True
LOG_ERROR_ENABLED=True
LOG_FILE_NAME=app.log
LOG_JSON_FILE_NAME=app.jsonl
LOG_ERROR_FILE_NAME=error.log
LOG_MAX_BYTES=10485760
LOG_BACKUP_COUNT=5
LOG_REPEAT_WINDOW_SEC=5
```

`LOG_REPEAT_WINDOW_SEC=5` suppresses identical messages repeatedly emitted within five seconds. Set it to `0` to disable suppression.

## Existing code

Existing code still works:

```python
import logging
logger = logging.getLogger(__name__)
logger.info("Existing message")
```

New or modified modules should identify their component:

```python
from src.COMMON.structured_logging import get_logger

logger = get_logger(__name__, component="CAMERA")
logger.info(
    "Camera opened",
    extra={
        "event_code": "CAMERA_OPENED",
        "details": {"serial": serial_number},
    },
)
```

## Cycle and zone context

```python
from src.COMMON.structured_logging import get_logger, log_context

logger = get_logger(__name__, component="AI_PIPELINE")

with log_context(cycle_id="Cycle_15", tyre_id="TYRE-208", zone="tread"):
    logger.info(
        "Inference started",
        extra={"event_code": "AI_INFERENCE_STARTED"},
    )
```

## Standard events

```python
import logging
from src.COMMON.structured_logging import get_logger, log_event

logger = get_logger(__name__, component="DATABASE")

log_event(
    logger,
    logging.ERROR,
    "Failed to save inspection cycle",
    event_code="DB_CYCLE_SAVE_FAILED",
    error_code="DB-101",
    cycle_id="Cycle_15",
)
```

## Operation timing

```python
from src.COMMON.structured_logging import get_logger, timed_operation

logger = get_logger(__name__, component="AI_PIPELINE")

with timed_operation(
    logger,
    "tread inference",
    event_code="TREAD_INFERENCE",
    cycle_id="Cycle_15",
    zone="tread",
):
    run_tread_inference()
```

This generates STARTED, COMPLETED or FAILED events, including `duration_ms`.

## Validation

```bat
python tools\validate_configuration.py
python tools\validate_structured_logging.py
python -m unittest tests.test_central_config tests.test_structured_logging -v
```

The logging validator does not connect to PLC, cameras, lasers or MongoDB.
