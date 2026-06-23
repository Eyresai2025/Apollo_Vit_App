"""Thread-safe structured logging for the Apollo VIT application.

The module keeps standard ``logging.getLogger`` calls compatible while adding:
- rotating human-readable, JSONL and error files;
- contextual fields such as cycle, tyre, zone and error code;
- queue-based file writing for GUI/camera/inference worker threads;
- repeated-message suppression;
- global uncaught-exception logging;
- timing helpers for industrial operations.
"""

from __future__ import annotations

import atexit
import copy
import contextvars
import json
import logging
import logging.handlers
import queue
import sys
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, Mapping, MutableMapping, Optional

from src.COMMON.config import AppConfig, get_config


_CONTEXT_DEFAULTS: Dict[str, Any] = {
    "component": "GENERAL",
    "cycle_id": "-",
    "tyre_id": "-",
    "sku_name": "-",
    "zone": "-",
    "operator": "-",
    "event_code": "-",
    "error_code": "-",
    "operation": "-",
    "status": "-",
    "duration_ms": None,
    "details": None,
}

_log_context: contextvars.ContextVar[Dict[str, Any]] = contextvars.ContextVar(
    "apollo_log_context", default={}
)

_listener_lock = threading.RLock()
_listener: Optional[logging.handlers.QueueListener] = None
_queue_handler: Optional[logging.handlers.QueueHandler] = None
_configured_paths: Dict[str, Path] = {}
_target_handlers: list[logging.Handler] = []
_original_sys_excepthook = sys.excepthook
_original_threading_excepthook = getattr(threading, "excepthook", None)
_hooks_installed = False


class PreservingQueueHandler(logging.handlers.QueueHandler):
    """In-process queue handler that keeps exception metadata for JSON output."""

    def prepare(self, record: logging.LogRecord) -> logging.LogRecord:
        return copy.copy(record)


class ApolloLoggerAdapter(logging.LoggerAdapter):
    """Logger adapter that safely merges persistent and per-call fields."""

    def process(self, msg: Any, kwargs: MutableMapping[str, Any]):
        merged = dict(self.extra or {})
        call_extra = kwargs.get("extra")
        if isinstance(call_extra, Mapping):
            merged.update(call_extra)
        kwargs["extra"] = merged
        return msg, kwargs


class ContextFilter(logging.Filter):
    """Inject Apollo context fields into every standard logging record."""

    def filter(self, record: logging.LogRecord) -> bool:
        context = dict(_log_context.get())
        for key, default in _CONTEXT_DEFAULTS.items():
            if not hasattr(record, key):
                setattr(record, key, context.get(key, default))

        # Normalize values used by text/JSON formatters.
        for key in ("component", "cycle_id", "tyre_id", "sku_name", "zone",
                    "operator", "event_code", "error_code", "operation", "status"):
            value = getattr(record, key, None)
            setattr(record, key, "-" if value in (None, "") else str(value))
        return True


class RepeatSuppressionFilter(logging.Filter):
    """Suppress identical log storms within a configurable time window."""

    def __init__(self, window_seconds: float = 5.0, max_keys: int = 5000):
        super().__init__()
        self.window_seconds = max(0.0, float(window_seconds))
        self.max_keys = max(100, int(max_keys))
        self._state: Dict[tuple, list] = {}
        self._lock = threading.Lock()

    def filter(self, record: logging.LogRecord) -> bool:
        if self.window_seconds <= 0:
            return True

        now = time.monotonic()
        try:
            message = record.getMessage()
        except Exception:
            message = str(record.msg)
        key = (
            record.name,
            record.levelno,
            getattr(record, "component", "GENERAL"),
            getattr(record, "event_code", "-"),
            message,
        )

        with self._lock:
            previous = self._state.get(key)
            if previous is not None and now - float(previous[0]) < self.window_seconds:
                previous[1] = int(previous[1]) + 1
                return False

            suppressed = int(previous[1]) if previous is not None else 0
            self._state[key] = [now, 0]

            if len(self._state) > self.max_keys:
                cutoff = now - max(self.window_seconds * 4, 60.0)
                self._state = {
                    stored_key: stored
                    for stored_key, stored in self._state.items()
                    if float(stored[0]) >= cutoff
                }

        if suppressed:
            record.msg = f"{message} [suppressed {suppressed} repeated messages]"
            record.args = ()
        return True


class JsonLineFormatter(logging.Formatter):
    """One valid JSON object per line for searching and later dashboards."""

    def format(self, record: logging.LogRecord) -> str:
        payload: Dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(
                record.created, tz=timezone.utc
            ).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "component": getattr(record, "component", "GENERAL"),
            "event_code": getattr(record, "event_code", "-"),
            "error_code": getattr(record, "error_code", "-"),
            "message": record.getMessage(),
            "cycle_id": getattr(record, "cycle_id", "-"),
            "tyre_id": getattr(record, "tyre_id", "-"),
            "sku_name": getattr(record, "sku_name", "-"),
            "zone": getattr(record, "zone", "-"),
            "operator": getattr(record, "operator", "-"),
            "operation": getattr(record, "operation", "-"),
            "status": getattr(record, "status", "-"),
            "duration_ms": getattr(record, "duration_ms", None),
            "process_id": record.process,
            "thread": record.threadName,
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        details = getattr(record, "details", None)
        if details not in (None, {}, ""):
            payload["details"] = _json_safe(details)
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        if record.stack_info:
            payload["stack"] = self.formatStack(record.stack_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _safe_log_filename(value: str, fallback: str) -> str:
    candidate = Path(str(value or fallback)).name
    return candidate or fallback


def _build_handlers(config: AppConfig) -> list[logging.Handler]:
    settings = config.logging
    log_dir = config.paths.logs_dir
    log_dir.mkdir(parents=True, exist_ok=True)

    handlers: list[logging.Handler] = []
    context_filter = ContextFilter()
    text_formatter = logging.Formatter(
        fmt=(
            "%(asctime)s | %(levelname)-8s | %(component)-14s | "
            "%(event_code)-20s | cycle=%(cycle_id)s | zone=%(zone)s | "
            "%(name)s | %(message)s"
        ),
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    def configure_handler(handler: logging.Handler, level: int, formatter: logging.Formatter):
        handler.setLevel(level)
        handler.setFormatter(formatter)
        handler.addFilter(context_filter)
        handlers.append(handler)

    level = getattr(logging, settings.level.upper(), logging.INFO)

    if settings.console_enabled:
        console = logging.StreamHandler()
        configure_handler(console, level, text_formatter)

    if settings.text_enabled:
        text_path = log_dir / _safe_log_filename(settings.text_file_name, "app.log")
        text_handler = logging.handlers.RotatingFileHandler(
            text_path,
            maxBytes=settings.max_bytes,
            backupCount=settings.backup_count,
            encoding="utf-8",
            delay=True,
        )
        configure_handler(text_handler, level, text_formatter)
        _configured_paths["text"] = text_path

    if settings.json_enabled:
        json_path = log_dir / _safe_log_filename(settings.json_file_name, "app.jsonl")
        json_handler = logging.handlers.RotatingFileHandler(
            json_path,
            maxBytes=settings.max_bytes,
            backupCount=settings.backup_count,
            encoding="utf-8",
            delay=True,
        )
        configure_handler(json_handler, level, JsonLineFormatter())
        _configured_paths["json"] = json_path

    if settings.error_enabled:
        error_path = log_dir / _safe_log_filename(settings.error_file_name, "error.log")
        error_handler = logging.handlers.RotatingFileHandler(
            error_path,
            maxBytes=settings.max_bytes,
            backupCount=settings.backup_count,
            encoding="utf-8",
            delay=True,
        )
        configure_handler(error_handler, logging.ERROR, text_formatter)
        _configured_paths["error"] = error_path

    if not handlers:
        fallback = logging.StreamHandler()
        configure_handler(fallback, level, text_formatter)

    return handlers


def configure_logging(
    config: Optional[AppConfig] = None,
    *,
    force: bool = False,
) -> Dict[str, str]:
    """Configure queue-based logging once and return active log paths."""
    global _listener, _queue_handler, _target_handlers

    with _listener_lock:
        if _listener is not None and not force:
            return {key: str(path) for key, path in _configured_paths.items()}
        if force:
            shutdown_logging()

        cfg = config or get_config()
        settings = cfg.logging
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, settings.level.upper(), logging.INFO))

        # Remove basicConfig/legacy handlers to prevent duplicate log lines.
        for handler in list(root_logger.handlers):
            root_logger.removeHandler(handler)
            try:
                handler.close()
            except Exception:
                pass

        _configured_paths.clear()
        target_handlers = _build_handlers(cfg)
        _target_handlers = target_handlers
        log_queue: queue.Queue = queue.Queue()
        queue_handler = PreservingQueueHandler(log_queue)
        queue_handler.addFilter(ContextFilter())
        queue_handler.addFilter(RepeatSuppressionFilter(settings.repeat_window_sec))
        setattr(queue_handler, "_apollo_structured_handler", True)
        root_logger.addHandler(queue_handler)

        listener = logging.handlers.QueueListener(
            log_queue, *target_handlers, respect_handler_level=True
        )
        listener.start()
        _queue_handler = queue_handler
        _listener = listener

        logger = get_logger(__name__, component="LOGGING")
        logger.info(
            "Structured logging initialized",
            extra={
                "event_code": "LOGGING_INITIALIZED",
                "details": {
                    "level": settings.level,
                    "log_dir": str(cfg.paths.logs_dir),
                    "files": {key: str(path) for key, path in _configured_paths.items()},
                },
            },
        )
        return {key: str(path) for key, path in _configured_paths.items()}


def shutdown_logging() -> None:
    """Flush and stop the queue listener without affecting later Python shutdown."""
    global _listener, _queue_handler, _target_handlers
    with _listener_lock:
        listener = _listener
        _listener = None
        if listener is not None:
            try:
                listener.stop()
            except Exception:
                pass

        root_logger = logging.getLogger()
        if _queue_handler is not None:
            try:
                root_logger.removeHandler(_queue_handler)
                _queue_handler.close()
            except Exception:
                pass
        _queue_handler = None
        for handler in _target_handlers:
            try:
                handler.flush()
                handler.close()
            except Exception:
                pass
        _target_handlers = []


def get_logger(name: str, *, component: Optional[str] = None, **fields: Any) -> ApolloLoggerAdapter:
    extra: Dict[str, Any] = dict(fields)
    if component:
        extra["component"] = component
    return ApolloLoggerAdapter(logging.getLogger(name), extra)


@contextmanager
def log_context(**fields: Any) -> Iterator[None]:
    """Temporarily attach fields to all log records in the current context/thread."""
    current = dict(_log_context.get())
    current.update({key: value for key, value in fields.items() if value is not None})
    token = _log_context.set(current)
    try:
        yield
    finally:
        _log_context.reset(token)


def set_log_context(**fields: Any) -> None:
    """Persistently update fields for the current context until explicitly cleared."""
    current = dict(_log_context.get())
    current.update({key: value for key, value in fields.items() if value is not None})
    _log_context.set(current)


def clear_log_context(*field_names: str) -> None:
    if not field_names:
        _log_context.set({})
        return
    current = dict(_log_context.get())
    for name in field_names:
        current.pop(name, None)
    _log_context.set(current)


def log_event(
    logger: logging.Logger | logging.LoggerAdapter,
    level: int,
    message: str,
    *,
    event_code: str,
    error_code: Optional[str] = None,
    exc_info: Any = None,
    **fields: Any,
) -> None:
    """Write a standard event without manually constructing ``extra`` dictionaries."""
    known = {
        key: fields.pop(key)
        for key in list(fields)
        if key in _CONTEXT_DEFAULTS and key != "details"
    }
    extra: Dict[str, Any] = {"event_code": event_code, **known}
    if error_code:
        extra["error_code"] = error_code
    if fields:
        extra["details"] = fields
    logger.log(level, message, extra=extra, exc_info=exc_info)


@contextmanager
def timed_operation(
    logger: logging.Logger | logging.LoggerAdapter,
    operation: str,
    *,
    component: Optional[str] = None,
    event_code: Optional[str] = None,
    **fields: Any,
) -> Iterator[None]:
    """Log start/completion/failure and duration for an operation."""
    start = time.perf_counter()
    base_code = event_code or operation.upper().replace(" ", "_")
    start_extra = {"operation": operation, "status": "STARTED", **fields}
    if component:
        start_extra["component"] = component
    log_event(
        logger,
        logging.INFO,
        f"{operation} started",
        event_code=f"{base_code}_STARTED",
        **start_extra,
    )
    try:
        yield
    except Exception:
        duration_ms = round((time.perf_counter() - start) * 1000.0, 3)
        failed_extra = {
            "operation": operation,
            "status": "FAILED",
            "duration_ms": duration_ms,
            **fields,
        }
        if component:
            failed_extra["component"] = component
        log_event(
            logger,
            logging.ERROR,
            f"{operation} failed",
            event_code=f"{base_code}_FAILED",
            error_code=f"{base_code}_ERROR",
            exc_info=True,
            **failed_extra,
        )
        raise
    else:
        duration_ms = round((time.perf_counter() - start) * 1000.0, 3)
        completed_extra = {
            "operation": operation,
            "status": "COMPLETED",
            "duration_ms": duration_ms,
            **fields,
        }
        if component:
            completed_extra["component"] = component
        log_event(
            logger,
            logging.INFO,
            f"{operation} completed",
            event_code=f"{base_code}_COMPLETED",
            **completed_extra,
        )


def install_global_exception_hooks() -> None:
    """Log uncaught main-thread and worker-thread exceptions."""
    global _hooks_installed
    if _hooks_installed:
        return

    def sys_hook(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            _original_sys_excepthook(exc_type, exc_value, exc_traceback)
            return
        logger = get_logger(__name__, component="APPLICATION")
        log_event(
            logger,
            logging.CRITICAL,
            "Uncaught application exception",
            event_code="UNCAUGHT_EXCEPTION",
            error_code="APP-UNCAUGHT",
            exc_info=(exc_type, exc_value, exc_traceback),
        )

    def thread_hook(args):
        logger = get_logger(__name__, component="THREAD")
        log_event(
            logger,
            logging.CRITICAL,
            f"Uncaught exception in thread {args.thread.name if args.thread else '-'}",
            event_code="UNCAUGHT_THREAD_EXCEPTION",
            error_code="THREAD-UNCAUGHT",
            exc_info=(args.exc_type, args.exc_value, args.exc_traceback),
        )

    sys.excepthook = sys_hook
    if hasattr(threading, "excepthook"):
        threading.excepthook = thread_hook  # type: ignore[assignment]
    _hooks_installed = True


def get_log_paths() -> Dict[str, str]:
    return {key: str(path) for key, path in _configured_paths.items()}


atexit.register(shutdown_logging)
