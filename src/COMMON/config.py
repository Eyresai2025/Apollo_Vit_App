"""Central configuration service for the Apollo VIT application.

Configuration precedence (highest first):
    1. Operating-system environment variables
    2. Project ``.env`` file
    3. Typed defaults defined in this module

The service intentionally keeps a legacy dictionary interface so existing
modules can be migrated gradually without changing production behaviour.
"""

from __future__ import annotations

import ipaddress
import json
import logging
import os
import re
import sys
import threading
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TRUE_VALUES = {"1", "true", "yes", "y", "on", "enabled"}
_FALSE_VALUES = {"0", "false", "no", "n", "off", "disabled"}
_SECRET_MARKERS = ("PASSWORD", "SECRET", "TOKEN", "API_KEY", "PRIVATE_KEY")
_WINDOWS_ABSOLUTE = re.compile(r"^[A-Za-z]:[\\/]")


def _application_root() -> Path:
    """Return the project/resource root for source and frozen execution."""
    if getattr(sys, "frozen", False):
        if hasattr(sys, "_MEIPASS"):
            return Path(sys._MEIPASS).resolve()  # type: ignore[attr-defined]
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[2]


def _resolve_env_path(env_or_root: Optional[os.PathLike[str] | str]) -> Tuple[Path, Path]:
    """Resolve a project root and .env path from either kind of input."""
    if env_or_root is None:
        root = _application_root()
        return root, root / ".env"

    candidate = Path(env_or_root).expanduser()
    if candidate.name.lower() == ".env" or candidate.is_file():
        env_path = candidate.resolve()
        return env_path.parent, env_path

    root = candidate.resolve()
    return root, root / ".env"


def _strip_inline_comment(value: str) -> str:
    """Strip comments only when they are outside matching quotes."""
    if not value:
        return value
    quote: Optional[str] = None
    escaped = False
    for index, char in enumerate(value):
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if quote:
            if char == quote:
                quote = None
            continue
        if char in {"'", '"'}:
            quote = char
            continue
        if char == "#" and (index == 0 or value[index - 1].isspace()):
            return value[:index].rstrip()
    return value.strip()


def _parse_env_file(path: Path) -> Tuple[Dict[str, str], Dict[str, int], List[str]]:
    """Read a .env file and return values, key line numbers and duplicate keys."""
    values: Dict[str, str] = {}
    lines: Dict[str, int] = {}
    duplicates: List[str] = []

    if not path.exists():
        return values, lines, duplicates

    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8-sig", errors="replace").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.lower().startswith("export "):
            line = line[7:].lstrip()
        if "=" not in line:
            continue

        key, raw_value = line.split("=", 1)
        key = key.strip()
        if not key:
            continue
        value = _strip_inline_comment(raw_value.strip())
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]

        if key in values:
            duplicates.append(key)
        values[key] = value
        lines[key] = line_number

    return values, lines, duplicates


def _expand_value(value: str, values: Mapping[str, str], project_root: Path) -> str:
    """Expand ${KEY}, %KEY%, user-home and PROJECT_ROOT references."""
    expansion_values = dict(values)
    expansion_values.setdefault("PROJECT_ROOT", str(project_root))

    def replace_braced(match: re.Match[str]) -> str:
        return expansion_values.get(match.group(1), match.group(0))

    def replace_percent(match: re.Match[str]) -> str:
        return expansion_values.get(match.group(1), match.group(0))

    expanded = re.sub(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}", replace_braced, value)
    expanded = re.sub(r"%([A-Za-z_][A-Za-z0-9_]*)%", replace_percent, expanded)
    return os.path.expanduser(expanded)


def _as_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None or str(value).strip() == "":
        return default
    normalized = str(value).strip().lower()
    if normalized in _TRUE_VALUES:
        return True
    if normalized in _FALSE_VALUES:
        return False
    raise ValueError(f"Expected boolean, received {value!r}")


def _as_int(value: Any, default: int = 0) -> int:
    if value is None or str(value).strip() == "":
        return default
    return int(str(value).strip())


def _as_float(value: Any, default: float = 0.0) -> float:
    if value is None or str(value).strip() == "":
        return default
    return float(str(value).strip())


def _as_csv(value: Any, default: Sequence[str] = ()) -> Tuple[str, ...]:
    if value is None or str(value).strip() == "":
        return tuple(default)
    return tuple(item.strip() for item in str(value).split(",") if item.strip())


def _is_absolute_path_string(value: str) -> bool:
    return Path(value).is_absolute() or bool(_WINDOWS_ABSOLUTE.match(value))


def _resolve_path(value: str, project_root: Path) -> Path:
    value = _expand_value(value, {}, project_root)
    if _is_absolute_path_string(value):
        return Path(value)
    return (project_root / value).resolve()


def _mask_value(key: str, value: str) -> str:
    upper = key.upper()
    if any(marker in upper for marker in _SECRET_MARKERS):
        return "***"
    if key == "DATABASE_URL" and "@" in value:
        # Preserve protocol and host while hiding credentials.
        return re.sub(r"(://)[^/@]+(?=@)", r"\1***", value)
    return value


# ---------------------------------------------------------------------------
# Validation result model
# ---------------------------------------------------------------------------


class ValidationSeverity(str, Enum):
    ERROR = "ERROR"
    WARNING = "WARNING"
    INFO = "INFO"


@dataclass(frozen=True)
class ValidationIssue:
    severity: ValidationSeverity
    code: str
    message: str
    key: Optional[str] = None
    source: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "severity": self.severity.value,
            "code": self.code,
            "message": self.message,
            "key": self.key,
            "source": self.source,
        }


@dataclass
class ValidationReport:
    issues: List[ValidationIssue] = field(default_factory=list)

    @property
    def errors(self) -> List[ValidationIssue]:
        return [item for item in self.issues if item.severity == ValidationSeverity.ERROR]

    @property
    def warnings(self) -> List[ValidationIssue]:
        return [item for item in self.issues if item.severity == ValidationSeverity.WARNING]

    @property
    def information(self) -> List[ValidationIssue]:
        return [item for item in self.issues if item.severity == ValidationSeverity.INFO]

    @property
    def is_valid(self) -> bool:
        return not self.errors

    @property
    def status(self) -> str:
        if self.errors:
            return "INVALID"
        if self.warnings:
            return "VALID_WITH_WARNINGS"
        return "VALID"

    def add(
        self,
        severity: ValidationSeverity,
        code: str,
        message: str,
        *,
        key: Optional[str] = None,
        source: Optional[str] = None,
    ) -> None:
        self.issues.append(ValidationIssue(severity, code, message, key, source))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "error_count": len(self.errors),
            "warning_count": len(self.warnings),
            "info_count": len(self.information),
            "issues": [item.to_dict() for item in self.issues],
        }


# ---------------------------------------------------------------------------
# Typed configuration sections
# ---------------------------------------------------------------------------


class DeviceType(str, Enum):
    CUDA = "cuda"
    CPU = "cpu"
    CUDA_FALLBACK_CPU = "cuda_fallback_cpu"


@dataclass(frozen=True)
class ApplicationInfoConfig:
    name: str = "Apollo VIT"
    version: str = "v1.0"
    build_number: str = ""
    build_date: str = ""
    deployment_mode: bool = False


@dataclass(frozen=True)
class DatabaseConfig:
    url: str = "mongodb://localhost:27017/"
    name: str = "EyresQC_Apollo"
    gridfs_bucket: str = "fs"
    pool_size: int = 50
    min_pool_size: int = 10
    timeout_ms: int = 5000
    connect_timeout_ms: int = 10000
    retry_writes: bool = True
    retry_reads: bool = True


@dataclass(frozen=True)
class PathConfig:
    project_root: Path
    env_file: Path
    media_root: Path
    model_dir: Path
    capture_dir: Path
    output_dir: Path
    recipe_backup_dir: Path
    validation_report_dir: Path
    logs_dir: Path


@dataclass(frozen=True)
class LoggingConfig:
    level: str = "INFO"
    console_enabled: bool = True
    text_enabled: bool = True
    json_enabled: bool = True
    error_enabled: bool = True
    text_file_name: str = "app.log"
    json_file_name: str = "app.jsonl"
    error_file_name: str = "error.log"
    max_bytes: int = 10 * 1024 * 1024
    backup_count: int = 5
    repeat_window_sec: float = 5.0


@dataclass(frozen=True)
class SecurityConfig:
    enabled: bool = True
    database_path: Path = Path("data/security/apollo_security.db")
    session_timeout_minutes: int = 30
    max_failed_attempts: int = 5
    lockout_minutes: int = 15
    password_min_length: int = 8
    password_max_length: int = 128
    require_complex_password: bool = True
    allow_self_signup: bool = False


@dataclass(frozen=True)
class ModelConfig:
    segmentation_weight: Optional[Path] = None
    r_detector_onnx: Optional[Path] = None
    classification_weight: Optional[Path] = None
    vit_checkpoint: Optional[Path] = None


@dataclass(frozen=True)
class InferenceConfig:
    device: DeviceType = DeviceType.CUDA
    r_align_gpu_concurrency: int = 5
    vit_gpu_concurrency: int = 5
    yolo_gpu_concurrency: int = 5
    enable_warmup: bool = True
    warmup_iterations: int = 2
    inference_timeout_sec: int = 30
    enable_stage_pipeline: bool = True
    use_shared_r_detector: bool = True
    save_cycle_summary: bool = True
    default_tyre_name: str = "195_65_R15"
    use_yolo_seg: bool = True
    seg_imgsz: int = 224
    enable_trt_vit: bool = True
    clean_yolo_cache: bool = True


@dataclass(frozen=True)
class PlcBitAddress:
    db: int
    byte: int
    bit: int


@dataclass(frozen=True)
class PlcConfig:
    ip: str = "192.168.10.1"
    plc_type: str = "Siemens S7-1500"
    rack: int = 0
    slot: int = 1
    retry_count: int = 3
    retry_delay_sec: float = 1.0
    result_pulse_ms: int = 500
    app_ok: PlcBitAddress = field(default_factory=lambda: PlcBitAddress(0, 0, 0))
    accept: PlcBitAddress = field(default_factory=lambda: PlcBitAddress(0, 0, 0))
    reject: PlcBitAddress = field(default_factory=lambda: PlcBitAddress(0, 0, 1))
    active_recipe_db: int = 74
    active_recipe_byte: int = 0
    active_recipe_size: int = 2
    active_recipe_type: str = "INT"


@dataclass(frozen=True)
class HealthConfig:
    monitor_interval_ms: int = 2000
    storage_min_free_gb: float = 10.0
    require_lights: bool = False
    require_laser: bool = False


@dataclass(frozen=True)
class CameraConfig:
    num_cameras: int = 5
    width: int = 4096
    camera_height: int = 14000
    final_height: int = 42000
    exposure_us: float = 200.0
    gain_db: float = 24.0
    line_rate: float = 4096.178266
    pixel_format: str = "Mono16"
    num_stream_buffers: int = 16
    shared_inner_bead: bool = False
    trigger_mode: str = "plc_software"
    serials: Mapping[str, str] = field(default_factory=dict)
    enabled: Mapping[str, bool] = field(default_factory=dict)
    buffer_timeout_ms: int = 5000
    flush_count: int = 3
    packet_size: int = 9000
    packet_delay: int = 1000
    parallel_capture: bool = True


@dataclass(frozen=True)
class DeviceConfig:
    sapera_camexpert_path: Optional[Path] = None
    laser_config_file: Optional[Path] = None
    teledyne_laser_mock: bool = False
    teledyne_cti_path: Optional[Path] = None


@dataclass(frozen=True)
class RecipeTargetConfig:
    index: int
    key: str
    group: str
    axis_id: int
    name: str
    write_db: int
    write_byte: int
    value_type: str


@dataclass(frozen=True)
class RecipeConfig:
    backup_dir: Path
    write_to_plc: bool = False
    plc_db: int = 74
    axis_value_type: str = "REAL"
    verify_tolerance: float = 0.01
    save_bit_enabled: bool = False
    target_count: int = 0
    targets: Tuple[RecipeTargetConfig, ...] = ()
    camera_axis_ids: Tuple[str, ...] = ()
    laser_axis_ids: Tuple[str, ...] = ()
    default_image_count_per_zone: int = 1
    default_train_good_count: int = 20
    default_zone_count: int = 5


@dataclass(frozen=True)
class TireConstants:
    ROLLER_DIAMETER_MM: int = 100
    ROLLER_DISTANCE_MM: int = 350
    DEFAULT_ASPECT_RATIO: int = 80
    BEAD_WIDTH_MM: int = 20
    BEAD_CENTER_OFFSET_MM: int = 0
    MM_PER_INCH: float = 25.4
    TIRE_NAME_PATTERN: str = r"^(\d{3})[/_-]?(\d{2})[_-]?R(\d{2,3})$"


@dataclass(frozen=True)
class AppConfig:
    application: ApplicationInfoConfig
    database: DatabaseConfig
    paths: PathConfig
    logging: LoggingConfig
    security: SecurityConfig
    models: ModelConfig
    inference: InferenceConfig
    plc: PlcConfig
    health: HealthConfig
    camera: CameraConfig
    devices: DeviceConfig
    recipe: RecipeConfig
    tire: TireConstants = field(default_factory=TireConstants)
    raw: Mapping[str, str] = field(default_factory=dict, repr=False)

    # Compatibility fields used by older code.
    @property
    def deployment_mode(self) -> bool:
        return self.application.deployment_mode

    @property
    def plc_ip(self) -> str:
        return self.plc.ip

    @property
    def model_dir(self) -> Path:
        return self.paths.model_dir

    @property
    def capture_dir(self) -> Path:
        return self.paths.capture_dir

    @property
    def output_dir(self) -> Path:
        return self.paths.output_dir

    @classmethod
    def load(cls, env_or_root: Optional[os.PathLike[str] | str] = None) -> "AppConfig":
        return ConfigManager(env_or_root).config

    def to_dict(self, *, mask_secrets: bool = True, include_raw: bool = False) -> Dict[str, Any]:
        def convert(value: Any) -> Any:
            if isinstance(value, Enum):
                return value.value
            if isinstance(value, Path):
                return str(value)
            if is_dataclass(value):
                return {f.name: convert(getattr(value, f.name)) for f in fields(value)}
            if isinstance(value, Mapping):
                return {str(k): convert(v) for k, v in value.items()}
            if isinstance(value, (tuple, list)):
                return [convert(item) for item in value]
            return value

        output = {
            "application": convert(self.application),
            "database": convert(self.database),
            "paths": convert(self.paths),
            "logging": convert(self.logging),
            "security": convert(self.security),
            "models": convert(self.models),
            "inference": convert(self.inference),
            "plc": convert(self.plc),
            "health": convert(self.health),
            "camera": convert(self.camera),
            "devices": convert(self.devices),
            "recipe": convert(self.recipe),
        }
        if mask_secrets:
            output["database"]["url"] = _mask_value("DATABASE_URL", output["database"]["url"])
        if include_raw:
            output["raw"] = {
                key: _mask_value(key, value) if mask_secrets else value
                for key, value in self.raw.items()
            }
        return output


# ---------------------------------------------------------------------------
# Configuration manager
# ---------------------------------------------------------------------------


class ConfigManager:
    """Load, convert, validate and expose all application configuration."""

    def __init__(self, env_or_root: Optional[os.PathLike[str] | str] = None):
        self.project_root, self.env_path = _resolve_env_path(env_or_root)
        self._file_values: Dict[str, str] = {}
        self._line_numbers: Dict[str, int] = {}
        self._duplicate_keys: List[str] = []
        self._values: Dict[str, str] = {}
        self._conversion_issues: List[ValidationIssue] = []
        self.config: AppConfig
        self.validation_report: ValidationReport
        self.reload()

    def reload(self) -> AppConfig:
        self._conversion_issues = []
        self._file_values, self._line_numbers, self._duplicate_keys = _parse_env_file(
            self.env_path
        )

        # Existing file values are the base. Any same-named OS environment
        # variable overrides them. OS-only APOLLO/config keys are also included.
        combined = dict(self._file_values)
        for key, value in os.environ.items():
            if key in combined or key.startswith("APOLLO_"):
                combined[key] = value
        combined.setdefault("PROJECT_ROOT", str(self.project_root))
        self._values = {
            key: _expand_value(str(value), combined, self.project_root)
            for key, value in combined.items()
        }

        self.config = self._build_config()
        self.validation_report = self.validate()
        return self.config

    def _raw(self, key: str, default: Any = "") -> Any:
        return self._values.get(key, default)

    def _convert(self, key: str, converter: Any, default: Any) -> Any:
        raw = self._values.get(key)
        if raw is None or str(raw).strip() == "":
            return default
        try:
            return converter(raw)
        except (TypeError, ValueError) as exc:
            self._conversion_issues.append(
                ValidationIssue(
                    ValidationSeverity.ERROR,
                    "INVALID_CONFIG_TYPE",
                    f"{key} has invalid value {raw!r}: {exc}",
                    key,
                    self.source_for(key),
                )
            )
            return default

    def get(self, key: str, default: Any = None) -> Any:
        return self._values.get(key, default)

    def get_str(self, key: str, default: str = "", *, required: bool = False) -> str:
        value = str(self._values.get(key, default)).strip()
        if required and not value:
            self._conversion_issues.append(
                ValidationIssue(
                    ValidationSeverity.ERROR,
                    "MISSING_REQUIRED_CONFIG",
                    f"Required configuration {key} is empty",
                    key,
                    self.source_for(key),
                )
            )
        return value

    def get_bool(self, key: str, default: bool = False) -> bool:
        return self._convert(key, lambda value: _as_bool(value, default), default)

    def get_int(self, key: str, default: int = 0) -> int:
        return self._convert(key, lambda value: _as_int(value, default), default)

    def get_float(self, key: str, default: float = 0.0) -> float:
        return self._convert(key, lambda value: _as_float(value, default), default)

    def get_csv(self, key: str, default: Sequence[str] = ()) -> Tuple[str, ...]:
        return self._convert(key, lambda value: _as_csv(value, default), tuple(default))

    def get_path(
        self,
        key: str,
        default: str | Path,
        *,
        allow_empty: bool = False,
    ) -> Optional[Path]:
        value = str(self._values.get(key, default)).strip()
        if not value and allow_empty:
            return None
        if not value:
            value = str(default)
        return _resolve_path(value, self.project_root)

    def source_for(self, key: str) -> str:
        if key in os.environ and (key in self._file_values or key.startswith("APOLLO_")):
            return "OS environment"
        if key in self._line_numbers:
            return f"{self.env_path} line {self._line_numbers[key]}"
        return "default"

    def as_legacy_dict(self, *, include_os_overrides: bool = True) -> Dict[str, str]:
        if include_os_overrides:
            return dict(self._values)
        return dict(self._file_values)

    def masked_raw_dict(self) -> Dict[str, str]:
        return {key: _mask_value(key, value) for key, value in self._values.items()}

    def _model_path(self, key: str) -> Optional[Path]:
        filename = self.get_str(key, "")
        if not filename:
            return None
        path = Path(filename)
        if _is_absolute_path_string(filename):
            return path
        return (self.project_root / "media" / "weights" / path).resolve()

    def _bit_address(self, prefix: str, defaults: Tuple[int, int, int]) -> PlcBitAddress:
        return PlcBitAddress(
            db=self.get_int(f"{prefix}_DB", defaults[0]),
            byte=self.get_int(f"{prefix}_BYTE", defaults[1]),
            bit=self.get_int(f"{prefix}_BIT", defaults[2]),
        )

    def _recipe_targets(self) -> Tuple[RecipeTargetConfig, ...]:
        count = max(0, self.get_int("RECIPE_TARGET_COUNT", 0))
        targets: List[RecipeTargetConfig] = []
        for index in range(1, count + 1):
            prefix = f"RECIPE_TARGET_{index}"
            targets.append(
                RecipeTargetConfig(
                    index=index,
                    key=self.get_str(f"{prefix}_KEY", f"target_{index}"),
                    group=self.get_str(f"{prefix}_GROUP", ""),
                    axis_id=self.get_int(f"{prefix}_AXIS_ID", 0),
                    name=self.get_str(f"{prefix}_NAME", f"Target {index}"),
                    write_db=self.get_int(f"{prefix}_WRITE_DB", 0),
                    write_byte=self.get_int(f"{prefix}_WRITE_BYTE", 0),
                    value_type=self.get_str(f"{prefix}_TYPE", "REAL").upper(),
                )
            )
        return tuple(targets)

    def _build_config(self) -> AppConfig:
        media_root = self.get_path("MEDIA_ROOT", "media") or self.project_root / "media"
        model_dir = self.get_path("MODEL_DIR", media_root / "weights") or media_root / "weights"
        capture_dir = self.get_path("CAPTURE_DIR", media_root / "capture") or media_root / "capture"
        output_dir = self.get_path("OUTPUT_DIR", media_root / "output") or media_root / "output"
        recipe_backup_dir = self.get_path("RECIPE_BACKUP_DIR", "media/recipe_backups") or media_root / "recipe_backups"
        validation_dir = self.get_path("VALIDATION_REPORT_DIR", "media/validation_reports") or media_root / "validation_reports"
        logs_dir = self.get_path("LOG_DIR", "logs") or self.project_root / "logs"

        device_value = self.get_str("INFERENCE_DEVICE", "cuda").lower()
        try:
            inference_device = DeviceType(device_value)
        except ValueError:
            self._conversion_issues.append(
                ValidationIssue(
                    ValidationSeverity.ERROR,
                    "INVALID_INFERENCE_DEVICE",
                    f"INFERENCE_DEVICE must be one of {[item.value for item in DeviceType]}",
                    "INFERENCE_DEVICE",
                    self.source_for("INFERENCE_DEVICE"),
                )
            )
            inference_device = DeviceType.CUDA

        # Preserve the historical safe CPU fallback. This is a local compute
        # capability check only; no PLC/camera/database connection is attempted.
        if inference_device == DeviceType.CUDA:
            try:
                import torch  # type: ignore

                if not torch.cuda.is_available():
                    inference_device = DeviceType.CPU
            except ImportError:
                inference_device = DeviceType.CPU

        serials = {
            "sidewall1": self.get_str("CAM_SIDEWALL1_SERIAL", ""),
            "sidewall2": self.get_str("CAM_SIDEWALL2_SERIAL", ""),
            "innerwall": self.get_str("CAM_INNERWALL_SERIAL", ""),
            "tread": self.get_str("CAM_TREAD_SERIAL", ""),
            "bead": self.get_str("CAM_BEAD_SERIAL", ""),
        }
        enabled = {
            "sidewall1": self.get_bool("CAM_SIDEWALL1_ENABLED", True),
            "sidewall2": self.get_bool("CAM_SIDEWALL2_ENABLED", True),
            "innerwall": self.get_bool("CAM_INNERWALL_ENABLED", True),
            "tread": self.get_bool("CAM_TREAD_ENABLED", True),
            "bead": self.get_bool("CAM_BEAD_ENABLED", True),
        }

        application = ApplicationInfoConfig(
            version=self.get_str("APP_VERSION", "v1.0"),
            build_number=self.get_str("BUILD_NUMBER", ""),
            build_date=self.get_str("BUILD_DATE", ""),
            deployment_mode=self.get_bool("DEPLOYMENT", False),
        )
        database = DatabaseConfig(
            url=self.get_str("DATABASE_URL", "mongodb://localhost:27017/", required=True),
            name=self.get_str("DATABASE_NAME", "EyresQC_Apollo", required=True),
            gridfs_bucket=self.get_str("GRIDFS_BUCKET", "fs"),
            pool_size=self.get_int("DB_POOL_SIZE", 50),
            min_pool_size=self.get_int("DB_MIN_POOL_SIZE", 10),
            timeout_ms=self.get_int("DB_TIMEOUT_MS", 5000),
            connect_timeout_ms=self.get_int("DB_CONNECT_TIMEOUT_MS", 10000),
            retry_writes=self.get_bool("DB_RETRY_WRITES", True),
            retry_reads=self.get_bool("DB_RETRY_READS", True),
        )
        paths = PathConfig(
            project_root=self.project_root,
            env_file=self.env_path,
            media_root=media_root,
            model_dir=model_dir,
            capture_dir=capture_dir,
            output_dir=output_dir,
            recipe_backup_dir=recipe_backup_dir,
            validation_report_dir=validation_dir,
            logs_dir=logs_dir,
        )
        logging_config = LoggingConfig(
            level=self.get_str("LOG_LEVEL", "INFO").upper(),
            console_enabled=self.get_bool("LOG_CONSOLE_ENABLED", True),
            text_enabled=self.get_bool("LOG_TEXT_ENABLED", True),
            json_enabled=self.get_bool("LOG_JSON_ENABLED", True),
            error_enabled=self.get_bool("LOG_ERROR_ENABLED", True),
            text_file_name=self.get_str("LOG_FILE_NAME", "app.log"),
            json_file_name=self.get_str("LOG_JSON_FILE_NAME", "app.jsonl"),
            error_file_name=self.get_str("LOG_ERROR_FILE_NAME", "error.log"),
            max_bytes=self.get_int("LOG_MAX_BYTES", 10 * 1024 * 1024),
            backup_count=self.get_int("LOG_BACKUP_COUNT", 5),
            repeat_window_sec=self.get_float("LOG_REPEAT_WINDOW_SEC", 5.0),
        )
        security = SecurityConfig(
            enabled=self.get_bool("AUTH_ENABLED", True),
            database_path=self.get_path(
                "AUTH_DB_PATH",
                "data/security/apollo_security.db",
            ) or (self.project_root / "data" / "security" / "apollo_security.db"),
            session_timeout_minutes=self.get_int("AUTH_SESSION_TIMEOUT_MIN", 30),
            max_failed_attempts=self.get_int("AUTH_MAX_FAILED_ATTEMPTS", 5),
            lockout_minutes=self.get_int("AUTH_LOCKOUT_MIN", 15),
            password_min_length=self.get_int("AUTH_PASSWORD_MIN_LENGTH", 8),
            password_max_length=self.get_int("AUTH_PASSWORD_MAX_LENGTH", 128),
            require_complex_password=self.get_bool("AUTH_REQUIRE_COMPLEX_PASSWORD", True),
            allow_self_signup=self.get_bool("AUTH_ALLOW_SELF_SIGNUP", False),
        )
        models = ModelConfig(
            segmentation_weight=self._model_path("WEIGHT_FILE_Apollo"),
            r_detector_onnx=self._model_path("WEIGHT_FILE_RE_ONNX"),
            classification_weight=self._model_path("WEIGHT_FILE_CLASS"),
            vit_checkpoint=self._model_path("VIT_CHECKPOINT"),
        )
        inference = InferenceConfig(
            device=inference_device,
            r_align_gpu_concurrency=self.get_int("R_ALIGN_CONC", 5),
            vit_gpu_concurrency=self.get_int("VIT_CONC", 5),
            yolo_gpu_concurrency=self.get_int("YOLO_CONC", 5),
            enable_warmup=self.get_bool("ENABLE_WARMUP", True),
            warmup_iterations=self.get_int("WARMUP_ITER", 2),
            inference_timeout_sec=self.get_int("INFERENCE_TIMEOUT", 30),
            enable_stage_pipeline=self.get_bool("ENABLE_STAGE_PIPELINE", True),
            use_shared_r_detector=self.get_bool("USE_SHARED_R_DETECTOR", True),
            save_cycle_summary=self.get_bool("SAVE_CYCLE_SUMMARY", True),
            default_tyre_name=self.get_str("DEFAULT_TYRE_NAME", "195_65_R15"),
            use_yolo_seg=self.get_bool("USE_YOLO_SEG", True),
            seg_imgsz=self.get_int("SEG_IMGSZ", 224),
            enable_trt_vit=self.get_bool("ENABLE_TRT_VIT", True),
            clean_yolo_cache=self.get_bool("CLEAN_YOLO_CACHE", True),
        )
        plc = PlcConfig(
            ip=self.get_str("PLC_IP", "192.168.10.1", required=application.deployment_mode),
            plc_type=self.get_str("PLC_TYPE", "Siemens S7-1500"),
            rack=self.get_int("PLC_RACK", 0),
            slot=self.get_int("PLC_SLOT", 1),
            retry_count=self.get_int("PLC_RETRY_COUNT", 3),
            retry_delay_sec=self.get_float("PLC_RETRY_DELAY_SEC", 1.0),
            result_pulse_ms=self.get_int("PLC_RESULT_PULSE_MS", 500),
            app_ok=self._bit_address("APP_OK", (0, 0, 0)),
            accept=self._bit_address("PLC_ACCEPT", (0, 0, 0)),
            reject=self._bit_address("PLC_REJECT", (0, 0, 1)),
            active_recipe_db=self.get_int("PLC_ACTIVE_RECIPE_DB", 74),
            active_recipe_byte=self.get_int("PLC_ACTIVE_RECIPE_BYTE", 0),
            active_recipe_size=self.get_int("PLC_ACTIVE_RECIPE_SIZE", 2),
            active_recipe_type=self.get_str("PLC_ACTIVE_RECIPE_TYPE", "INT").upper(),
        )
        health = HealthConfig(
            monitor_interval_ms=self.get_int("HEALTH_MONITOR_INTERVAL_MS", 2000),
            storage_min_free_gb=self.get_float("STORAGE_MIN_FREE_GB", 10.0),
            require_lights=self.get_bool("REQUIRE_LIGHTS", False),
            require_laser=self.get_bool("REQUIRE_LASER", False),
        )
        camera = CameraConfig(
            num_cameras=sum(1 for is_enabled in enabled.values() if is_enabled),
            width=self.get_int("CAM_WIDTH", 4096),
            camera_height=self.get_int("CAMERA_HEIGHT", 14000),
            final_height=self.get_int("FINAL_HEIGHT", 42000),
            exposure_us=self.get_float("CAM_EXPOSURE_US", 200.0),
            gain_db=self.get_float("CAM_GAIN_DB", 24.0),
            line_rate=self.get_float("CAM_LINE_RATE", 4096.178266),
            pixel_format=self.get_str("CAM_PIXEL_FORMAT", "Mono16"),
            num_stream_buffers=self.get_int("CAM_NUM_STREAM_BUFFERS", 16),
            shared_inner_bead=self.get_bool("CAM_SHARED_INNER_BEAD", False),
            trigger_mode=self.get_str("CAM_TRIGGER_MODE", "plc_software"),
            serials=serials,
            enabled=enabled,
            buffer_timeout_ms=self.get_int("CAM_BUFFER_TIMEOUT_MS", 5000),
            flush_count=self.get_int("CAM_FLUSH_COUNT", 3),
            packet_size=self.get_int("CAM_PACKET_SIZE", 9000),
            packet_delay=self.get_int("CAM_PACKET_DELAY", 1000),
            parallel_capture=self.get_bool("CAM_PARALLEL_CAPTURE", True),
        )
        devices = DeviceConfig(
            sapera_camexpert_path=self.get_path("SAPERA_CAMEXPERT_PATH", "", allow_empty=True),
            laser_config_file=self.get_path("LASER_CONFIG_FILE", "", allow_empty=True),
            teledyne_laser_mock=self.get_bool("TELEDYNE_LASER_MOCK", False),
            teledyne_cti_path=self.get_path("TELEDYNE_CTI_PATH", "", allow_empty=True),
        )
        targets = self._recipe_targets()
        recipe = RecipeConfig(
            backup_dir=recipe_backup_dir,
            write_to_plc=self.get_bool("RECIPE_WRITE_TO_PLC", False),
            plc_db=self.get_int("RECIPE_PLC_DB", 74),
            axis_value_type=self.get_str("RECIPE_AXIS_VALUE_TYPE", "REAL").upper(),
            verify_tolerance=self.get_float("RECIPE_VERIFY_TOLERANCE", 0.01),
            save_bit_enabled=self.get_bool("RECIPE_SAVE_BIT_ENABLED", False),
            target_count=len(targets),
            targets=targets,
            camera_axis_ids=self.get_csv("CAMERA_AXIS_IDS", ()),
            laser_axis_ids=self.get_csv("LASER_AXIS_IDS", ()),
            default_image_count_per_zone=self.get_int("NEW_SKU_DEFAULT_IMAGE_COUNT_PER_ZONE", 1),
            default_train_good_count=self.get_int("NEW_SKU_DEFAULT_TRAIN_GOOD_COUNT", 20),
            default_zone_count=self.get_int("NEW_SKU_DEFAULT_ZONE_COUNT", 5),
        )

        return AppConfig(
            application=application,
            database=database,
            paths=paths,
            logging=logging_config,
            security=security,
            models=models,
            inference=inference,
            plc=plc,
            health=health,
            camera=camera,
            devices=devices,
            recipe=recipe,
            raw=dict(self._values),
        )

    def validate(self) -> ValidationReport:
        report = ValidationReport(list(self._conversion_issues))

        if not self.env_path.exists():
            report.add(
                ValidationSeverity.ERROR,
                "ENV_FILE_NOT_FOUND",
                f"Configuration file was not found: {self.env_path}",
                source=str(self.env_path),
            )
        for key in sorted(set(self._duplicate_keys)):
            report.add(
                ValidationSeverity.ERROR,
                "DUPLICATE_CONFIG_KEY",
                f"{key} is defined more than once; the final value would win",
                key=key,
                source=str(self.env_path),
            )

        cfg = self.config

        # Logging checks.
        allowed_log_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if cfg.logging.level not in allowed_log_levels:
            report.add(
                ValidationSeverity.ERROR,
                "INVALID_LOG_LEVEL",
                f"LOG_LEVEL must be one of {sorted(allowed_log_levels)}",
                key="LOG_LEVEL",
            )
        if cfg.logging.max_bytes < 1024:
            report.add(
                ValidationSeverity.ERROR,
                "INVALID_LOG_MAX_BYTES",
                "LOG_MAX_BYTES must be at least 1024",
                key="LOG_MAX_BYTES",
            )
        if cfg.logging.backup_count < 1:
            report.add(
                ValidationSeverity.ERROR,
                "INVALID_LOG_BACKUP_COUNT",
                "LOG_BACKUP_COUNT must be at least 1",
                key="LOG_BACKUP_COUNT",
            )
        if cfg.logging.repeat_window_sec < 0:
            report.add(
                ValidationSeverity.ERROR,
                "INVALID_LOG_REPEAT_WINDOW",
                "LOG_REPEAT_WINDOW_SEC cannot be negative",
                key="LOG_REPEAT_WINDOW_SEC",
            )
        if not any((
            cfg.logging.console_enabled,
            cfg.logging.text_enabled,
            cfg.logging.json_enabled,
            cfg.logging.error_enabled,
        )):
            report.add(
                ValidationSeverity.WARNING,
                "ALL_LOG_OUTPUTS_DISABLED",
                "All configured log outputs are disabled; console fallback will be used",
            )

        # Security and role-based access checks.
        if cfg.security.session_timeout_minutes < 1:
            report.add(
                ValidationSeverity.ERROR,
                "INVALID_AUTH_SESSION_TIMEOUT",
                "AUTH_SESSION_TIMEOUT_MIN must be at least 1 minute",
                key="AUTH_SESSION_TIMEOUT_MIN",
            )
        if cfg.security.max_failed_attempts < 1:
            report.add(
                ValidationSeverity.ERROR,
                "INVALID_AUTH_MAX_ATTEMPTS",
                "AUTH_MAX_FAILED_ATTEMPTS must be at least 1",
                key="AUTH_MAX_FAILED_ATTEMPTS",
            )
        if cfg.security.lockout_minutes < 1:
            report.add(
                ValidationSeverity.ERROR,
                "INVALID_AUTH_LOCKOUT",
                "AUTH_LOCKOUT_MIN must be at least 1 minute",
                key="AUTH_LOCKOUT_MIN",
            )
        if cfg.security.password_min_length < 8:
            report.add(
                ValidationSeverity.ERROR,
                "WEAK_AUTH_PASSWORD_MIN_LENGTH",
                "AUTH_PASSWORD_MIN_LENGTH must be at least 8",
                key="AUTH_PASSWORD_MIN_LENGTH",
            )
        if cfg.security.password_max_length < cfg.security.password_min_length:
            report.add(
                ValidationSeverity.ERROR,
                "INVALID_AUTH_PASSWORD_RANGE",
                "AUTH_PASSWORD_MAX_LENGTH cannot be lower than AUTH_PASSWORD_MIN_LENGTH",
            )
        if not cfg.security.enabled:
            report.add(
                ValidationSeverity.WARNING,
                "AUTHENTICATION_DISABLED",
                "AUTH_ENABLED=False bypasses login and should be used only for controlled development",
                key="AUTH_ENABLED",
            )

        # Database checks.
        if not re.match(r"^mongodb(?:\+srv)?://", cfg.database.url, flags=re.IGNORECASE):
            report.add(
                ValidationSeverity.ERROR,
                "INVALID_DATABASE_URL",
                "DATABASE_URL must begin with mongodb:// or mongodb+srv://",
                key="DATABASE_URL",
                source=self.source_for("DATABASE_URL"),
            )
        if cfg.database.min_pool_size < 0 or cfg.database.pool_size <= 0:
            report.add(
                ValidationSeverity.ERROR,
                "INVALID_DATABASE_POOL",
                "Database pool sizes must be positive",
            )
        elif cfg.database.min_pool_size > cfg.database.pool_size:
            report.add(
                ValidationSeverity.ERROR,
                "INVALID_DATABASE_POOL_RANGE",
                "DB_MIN_POOL_SIZE cannot exceed DB_POOL_SIZE",
            )
        if cfg.database.timeout_ms < 100:
            report.add(
                ValidationSeverity.WARNING,
                "LOW_DATABASE_TIMEOUT",
                "DB_TIMEOUT_MS is below 100 ms and may cause unnecessary failures",
                key="DB_TIMEOUT_MS",
            )

        # PLC checks are syntactic only; no network connection is attempted.
        if cfg.plc.ip:
            try:
                ipaddress.ip_address(cfg.plc.ip)
            except ValueError:
                report.add(
                    ValidationSeverity.ERROR,
                    "INVALID_PLC_IP",
                    f"PLC_IP is not a valid IP address: {cfg.plc.ip!r}",
                    key="PLC_IP",
                    source=self.source_for("PLC_IP"),
                )
        elif cfg.deployment_mode:
            report.add(
                ValidationSeverity.ERROR,
                "MISSING_PLC_IP",
                "PLC_IP is required while DEPLOYMENT=True",
                key="PLC_IP",
            )

        for name, address in {
            "APP_OK": cfg.plc.app_ok,
            "PLC_ACCEPT": cfg.plc.accept,
            "PLC_REJECT": cfg.plc.reject,
        }.items():
            if address.db < 0 or address.byte < 0 or not 0 <= address.bit <= 7:
                report.add(
                    ValidationSeverity.ERROR,
                    "INVALID_PLC_BIT_ADDRESS",
                    f"{name} address must use non-negative DB/byte and bit 0..7",
                    key=name,
                )

        if cfg.plc.accept == cfg.plc.reject:
            report.add(
                ValidationSeverity.ERROR,
                "PLC_RESULT_ADDRESS_COLLISION",
                "ACCEPT and REJECT cannot use the same PLC bit address",
            )
        if cfg.plc.result_pulse_ms <= 0:
            report.add(
                ValidationSeverity.ERROR,
                "INVALID_RESULT_PULSE",
                "PLC_RESULT_PULSE_MS must be greater than zero",
                key="PLC_RESULT_PULSE_MS",
            )

        # Inference checks.
        if cfg.inference.inference_timeout_sec < 5:
            report.add(
                ValidationSeverity.WARNING,
                "LOW_INFERENCE_TIMEOUT",
                "Inference timeout below 5 seconds may terminate valid inspections",
                key="INFERENCE_TIMEOUT",
            )
        for key, value in {
            "R_ALIGN_CONC": cfg.inference.r_align_gpu_concurrency,
            "VIT_CONC": cfg.inference.vit_gpu_concurrency,
            "YOLO_CONC": cfg.inference.yolo_gpu_concurrency,
        }.items():
            if value < 1:
                report.add(
                    ValidationSeverity.ERROR,
                    "INVALID_CONCURRENCY",
                    f"{key} must be at least 1",
                    key=key,
                )

        # Model files are warnings because model binaries are intentionally
        # excluded from source-control archives.
        for key, path in {
            "WEIGHT_FILE_Apollo": cfg.models.segmentation_weight,
            "WEIGHT_FILE_RE_ONNX": cfg.models.r_detector_onnx,
            "WEIGHT_FILE_CLASS": cfg.models.classification_weight,
            "VIT_CHECKPOINT": cfg.models.vit_checkpoint,
        }.items():
            if path is None:
                report.add(
                    ValidationSeverity.WARNING,
                    "MODEL_NOT_CONFIGURED",
                    f"{key} is empty",
                    key=key,
                )
            elif not path.exists():
                report.add(
                    ValidationSeverity.WARNING,
                    "MODEL_FILE_NOT_FOUND",
                    f"Configured model file does not exist: {path}",
                    key=key,
                    source=self.source_for(key),
                )

        # Runtime path checks. We do not create anything during validation.
        for key, path in {
            "MEDIA_ROOT": cfg.paths.media_root,
            "MODEL_DIR": cfg.paths.model_dir,
        }.items():
            if not path.exists():
                report.add(
                    ValidationSeverity.WARNING,
                    "PATH_NOT_FOUND",
                    f"Configured path does not exist: {path}",
                    key=key,
                )

        # Camera mapping checks.
        enabled_serials: Dict[str, List[str]] = {}
        for zone, enabled in cfg.camera.enabled.items():
            if not enabled:
                continue
            serial = cfg.camera.serials.get(zone, "").strip()
            if not serial:
                report.add(
                    ValidationSeverity.WARNING,
                    "CAMERA_SERIAL_MISSING",
                    f"Camera {zone} is enabled but has no serial number",
                    key=f"CAM_{zone.upper()}_SERIAL",
                )
                continue
            enabled_serials.setdefault(serial, []).append(zone)

        for serial, zones in enabled_serials.items():
            allowed_shared = cfg.camera.shared_inner_bead and set(zones) == {"innerwall", "bead"}
            if len(zones) > 1 and not allowed_shared:
                report.add(
                    ValidationSeverity.ERROR,
                    "DUPLICATE_CAMERA_SERIAL",
                    f"Camera serial {serial} is assigned to multiple enabled zones: {', '.join(zones)}",
                )
        if cfg.camera.packet_size <= 0:
            report.add(
                ValidationSeverity.ERROR,
                "INVALID_CAMERA_PACKET_SIZE",
                "CAM_PACKET_SIZE must be greater than zero",
                key="CAM_PACKET_SIZE",
            )

        # Recipe checks.
        configured_count = self.get_int("RECIPE_TARGET_COUNT", 0)
        if configured_count != len(cfg.recipe.targets):
            report.add(
                ValidationSeverity.ERROR,
                "RECIPE_TARGET_COUNT_MISMATCH",
                "Configured recipe target count could not be loaded completely",
                key="RECIPE_TARGET_COUNT",
            )
        seen_target_keys: set[str] = set()
        seen_write_addresses: set[Tuple[int, int]] = set()
        for target in cfg.recipe.targets:
            if not target.key:
                report.add(
                    ValidationSeverity.ERROR,
                    "RECIPE_TARGET_KEY_EMPTY",
                    f"Recipe target {target.index} has an empty key",
                )
            elif target.key in seen_target_keys:
                report.add(
                    ValidationSeverity.ERROR,
                    "DUPLICATE_RECIPE_TARGET_KEY",
                    f"Recipe target key is duplicated: {target.key}",
                )
            seen_target_keys.add(target.key)

            address = (target.write_db, target.write_byte)
            if address in seen_write_addresses:
                report.add(
                    ValidationSeverity.WARNING,
                    "DUPLICATE_RECIPE_WRITE_ADDRESS",
                    f"Multiple recipe targets use DB{target.write_db}.DBB{target.write_byte}",
                )
            seen_write_addresses.add(address)
            if target.value_type not in {"REAL", "INT", "DINT", "WORD", "DWORD"}:
                report.add(
                    ValidationSeverity.ERROR,
                    "INVALID_RECIPE_VALUE_TYPE",
                    f"Recipe target {target.index} has unsupported type {target.value_type}",
                )

        report.add(
            ValidationSeverity.INFO,
            "CONFIGURATION_LOADED",
            f"Loaded {len(self._values)} configuration values from {self.env_path}",
            source=str(self.env_path),
        )
        return report

    def export_snapshot(
        self,
        path: os.PathLike[str] | str,
        *,
        include_raw: bool = True,
        mask_secrets: bool = True,
    ) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "configuration": self.config.to_dict(
                mask_secrets=mask_secrets, include_raw=include_raw
            ),
            "validation": self.validation_report.to_dict(),
            "sources": {
                key: self.source_for(key)
                for key in sorted(self._values)
            },
        }
        destination.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return destination


# ---------------------------------------------------------------------------
# Thread-safe global access
# ---------------------------------------------------------------------------


_config_lock = threading.RLock()
_config_manager: Optional[ConfigManager] = None


def get_config_manager(
    env_or_root: Optional[os.PathLike[str] | str] = None,
    *,
    force_reload: bool = False,
) -> ConfigManager:
    global _config_manager
    with _config_lock:
        requested_root, requested_env = _resolve_env_path(env_or_root)
        needs_new = (
            _config_manager is None
            or _config_manager.env_path != requested_env
            or _config_manager.project_root != requested_root
        )
        if needs_new:
            _config_manager = ConfigManager(requested_env)
        elif force_reload:
            _config_manager.reload()
        return _config_manager


def get_config(
    env_or_root: Optional[os.PathLike[str] | str] = None,
    *,
    force_reload: bool = False,
) -> AppConfig:
    manager = get_config_manager(env_or_root, force_reload=force_reload)
    return manager.config


def reset_config() -> None:
    global _config_manager
    with _config_lock:
        _config_manager = None


def reload_config() -> AppConfig:
    return get_config(force_reload=True)


def load_legacy_env(
    env_or_root: Optional[os.PathLike[str] | str] = None,
) -> Dict[str, str]:
    """Compatibility replacement for the historical ``load_env`` function."""
    return get_config_manager(env_or_root).as_legacy_dict()
