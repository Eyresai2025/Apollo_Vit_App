"""
Centralized configuration for Apollo VIT application.
All configuration values should come from here, not hardcoded in code.
"""

from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum
from typing import Optional, Dict, Any
import os
import logging

logger = logging.getLogger(__name__)


class DeviceType(Enum):
    """Supported compute devices."""
    CUDA = "cuda"
    CPU = "cpu"
    CUDA_FALLBACK_CPU = "cuda_fallback_cpu"


@dataclass
class InferenceConfig:
    """Inference pipeline configuration."""
    
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
    
    @classmethod
    def from_env(cls) -> "InferenceConfig":
        """Load configuration from environment variables."""
        try:
            device_str = os.getenv("INFERENCE_DEVICE", "cuda").lower()
            device = DeviceType(device_str)
        except ValueError:
            logger.warning(f"Invalid device '{device_str}', using CUDA")
            device = DeviceType.CUDA
        
        return cls(
            device=device,
            r_align_gpu_concurrency=int(os.getenv("R_ALIGN_CONC", 5)),
            vit_gpu_concurrency=int(os.getenv("VIT_CONC", 5)),
            yolo_gpu_concurrency=int(os.getenv("YOLO_CONC", 5)),
            enable_warmup=os.getenv("ENABLE_WARMUP", "True").lower() == "true",
            warmup_iterations=int(os.getenv("WARMUP_ITER", 2)),
            inference_timeout_sec=int(os.getenv("INFERENCE_TIMEOUT", 30)),
            enable_stage_pipeline=os.getenv("ENABLE_STAGE_PIPELINE", "True").lower() == "true",
        )


@dataclass
class DatabaseConfig:
    """MongoDB database configuration."""
    
    url: str = "mongodb://localhost:27017/"
    name: str = "EyresQC_Apollo"
    pool_size: int = 50
    min_pool_size: int = 10
    timeout_ms: int = 5000
    connect_timeout_ms: int = 10000
    retry_writes: bool = True
    retry_reads: bool = True
    
    @classmethod
    def from_env(cls) -> "DatabaseConfig":
        """Load configuration from environment variables."""
        from src.COMMON.common import load_env
        env = load_env()
        
        return cls(
            url=env.get("DATABASE_URL", cls.url),
            name=env.get("DATABASE_NAME", "EyresQC_Apollo"),
            pool_size=int(os.getenv("DB_POOL_SIZE", 50)),
            min_pool_size=int(os.getenv("DB_MIN_POOL_SIZE", 10)),
            timeout_ms=int(os.getenv("DB_TIMEOUT_MS", 5000)),
        )


@dataclass
class TireConstants:
    """Physical constants for tire dimension calculations."""
    
    ROLLER_DIAMETER_MM: int = 100
    ROLLER_DISTANCE_MM: int = 350
    DEFAULT_ASPECT_RATIO: int = 80
    BEAD_WIDTH_MM: int = 20
    BEAD_CENTER_OFFSET_MM: int = 0
    MM_PER_INCH: float = 25.4
    
    # Tire name format: NNN/NNRNNN (e.g., 195/65R15)
    TIRE_NAME_PATTERN: str = r'^(\d{3})[/_-]?(\d{2})[_-]?R(\d{2,3})$'


@dataclass
class CameraConfig:
    """Camera hardware configuration."""
    
    num_cameras: int = 5
    width: int = 4096
    camera_height: int = 14000
    final_height: int = 42000
    exposure_us: float = 200.0
    gain_db: float = 24.0
    line_rate: float = 4096.178266
    pixel_format: str = "Mono16"
    num_stream_buffers: int = 16


@dataclass
class AppConfig:
    """Main application configuration."""
    
    inference: InferenceConfig = field(default_factory=InferenceConfig.from_env)
    database: DatabaseConfig = field(default_factory=DatabaseConfig.from_env)
    tire: TireConstants = field(default_factory=TireConstants)
    camera: CameraConfig = field(default_factory=CameraConfig)
    
    # Deployment settings
    deployment_mode: bool = os.getenv("DEPLOYMENT", "False").lower() == "true"
    plc_ip: str = os.getenv("PLC_IP", "192.168.10.1")
    
    # Model paths
    model_dir: Path = Path(os.getenv("MODEL_DIR", "./models"))
    capture_dir: Path = Path(os.getenv("CAPTURE_DIR", "./media/capture"))
    output_dir: Path = Path(os.getenv("OUTPUT_DIR", "./media/output"))
    
    @classmethod
    def load(cls) -> "AppConfig":
        """Load application configuration with validation."""
        config = cls()
        config._validate()
        return config
    
    def _validate(self) -> None:
        """Validate configuration values."""
        if self.inference.device == DeviceType.CUDA:
            try:
                import torch
                if not torch.cuda.is_available():
                    logger.warning("CUDA requested but not available, using CPU")
                    self.inference.device = DeviceType.CPU
            except ImportError:
                logger.warning("PyTorch not available, using CPU")
                self.inference.device = DeviceType.CPU
        
        if self.inference.inference_timeout_sec < 5:
            logger.warning("Inference timeout too low, setting to 5 seconds")
            self.inference.inference_timeout_sec = 5
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary for logging."""
        return {
            "device": self.inference.device.value,
            "database": self.database.name,
            "deployment": self.deployment_mode,
            "warmup_enabled": self.inference.enable_warmup,
            "yolo_enabled": self.inference.use_yolo_seg,
        }


# Global configuration instance
_app_config: Optional[AppConfig] = None


def get_config() -> AppConfig:
    """Get or create global application configuration."""
    global _app_config
    if _app_config is None:
        _app_config = AppConfig.load()
        logger.info(f"Configuration loaded: {_app_config.to_dict()}")
    return _app_config


def reset_config() -> None:
    """Reset configuration (for testing)."""
    global _app_config
    _app_config = None
