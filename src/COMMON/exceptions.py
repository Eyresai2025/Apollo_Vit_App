"""
Production-grade custom exceptions for Apollo VIT application.
Provides structured error handling with error codes and context.
"""

from typing import Optional, Dict, Any


class ApolloException(Exception):
    """Base exception for Apollo application."""
    
    def __init__(
        self,
        message: str,
        code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize Apollo exception.
        
        Args:
            message: Human-readable error message
            code: Machine-readable error code (e.g., 'MODEL_LOAD_FAILED')
            details: Additional context dictionary
        """
        self.message = message
        self.code = code or self.__class__.__name__
        self.details = details or {}
        super().__init__(self.message)
    
    def __str__(self) -> str:
        base_msg = f"[{self.code}] {self.message}"
        if self.details:
            details_str = ", ".join(f"{k}={v}" for k, v in self.details.items())
            return f"{base_msg} ({details_str})"
        return base_msg


class ConfigError(ApolloException):
    """Configuration loading or validation error."""
    pass


class ModelLoadError(ApolloException):
    """Failed to load AI model or checkpoint."""
    pass


class InferenceError(ApolloException):
    """Inference pipeline execution failed."""
    pass


class DatabaseError(ApolloException):
    """Database operation failed."""
    pass


class ValidationError(ApolloException):
    """Input validation failed."""
    pass


class DeviceError(ApolloException):
    """Hardware device (camera, laser, PLC) error."""
    pass


class CycleExecutionError(ApolloException):
    """Tire inspection cycle execution failed."""
    pass


class CaptureError(ApolloException):
    """Image capture operation failed."""
    pass
