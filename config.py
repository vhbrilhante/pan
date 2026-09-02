"""
Central Configuration Module for Edge AI YOLO Vision System.
Uses Pydantic Settings for environment-driven, type-safe configuration.
"""

from typing import List, Literal, Union
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    # Server Configurations
    APP_NAME: str = "Edge AI YOLO Vision Engine"
    APP_ENV: str = "development"
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = False
    CORS_ORIGINS: Union[str, List[str]] = "*"

    # Video Source Configurations
    # Accepts int for camera index (e.g., "0") or string for RTSP/Video file
    CAMERA_SOURCE: str = "0"
    CAMERA_WIDTH: int = 1280
    CAMERA_HEIGHT: int = 720
    CAMERA_FPS: int = 30
    ENABLE_MOCK_FALLBACK: bool = True

    # Model & Inference Configurations
    MODEL_PATH: str = "yolov8n.pt"
    INFERENCE_BACKEND: Literal["ultralytics", "onnxruntime"] = "ultralytics"
    IMAGE_SIZE: int = 640
    CONFIDENCE_THRESHOLD: float = Field(default=0.45, ge=0.0, le=1.0)
    IOU_THRESHOLD: float = Field(default=0.45, ge=0.0, le=1.0)
    HALF_PRECISION: bool = False
    DEVICE: str = "auto"  # "auto", "cpu", "cuda", "0"

    # Streaming Configurations
    WS_JPEG_QUALITY: int = Field(default=75, ge=10, le=100)
    WS_MAX_CLIENTS: int = 10
    WS_TARGET_FPS: int = 30

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    @property
    def parsed_camera_source(self) -> Union[int, str]:
        """Converts numerical string to int for OpenCV VideoCapture."""
        if self.CAMERA_SOURCE.isdigit():
            return int(self.CAMERA_SOURCE)
        return self.CAMERA_SOURCE


# Global Settings Singleton
settings = Settings()
