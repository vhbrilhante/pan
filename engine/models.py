"""
Data Models and Schemas for Engine Detections, Telemetry and API Serialization.
"""

from typing import List, Optional, Dict
from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    """Normalized and pixel bounding box coordinates."""
    x1: float = Field(..., description="Top-left X coordinate (pixels)")
    y1: float = Field(..., description="Top-left Y coordinate (pixels)")
    x2: float = Field(..., description="Bottom-right X coordinate (pixels)")
    y2: float = Field(..., description="Bottom-right Y coordinate (pixels)")
    norm_x1: float = Field(..., description="Normalized top-left X [0.0, 1.0]")
    norm_y1: float = Field(..., description="Normalized top-left Y [0.0, 1.0]")
    norm_x2: float = Field(..., description="Normalized bottom-right X [0.0, 1.0]")
    norm_y2: float = Field(..., description="Normalized bottom-right Y [0.0, 1.0]")


class Detection(BaseModel):
    """Individual object detection item."""
    class_id: int
    class_name: str
    confidence: float
    bbox: BoundingBox
    track_id: Optional[int] = None


class InferenceTelemetry(BaseModel):
    """Detailed latency and throughput metrics for edge monitoring."""
    preprocess_ms: float = 0.0
    inference_ms: float = 0.0
    postprocess_ms: float = 0.0
    total_latency_ms: float = 0.0
    inference_fps: float = 0.0
    capture_fps: float = 0.0
    device: str = "cpu"
    backend: str = "ultralytics"


class FrameMetadata(BaseModel):
    """Complete metadata payload sent alongside or inside WebSocket frame packets."""
    frame_id: int
    timestamp: float
    width: int
    height: int
    detections: List[Detection] = []
    class_counts: Dict[str, int] = {}
    telemetry: InferenceTelemetry


class EngineConfigUpdate(BaseModel):
    """Payload to update inference parameters dynamically without server restart."""
    confidence_threshold: Optional[float] = Field(None, ge=0.0, le=1.0)
    iou_threshold: Optional[float] = Field(None, ge=0.0, le=1.0)
    active_classes: Optional[List[str]] = None
    target_fps: Optional[int] = Field(None, ge=1, le=120)
    jpeg_quality: Optional[int] = Field(None, ge=10, le=100)


class SystemMetrics(BaseModel):
    """Hardware health and utilization metrics."""
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    memory_total_mb: float
    gpu_available: bool = False
    gpu_name: Optional[str] = None
    gpu_memory_used_mb: Optional[float] = None
    uptime_seconds: float
