"""
Engine package for Edge AI Vision: Video Capture, YOLO Inference, and Telemetry.
"""

from .inference import YOLOInferenceEngine
from .capture import VideoCaptureThread
from .models import Detection, BoundingBox, InferenceTelemetry, FrameMetadata

__all__ = [
    "YOLOInferenceEngine",
    "VideoCaptureThread",
    "Detection",
    "BoundingBox",
    "InferenceTelemetry",
    "FrameMetadata",
]
