"""
Control and Dynamic Configuration REST Endpoints.
"""

from fastapi import APIRouter, Request, HTTPException
from engine.models import EngineConfigUpdate
from config import settings

router = APIRouter(prefix="/api/v1", tags=["Engine Control"])


@router.get("/config")
async def get_engine_config(request: Request):
    """Retrieves the current runtime configuration of the inference engine."""
    engine = getattr(request.app.state, "engine", None)
    capture = getattr(request.app.state, "capture", None)

    if not engine:
        raise HTTPException(status_code=503, detail="Inference engine not initialized.")

    return {
        "confidence_threshold": engine.conf_threshold,
        "iou_threshold": engine.iou_threshold,
        "active_classes": engine.active_classes,
        "available_classes": engine.class_names,
        "model_path": engine.model_path,
        "backend": engine.backend,
        "device": engine.device,
        "camera_source": str(settings.CAMERA_SOURCE),
        "is_mock_camera": getattr(capture, "is_mock", False) if capture else False,
        "jpeg_quality": settings.WS_JPEG_QUALITY,
        "target_fps": settings.WS_TARGET_FPS,
    }


@router.post("/config")
async def update_engine_config(payload: EngineConfigUpdate, request: Request):
    """Dynamically updates inference parameters without restarting the service."""
    engine = getattr(request.app.state, "engine", None)
    if not engine:
        raise HTTPException(status_code=503, detail="Inference engine not initialized.")

    if payload.confidence_threshold is not None:
        settings.CONFIDENCE_THRESHOLD = payload.confidence_threshold
    if payload.iou_threshold is not None:
        settings.IOU_THRESHOLD = payload.iou_threshold
    if payload.jpeg_quality is not None:
        settings.WS_JPEG_QUALITY = payload.jpeg_quality
    if payload.target_fps is not None:
        settings.WS_TARGET_FPS = payload.target_fps

    engine.update_config(
        conf_threshold=payload.confidence_threshold,
        iou_threshold=payload.iou_threshold,
        active_classes=payload.active_classes,
    )

    return {
        "status": "success",
        "message": "Engine configuration updated dynamically.",
        "current_config": {
            "confidence_threshold": engine.conf_threshold,
            "iou_threshold": engine.iou_threshold,
            "active_classes": engine.active_classes,
            "jpeg_quality": settings.WS_JPEG_QUALITY,
            "target_fps": settings.WS_TARGET_FPS,
        },
    }
