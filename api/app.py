"""
FastAPI Application Entrypoint and Lifespan Orchestration.
Coordinates the VideoCaptureThread, YOLOInferenceEngine, and WebSocket Streaming.
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from config import settings
from engine.capture import VideoCaptureThread
from engine.inference import YOLOInferenceEngine
from api.routes.health import router as health_router
from api.routes.control import router as control_router
from api.websocket.stream_handler import router as ws_router, streaming_worker_loop

# Configure structured logging
logging.basicConfig(
    level=logging.DEBUG if settings.DEBUG else logging.INFO,
    format="%(asctime)s [%(levelname)s] (%(name)s) %(message)s",
)
logger = logging.getLogger("edge_vision.app")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    Handles startup initialization and graceful resource shutdown on exit.
    """
    logger.info("==================================================")
    logger.info(f"Starting {settings.APP_NAME}...")
    logger.info(f"Inference Backend: {settings.INFERENCE_BACKEND} | Model: {settings.MODEL_PATH}")
    logger.info(f"Camera Source: {settings.CAMERA_SOURCE}")
    logger.info("==================================================")

    # 1. Initialize YOLO Inference Engine
    try:
        engine = YOLOInferenceEngine(
            model_path=settings.MODEL_PATH,
            backend=settings.INFERENCE_BACKEND,
            conf_threshold=settings.CONFIDENCE_THRESHOLD,
            iou_threshold=settings.IOU_THRESHOLD,
            image_size=settings.IMAGE_SIZE,
            device=settings.DEVICE,
            half_precision=settings.HALF_PRECISION,
        )
        app.state.engine = engine
    except Exception as e:
        logger.error(f"Failed to initialize YOLOInferenceEngine: {e}")
        raise e

    # 2. Initialize Video Capture Thread
    capture = VideoCaptureThread(
        source=settings.parsed_camera_source,
        width=settings.CAMERA_WIDTH,
        height=settings.CAMERA_HEIGHT,
        target_fps=settings.CAMERA_FPS,
        enable_mock_fallback=settings.ENABLE_MOCK_FALLBACK,
    ).start()
    app.state.capture = capture

    # 3. Start WebSocket Background Streaming Worker
    app.state.is_running = True
    streaming_task = asyncio.create_task(streaming_worker_loop(app))

    yield

    # Graceful Shutdown
    logger.info("Shutting down Edge Vision Engine...")
    app.state.is_running = False
    streaming_task.cancel()
    try:
        await streaming_task
    except asyncio.CancelledError:
        pass

    if hasattr(app.state, "capture") and app.state.capture:
        app.state.capture.stop()

    logger.info("Shutdown completed successfully.")


def create_app() -> FastAPI:
    """Factory creating and configuring the FastAPI application instance."""
    app = FastAPI(
        title=settings.APP_NAME,
        description="High-Performance Edge Computer Vision Engine with YOLO & WebSockets",
        version="1.0.0",
        lifespan=lifespan,
    )

    # CORS Setup
    origins = settings.CORS_ORIGINS
    if isinstance(origins, str):
        origins = [o.strip() for o in origins.split(",")] if origins != "*" else ["*"]

    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register Routers
    app.include_router(health_router)
    app.include_router(control_router)
    app.include_router(ws_router)

    @app.get("/", include_in_schema=False)
    async def root():
        return RedirectResponse(url="/docs")

    return app


# Root Application Instance for Uvicorn
app = create_app()
