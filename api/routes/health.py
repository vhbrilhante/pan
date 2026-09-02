"""
Health and Hardware Telemetry REST Endpoints.
"""

import time
import psutil
from fastapi import APIRouter, Request
from engine.models import SystemMetrics
from config import settings

router = APIRouter(prefix="/api/v1", tags=["Health & Telemetry"])
SERVER_START_TIME = time.time()


@router.get("/health")
async def health_check():
    """Basic health check endpoint for container orchestrators and load balancers."""
    return {
        "status": "healthy",
        "app_name": settings.APP_NAME,
        "environment": settings.APP_ENV,
        "backend": settings.INFERENCE_BACKEND,
        "model": settings.MODEL_PATH,
        "timestamp": time.time(),
    }


@router.get("/metrics", response_model=SystemMetrics)
async def get_system_metrics(request: Request):
    """Detailed hardware utilization metrics (CPU, RAM, GPU, Uptime)."""
    mem = psutil.virtual_memory()
    uptime = time.time() - SERVER_START_TIME

    # Check for PyTorch GPU info if available
    gpu_available = False
    gpu_name = None
    gpu_mem = None

    try:
        import torch
        if torch.cuda.is_available():
            gpu_available = True
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = round(torch.cuda.memory_allocated(0) / (1024 * 1024), 2)
    except ImportError:
        pass

    return SystemMetrics(
        cpu_percent=psutil.cpu_percent(interval=None),
        memory_percent=mem.percent,
        memory_used_mb=round(mem.used / (1024 * 1024), 2),
        memory_total_mb=round(mem.total / (1024 * 1024), 2),
        gpu_available=gpu_available,
        gpu_name=gpu_name,
        gpu_memory_used_mb=gpu_mem,
        uptime_seconds=round(uptime, 1),
    )
