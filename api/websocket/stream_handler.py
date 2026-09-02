"""
High-Performance WebSocket Streaming Handler.
Multiplexes compressed JPEG video frames and JSON detection metadata to web clients.
"""

import asyncio
import base64
import json
import logging
import time
from typing import Set, Dict
from collections import Counter
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from engine.models import FrameMetadata
from engine.utils import encode_frame_to_jpeg
from config import settings

logger = logging.getLogger("edge_vision.websocket")
router = APIRouter(tags=["Streaming WebSocket"])


class ConnectionManager:
    """Manages active client WebSocket connections and multi-client broadcasts."""

    def __init__(self):
        self.active_connections: Set[WebSocket] = set()
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket) -> bool:
        """Accepts and registers a new WebSocket client if under capacity."""
        if len(self.active_connections) >= settings.WS_MAX_CLIENTS:
            await websocket.close(code=1008, reason="Max streaming clients reached.")
            return False

        await websocket.accept()
        async with self._lock:
            self.active_connections.add(websocket)
        logger.info(f"WebSocket client connected. Total active clients: {len(self.active_connections)}")
        return True

    async def disconnect(self, websocket: WebSocket) -> None:
        """Unregisters a WebSocket client safely."""
        async with self._lock:
            if websocket in self.active_connections:
                self.active_connections.remove(websocket)
        logger.info(f"WebSocket client disconnected. Total active clients: {len(self.active_connections)}")

    async def broadcast_frame_packet(self, packet_dict: dict) -> None:
        """Broadcasts unified JSON frame packet to all connected clients."""
        if not self.active_connections:
            return

        json_data = json.dumps(packet_dict)
        disconnected = []

        async with self._lock:
            for connection in list(self.active_connections):
                try:
                    await connection.send_text(json_data)
                except Exception:
                    disconnected.append(connection)

            for conn in disconnected:
                if conn in self.active_connections:
                    self.active_connections.remove(conn)


# Global connection manager singleton
manager = ConnectionManager()


@router.websocket("/ws/stream")
async def websocket_stream_endpoint(websocket: WebSocket):
    """
    Real-time WebSocket streaming endpoint.
    Transmits compressed JPEG video frames along with detection bounding boxes and telemetry.
    """
    connected = await manager.connect(websocket)
    if not connected:
        return

    try:
        while True:
            # Keep-alive listener / client control messages
            data = await websocket.receive_text()
            try:
                msg = json.loads(data)
                if msg.get("type") == "ping":
                    await websocket.send_text(json.dumps({"type": "pong", "timestamp": time.time()}))
            except json.JSONDecodeError:
                pass
    except WebSocketDisconnect:
        await manager.disconnect(websocket)
    except Exception as e:
        logger.warning(f"WebSocket error: {e}")
        await manager.disconnect(websocket)


async def streaming_worker_loop(app) -> None:
    """
    Continuous background task running at WS_TARGET_FPS.
    Captures frames, runs YOLO inference, serializes metadata, and broadcasts to clients.
    """
    logger.info("Starting background WebSocket streaming worker loop...")
    frame_interval = 1.0 / max(1, settings.WS_TARGET_FPS)

    while getattr(app.state, "is_running", True):
        loop_start = time.perf_counter()

        # Only process inference if there are active connected clients to conserve Edge compute
        if len(manager.active_connections) > 0:
            capture = getattr(app.state, "capture", None)
            engine = getattr(app.state, "engine", None)

            if capture and engine:
                success, frame, frame_id, capture_fps = capture.read()

                if success and frame is not None:
                    h, w = frame.shape[:2]

                    # Run YOLO inference
                    annotated_frame, detections, telemetry = engine.infer(
                        frame, annotate=True, capture_fps=capture_fps
                    )

                    # Encode annotated frame as JPEG
                    try:
                        jpeg_bytes = encode_frame_to_jpeg(annotated_frame, quality=settings.WS_JPEG_QUALITY)
                        b64_image = base64.b64encode(jpeg_bytes).decode("utf-8")

                        # Summarize class counts
                        counts = dict(Counter([d.class_name for d in detections]))

                        # Assemble metadata payload
                        metadata = FrameMetadata(
                            frame_id=frame_id,
                            timestamp=time.time(),
                            width=w,
                            height=h,
                            detections=detections,
                            class_counts=counts,
                            telemetry=telemetry,
                        )

                        packet = {
                            "type": "frame",
                            "image": f"data:image/jpeg;base64,{b64_image}",
                            "metadata": metadata.model_dump(),
                        }

                        # Broadcast to all clients
                        await manager.broadcast_frame_packet(packet)
                    except Exception as e:
                        logger.error(f"Error encoding or broadcasting frame: {e}")

        # Maintain target frame rate
        elapsed = time.perf_counter() - loop_start
        sleep_duration = max(0.001, frame_interval - elapsed)
        await asyncio.sleep(sleep_duration)

    logger.info("WebSocket streaming worker loop stopped.")
