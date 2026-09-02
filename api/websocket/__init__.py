"""
WebSocket package for Real-Time Streaming and Metadata Distribution.
"""

from .stream_handler import router as ws_router, ConnectionManager

__all__ = ["ws_router", "ConnectionManager"]
