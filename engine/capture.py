"""
Threaded Video Capture Pipeline with Zero-Latency Frame Dropping.
Designed for Edge Devices, RTSP Streams, and Webcams.
"""

import cv2
import time
import logging
import threading
import numpy as np
from typing import Optional, Tuple, Union

logger = logging.getLogger("edge_vision.capture")


class VideoCaptureThread:
    """
    High-performance, threaded video capture worker.
    Drops stale frames (maxlen=1 buffer) to prevent latency buildup on Edge devices.
    Includes automatic reconnection for RTSP streams and a synthetic test generator fallback.
    """

    def __init__(
        self,
        source: Union[int, str] = 0,
        width: int = 1280,
        height: int = 720,
        target_fps: int = 30,
        enable_mock_fallback: bool = True,
    ):
        self.source = source
        self.width = width
        self.height = height
        self.target_fps = target_fps
        self.enable_mock_fallback = enable_mock_fallback

        self.cap: Optional[cv2.VideoCapture] = None
        self.is_running = False
        self.thread: Optional[threading.Thread] = None

        # Thread-safe single-frame buffer
        self._lock = threading.Lock()
        self._latest_frame: Optional[np.ndarray] = None
        self._frame_id: int = 0
        self._fps: float = 0.0
        self._fps_counter: int = 0
        self._fps_timer: float = time.time()
        self.is_mock: bool = False

    def start(self) -> "VideoCaptureThread":
        """Starts the capture thread."""
        if self.is_running:
            return self

        self.is_running = True
        self._init_camera()
        self.thread = threading.Thread(target=self._capture_loop, name="VideoCaptureWorker", daemon=True)
        self.thread.start()
        logger.info(f"VideoCaptureThread started for source: {self.source}")
        return self

    def _init_camera(self) -> bool:
        """Attempts to initialize OpenCV VideoCapture or falls back to synthetic generator."""
        try:
            # If numerical source on Windows, use CAP_DSHOW for faster startup
            if isinstance(self.source, int):
                self.cap = cv2.VideoCapture(self.source, cv2.CAP_DSHOW)
            else:
                self.cap = cv2.VideoCapture(self.source)

            if self.cap and self.cap.isOpened():
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                self.cap.set(cv2.CAP_PROP_FPS, self.target_fps)
                self.is_mock = False
                logger.info(f"Hardware camera initialized successfully: {self.source}")
                return True
        except Exception as e:
            logger.warning(f"Failed to open hardware camera {self.source}: {e}")

        if self.enable_mock_fallback:
            self.is_mock = True
            logger.warning("Falling back to Synthetic Mock Video Stream Generator.")
            return True
        else:
            logger.error("Camera open failed and mock fallback is disabled.")
            return False

    def _capture_loop(self) -> None:
        """Continuous frame reading loop running in background thread."""
        frame_interval = 1.0 / max(self.target_fps, 1)
        mock_angle = 0

        while self.is_running:
            start_time = time.time()

            if not self.is_mock and self.cap and self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret or frame is None:
                    logger.warning("Failed to read frame from camera. Attempting reconnect...")
                    self.cap.release()
                    time.sleep(1.0)
                    self._init_camera()
                    continue
            else:
                # Generate synthetic test frame with dynamic moving targets
                frame = self._generate_mock_frame(mock_angle)
                mock_angle = (mock_angle + 3) % 360

            # Update latest frame atomically
            with self._lock:
                self._latest_frame = frame
                self._frame_id += 1
                self._fps_counter += 1

                # Calculate capture FPS every 1 second
                now = time.time()
                elapsed = now - self._fps_timer
                if elapsed >= 1.0:
                    self._fps = self._fps_counter / elapsed
                    self._fps_counter = 0
                    self._fps_timer = now

            # Sleep to match target FPS
            loop_duration = time.time() - start_time
            sleep_time = frame_interval - loop_duration
            if sleep_time > 0:
                time.sleep(sleep_time)

    def _generate_mock_frame(self, angle: int) -> np.ndarray:
        """Generates dynamic synthetic pattern frame for edge testing without hardware."""
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        # Dark cyber grid background
        frame[:] = (20, 24, 30)
        grid_step = 60
        for x in range(0, self.width, grid_step):
            cv2.line(frame, (x, 0), (x, self.height), (35, 42, 50), 1)
        for y in range(0, self.height, grid_step):
            cv2.line(frame, (0, y), (self.width, y), (35, 42, 50), 1)

        # Moving synthetic object (bouncing ball & simulated person silhouette)
        rad = np.radians(angle)
        cx = int(self.width / 2 + (self.width / 3) * np.cos(rad))
        cy = int(self.height / 2 + (self.height / 4) * np.sin(rad * 2))

        # Draw moving circle
        cv2.circle(frame, (cx, cy), 50, (0, 210, 255), -1, lineType=cv2.LINE_AA)
        cv2.circle(frame, (cx, cy), 55, (255, 255, 255), 2, lineType=cv2.LINE_AA)

        # Draw stationary simulated target
        target_x, target_y = int(self.width * 0.2), int(self.height * 0.3)
        cv2.rectangle(frame, (target_x, target_y), (target_x + 160, target_y + 300), (0, 180, 80), -1)
        cv2.putText(
            frame,
            "SYNTHETIC TEST STREAM (MOCK CAMERA)",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 220, 255),
            2,
            lineType=cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"Resolution: {self.width}x{self.height} | Frame: {self._frame_id}",
            (30, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (180, 190, 200),
            1,
            lineType=cv2.LINE_AA,
        )
        return frame

    def read(self) -> Tuple[bool, Optional[np.ndarray], int, float]:
        """
        Retrieves the latest available frame without blocking.
        Returns: (success, frame, frame_id, capture_fps)
        """
        with self._lock:
            if self._latest_frame is None:
                return False, None, self._frame_id, self._fps
            return True, self._latest_frame.copy(), self._frame_id, self._fps

    def stop(self) -> None:
        """Stops the capture thread and releases video devices."""
        self.is_running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)
        if self.cap:
            self.cap.release()
        logger.info("VideoCaptureThread stopped.")
