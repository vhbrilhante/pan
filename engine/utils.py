"""
Visualization, Annotation, and Image Processing Utilities.
Optimized for low-latency Edge AI overlays and WebSocket encoding.
"""

import cv2
import hashlib
import numpy as np
from typing import List, Tuple
from .models import Detection, InferenceTelemetry


def get_class_color(class_name: str) -> Tuple[int, int, int]:
    """Generates a consistent, visually distinct BGR color for a given class name."""
    hash_digest = hashlib.md5(class_name.encode("utf-8")).hexdigest()
    r = int(hash_digest[0:2], 16)
    g = int(hash_digest[2:4], 16)
    b = int(hash_digest[4:6], 16)
    # Ensure colors are bright enough against dark/light backgrounds
    r = max(50, min(230, r))
    g = max(50, min(230, g))
    b = max(50, min(230, b))
    return (b, g, r)  # OpenCV uses BGR


def draw_detections(
    frame: np.ndarray,
    detections: List[Detection],
    draw_hud: bool = True,
    telemetry: InferenceTelemetry = None,
) -> np.ndarray:
    """
    Renders bounding boxes, badges, labels, and optional HUD telemetry over the frame.
    Modifies and returns the frame in-place for maximum performance.
    """
    for det in detections:
        color = get_class_color(det.class_name)
        bbox = det.bbox
        x1, y1, x2, y2 = int(bbox.x1), int(bbox.y1), int(bbox.x2), int(bbox.y2)

        # Draw smooth bounding box with corner accents
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, lineType=cv2.LINE_AA)

        # Draw decorative corner brackets (Cyber-Industrial style)
        corner_len = min(20, (x2 - x1) // 4, (y2 - y1) // 4)
        thickness = 3
        # Top-Left
        cv2.line(frame, (x1, y1), (x1 + corner_len, y1), (255, 255, 255), thickness, cv2.LINE_AA)
        cv2.line(frame, (x1, y1), (x1, y1 + corner_len), (255, 255, 255), thickness, cv2.LINE_AA)
        # Bottom-Right
        cv2.line(frame, (x2, y2), (x2 - corner_len, y2), (255, 255, 255), thickness, cv2.LINE_AA)
        cv2.line(frame, (x2, y2), (x2, y2 - corner_len), (255, 255, 255), thickness, cv2.LINE_AA)

        # Label pill background
        label = f"{det.class_name} {det.confidence:.2f}"
        font_scale = 0.5
        font_thickness = 1
        (text_w, text_h), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
        )

        label_y1 = max(0, y1 - text_h - 8)
        label_y2 = y1
        label_x2 = min(frame.shape[1], x1 + text_w + 10)

        cv2.rectangle(frame, (x1, label_y1), (label_x2, label_y2), color, -1)
        cv2.putText(
            frame,
            label,
            (x1 + 5, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255) if sum(color) < 400 else (0, 0, 0),
            font_thickness,
            lineType=cv2.LINE_AA,
        )

    # Optional on-screen HUD telemetry
    if draw_hud and telemetry:
        hud_bg_w = 260
        hud_bg_h = 75
        cv2.rectangle(frame, (10, 10), (10 + hud_bg_w, 10 + hud_bg_h), (20, 24, 30), -1)
        cv2.rectangle(frame, (10, 10), (10 + hud_bg_w, 10 + hud_bg_h), (0, 210, 255), 1)

        cv2.putText(
            frame,
            f"INFER FPS: {telemetry.inference_fps:.1f} | CAP: {telemetry.capture_fps:.1f}",
            (20, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 220, 255),
            1,
            lineType=cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"LATENCY: {telemetry.total_latency_ms:.1f}ms ({telemetry.backend})",
            (20, 52),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (200, 210, 220),
            1,
            lineType=cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"OBJECTS: {len(detections)}",
            (20, 72),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 120),
            1,
            lineType=cv2.LINE_AA,
        )

    return frame


def encode_frame_to_jpeg(frame: np.ndarray, quality: int = 75) -> bytes:
    """Encodes a BGR OpenCV frame into compressed JPEG byte buffer."""
    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), max(10, min(100, quality))]
    success, buffer = cv2.imencode(".jpg", frame, encode_params)
    if not success:
        raise ValueError("Failed to encode frame to JPEG")
    return buffer.tobytes()
