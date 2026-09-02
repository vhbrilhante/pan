"""
Thread-Safe YOLO Inference Engine supporting Ultralytics PyTorch and ONNX Runtime.
Engineered for low-latency Edge Computer Vision pipelines.
"""

import os
import cv2
import time
import logging
import threading
import numpy as np
from typing import List, Tuple, Optional, Dict, Union

from .models import Detection, BoundingBox, InferenceTelemetry
from .utils import draw_detections

logger = logging.getLogger("edge_vision.inference")

# Standard COCO 80 Class Names
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush"
]


class YOLOInferenceEngine:
    """
    Thread-safe YOLO Inference Engine.
    Supports both Ultralytics PyTorch models (.pt) and optimized ONNX Runtime sessions (.onnx).
    """

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        backend: str = "ultralytics",
        conf_threshold: float = 0.45,
        iou_threshold: float = 0.45,
        image_size: int = 640,
        device: str = "auto",
        half_precision: bool = False,
    ):
        self.model_path = model_path
        self.backend = backend.lower()
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.image_size = image_size
        self.device = device
        self.half_precision = half_precision
        self.active_classes: Optional[List[str]] = None

        self._lock = threading.Lock()
        self.model = None
        self.onnx_session = None
        self.class_names: List[str] = COCO_CLASSES

        # Telemetry & Performance Tracking
        self._fps_history: List[float] = []
        self._last_inference_time = time.time()
        self.current_fps: float = 0.0

        self._initialize_model()

    def _initialize_model(self) -> None:
        """Loads and prepares the model backend with hardware acceleration if available."""
        with self._lock:
            if self.backend == "onnxruntime" or self.model_path.endswith(".onnx"):
                self._init_onnxruntime()
            else:
                self._init_ultralytics()

    def _init_ultralytics(self) -> None:
        """Initializes Ultralytics YOLO PyTorch model."""
        try:
            from ultralytics import YOLO
            logger.info(f"Loading Ultralytics YOLO model from: {self.model_path}")
            self.model = YOLO(self.model_path)
            
            # Extract class names if available
            if hasattr(self.model, "names") and self.model.names:
                self.class_names = [self.model.names[i] for i in sorted(self.model.names.keys())]

            self.backend = "ultralytics"
            logger.info(f"Ultralytics YOLO initialized successfully. Classes: {len(self.class_names)}")
        except Exception as e:
            logger.error(f"Failed to load Ultralytics model {self.model_path}: {e}")
            raise e

    def _init_onnxruntime(self) -> None:
        """Initializes ONNX Runtime InferenceSession with available Execution Providers."""
        try:
            import onnxruntime as ort

            # Select available hardware execution providers
            available_providers = ort.get_available_providers()
            providers = []
            if self.device != "cpu":
                if "TensorrtExecutionProvider" in available_providers:
                    providers.append("TensorrtExecutionProvider")
                if "CUDAExecutionProvider" in available_providers:
                    providers.append("CUDAExecutionProvider")
            providers.append("CPUExecutionProvider")

            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            sess_options.intra_op_num_threads = max(1, os.cpu_count() or 4)

            logger.info(f"Loading ONNX Runtime session: {self.model_path} with providers: {providers}")
            self.onnx_session = ort.InferenceSession(self.model_path, sess_options, providers=providers)
            self.backend = "onnxruntime"
            logger.info("ONNX Runtime session initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to load ONNX Runtime session {self.model_path}: {e}")
            raise e

    def update_config(
        self,
        conf_threshold: Optional[float] = None,
        iou_threshold: Optional[float] = None,
        active_classes: Optional[List[str]] = None,
    ) -> None:
        """Thread-safe dynamic update of inference parameters."""
        with self._lock:
            if conf_threshold is not None:
                self.conf_threshold = conf_threshold
            if iou_threshold is not None:
                self.iou_threshold = iou_threshold
            if active_classes is not None:
                self.active_classes = active_classes
            logger.info(f"Updated engine config: conf={self.conf_threshold}, iou={self.iou_threshold}")

    def infer(
        self,
        frame: np.ndarray,
        annotate: bool = True,
        capture_fps: float = 0.0,
    ) -> Tuple[np.ndarray, List[Detection], InferenceTelemetry]:
        """
        Executes end-to-end inference on a single frame in a thread-safe manner.
        Returns:
            - Annotated (or original) image frame
            - List of Detection models
            - InferenceTelemetry performance metrics
        """
        with self._lock:
            start_total = time.perf_counter()

            if self.backend == "onnxruntime" and self.onnx_session is not None:
                detections, telemetry = self._infer_onnx(frame, start_total)
            else:
                detections, telemetry = self._infer_ultralytics(frame, start_total)

            telemetry.capture_fps = capture_fps

            # Filter by active classes if configured
            if self.active_classes:
                detections = [d for d in detections if d.class_name in self.active_classes]

            # Render visual overlays
            out_frame = frame.copy()
            if annotate:
                out_frame = draw_detections(out_frame, detections, draw_hud=True, telemetry=telemetry)

            return out_frame, detections, telemetry

    def _infer_ultralytics(
        self, frame: np.ndarray, start_total: float
    ) -> Tuple[List[Detection], InferenceTelemetry]:
        """Runs inference using Ultralytics YOLO pipeline."""
        h, w = frame.shape[:2]
        t0 = time.perf_counter()

        # Run inference
        results = self.model.predict(
            source=frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            imgsz=self.image_size,
            verbose=False,
            device=None if self.device == "auto" else self.device,
            half=self.half_precision,
        )
        t_infer_end = time.perf_counter()

        detections: List[Detection] = []
        if results and len(results) > 0:
            res = results[0]
            boxes = res.boxes
            if boxes is not None:
                for box in boxes:
                    xyxy = box.xyxy[0].tolist()
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0])
                    cls_name = self.class_names[cls_id] if cls_id < len(self.class_names) else f"class_{cls_id}"

                    x1, y1, x2, y2 = xyxy[0], xyxy[1], xyxy[2], xyxy[3]
                    bbox = BoundingBox(
                        x1=x1,
                        y1=y1,
                        x2=x2,
                        y2=y2,
                        norm_x1=max(0.0, min(1.0, x1 / w)),
                        norm_y1=max(0.0, min(1.0, y1 / h)),
                        norm_x2=max(0.0, min(1.0, x2 / w)),
                        norm_y2=max(0.0, min(1.0, y2 / h)),
                    )
                    detections.append(
                        Detection(
                            class_id=cls_id,
                            class_name=cls_name,
                            confidence=conf,
                            bbox=bbox,
                        )
                    )

        t_post_end = time.perf_counter()

        # Extract timing breakdowns
        speed = results[0].speed if results else {}
        prep_ms = speed.get("preprocess", 0.0)
        inf_ms = speed.get("inference", (t_infer_end - t0) * 1000.0)
        post_ms = speed.get("postprocess", (t_post_end - t_infer_end) * 1000.0)
        total_ms = (t_post_end - start_total) * 1000.0

        fps = self._update_fps(total_ms)

        telemetry = InferenceTelemetry(
            preprocess_ms=round(prep_ms, 2),
            inference_ms=round(inf_ms, 2),
            postprocess_ms=round(post_ms, 2),
            total_latency_ms=round(total_ms, 2),
            inference_fps=round(fps, 1),
            device=str(self.device),
            backend="ultralytics",
        )
        return detections, telemetry

    def _infer_onnx(
        self, frame: np.ndarray, start_total: float
    ) -> Tuple[List[Detection], InferenceTelemetry]:
        """Runs inference directly using ONNX Runtime with custom letterbox and NMS."""
        orig_h, orig_w = frame.shape[:2]
        t_prep_start = time.perf_counter()

        # Preprocessing: Letterbox resize to square image_size
        img, ratio, (pad_w, pad_h) = self._letterbox(frame, new_shape=(self.image_size, self.image_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.transpose((2, 0, 1)).astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0)  # Shape: [1, 3, 640, 640]

        t_infer_start = time.perf_counter()
        prep_ms = (t_infer_start - t_prep_start) * 1000.0

        # ONNX Runtime Execution
        input_name = self.onnx_session.get_inputs()[0].name
        outputs = self.onnx_session.run(None, {input_name: img})
        t_post_start = time.perf_counter()
        infer_ms = (t_post_start - t_infer_start) * 1000.0

        # Post-processing: YOLOv8/v11 output shape is [1, 4 + num_classes, num_anchors]
        output = outputs[0]  # Shape: [1, 84, 8400]
        output = np.squeeze(output, axis=0)  # [84, 8400]
        output = output.T  # [8400, 84]

        boxes = output[:, :4]
        scores = output[:, 4:]

        class_ids = np.argmax(scores, axis=1)
        confidences = np.max(scores, axis=1)

        # Filter by confidence
        mask = confidences >= self.conf_threshold
        boxes = boxes[mask]
        confidences = confidences[mask]
        class_ids = class_ids[mask]

        detections: List[Detection] = []
        if len(boxes) > 0:
            # Convert center_x, center_y, width, height -> x1, y1, x2, y2
            cx, cy, bw, bh = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
            x1 = cx - bw / 2
            y1 = cy - bh / 2
            x2 = cx + bw / 2
            y2 = cy + bh / 2

            # Apply OpenCV NMS
            nms_boxes = [[int(a), int(b), int(c - a), int(d - b)] for a, b, c, d in zip(x1, y1, x2, y2)]
            indices = cv2.dnn.NMSBoxes(nms_boxes, confidences.tolist(), self.conf_threshold, self.iou_threshold)

            if len(indices) > 0:
                for idx in indices.flatten():
                    bx1, by1 = x1[idx], y1[idx]
                    bx2, by2 = x2[idx], y2[idx]

                    # Rescale coordinates back to original frame
                    bx1 = (bx1 - pad_w) / ratio
                    by1 = (by1 - pad_h) / ratio
                    bx2 = (bx2 - pad_w) / ratio
                    by2 = (by2 - pad_h) / ratio

                    # Clip to bounds
                    bx1 = max(0.0, min(float(orig_w), float(bx1)))
                    by1 = max(0.0, min(float(orig_h), float(by1)))
                    bx2 = max(0.0, min(float(orig_w), float(bx2)))
                    by2 = max(0.0, min(float(orig_h), float(by2)))

                    cls_id = int(class_ids[idx])
                    cls_name = self.class_names[cls_id] if cls_id < len(self.class_names) else f"class_{cls_id}"

                    bbox = BoundingBox(
                        x1=bx1,
                        y1=by1,
                        x2=bx2,
                        y2=by2,
                        norm_x1=bx1 / orig_w,
                        norm_y1=by1 / orig_h,
                        norm_x2=bx2 / orig_w,
                        norm_y2=by2 / orig_h,
                    )
                    detections.append(
                        Detection(
                            class_id=cls_id,
                            class_name=cls_name,
                            confidence=float(confidences[idx]),
                            bbox=bbox,
                        )
                    )

        t_post_end = time.perf_counter()
        post_ms = (t_post_end - t_post_start) * 1000.0
        total_ms = (t_post_end - start_total) * 1000.0

        fps = self._update_fps(total_ms)

        telemetry = InferenceTelemetry(
            preprocess_ms=round(prep_ms, 2),
            inference_ms=round(infer_ms, 2),
            postprocess_ms=round(post_ms, 2),
            total_latency_ms=round(total_ms, 2),
            inference_fps=round(fps, 1),
            device=str(self.device),
            backend="onnxruntime",
        )
        return detections, telemetry

    def _letterbox(
        self, im: np.ndarray, new_shape: Tuple[int, int] = (640, 640), color: Tuple[int, int, int] = (114, 114, 114)
    ) -> Tuple[np.ndarray, float, Tuple[float, float]]:
        """Letterbox image resizing preserving aspect ratio with edge padding."""
        shape = im.shape[:2]  # [h, w]
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
        dw /= 2
        dh /= 2

        if shape[::-1] != new_unpad:
            im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
        return im, r, (dw, dh)

    def _update_fps(self, latency_ms: float) -> float:
        """Calculates smoothed exponential moving average FPS."""
        instant_fps = 1000.0 / max(latency_ms, 0.001)
        if self.current_fps == 0.0:
            self.current_fps = instant_fps
        else:
            self.current_fps = 0.85 * self.current_fps + 0.15 * instant_fps
        return self.current_fps
