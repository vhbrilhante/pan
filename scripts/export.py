"""
YOLO to ONNX Export and Optimization Pipeline for Edge AI Deployment.
Supports FP16 Half Precision, Graph Simplification, Dynamic Axes, and ONNX Runtime Validation.
"""

import os
import sys
import time
import argparse
import logging
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("export")


def parse_arguments():
    parser = argparse.ArgumentParser(description="Export YOLO PyTorch models to optimized ONNX format for Edge.")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Path or name of the YOLO PyTorch model (.pt)")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image resolution (square)")
    parser.add_argument("--half", action="store_true", help="Export in FP16 Half-Precision (recommended for CUDA/Jetson)")
    parser.add_argument("--dynamic", action="store_true", help="Enable dynamic batch and spatial dimensions")
    parser.add_argument("--simplify", action="store_true", default=True, help="Run onnxsim graph simplification")
    parser.add_argument("--opset", type=int, default=17, help="ONNX Opset version (12-18)")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use for export: 'cpu' or '0'")
    return parser.parse_args()


def export_model():
    args = parse_arguments()
    logger.info("=====================================================")
    logger.info("  Starting YOLO to ONNX Export & Edge Optimization   ")
    logger.info("=====================================================")
    logger.info(f"Target Model: {args.model}")
    logger.info(f"Image Size:   {args.imgsz}x{args.imgsz}")
    logger.info(f"FP16 Half:    {args.half}")
    logger.info(f"Dynamic:      {args.dynamic}")
    logger.info(f"Opset:        {args.opset}")
    logger.info(f"Device:       {args.device}")

    try:
        from ultralytics import YOLO
    except ImportError:
        logger.error("Ultralytics package is not installed. Please run: pip install ultralytics")
        sys.exit(1)

    # 1. Load PyTorch Model
    logger.info("Loading PyTorch YOLO model...")
    model = YOLO(args.model)

    # 2. Execute Ultralytics ONNX Export
    logger.info("Exporting to ONNX...")
    exported_path = model.export(
        format="onnx",
        imgsz=args.imgsz,
        half=args.half,
        dynamic=args.dynamic,
        simplify=args.simplify,
        opset=args.opset,
        device=args.device,
    )

    logger.info(f"Export completed! ONNX Model saved at: {exported_path}")

    # 3. Validate with ONNX Runtime
    logger.info("\n--- Validating Export with ONNX Runtime ---")
    try:
        import onnxruntime as ort

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session = ort.InferenceSession(exported_path, sess_options, providers=["CPUExecutionProvider"])

        input_meta = session.get_inputs()[0]
        output_meta = session.get_outputs()[0]

        logger.info(f"Input Name:  {input_meta.name}")
        logger.info(f"Input Shape: {input_meta.shape}")
        logger.info(f"Input Type:  {input_meta.type}")
        logger.info(f"Output Name: {output_meta.name}")
        logger.info(f"Output Shape:{output_meta.shape}")

        # Benchmark warm-up and test forward pass
        dtype = np.float16 if args.half else np.float32
        dummy_input = np.random.randn(1, 3, args.imgsz, args.imgsz).astype(dtype)

        logger.info("Running 10 test iterations for latency benchmark...")
        latencies = []
        for _ in range(10):
            t0 = time.perf_counter()
            _ = session.run(None, {input_meta.name: dummy_input})
            latencies.append((time.perf_counter() - t0) * 1000.0)

        avg_latency = np.mean(latencies)
        fps = 1000.0 / avg_latency
        logger.info(f"Validation SUCCESS! Average ONNX CPU Latency: {avg_latency:.2f}ms ({fps:.1f} FPS)")
        logger.info("=====================================================")
        logger.info(f"To use this model in the server, update .env with:\nMODEL_PATH={exported_path}\nINFERENCE_BACKEND=onnxruntime")
        logger.info("=====================================================")

    except ImportError:
        logger.warning("onnxruntime is not installed for verification test. Model was exported successfully.")
    except Exception as e:
        logger.error(f"Validation error with ONNX Runtime: {e}")


if __name__ == "__main__":
    export_model()
