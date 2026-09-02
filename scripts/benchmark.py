"""
Edge AI Inference Benchmark Suite.
Compares PyTorch vs ONNX Runtime throughput and latency across iterations.
"""

import time
import argparse
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("benchmark")


def benchmark_pytorch(model_path: str, imgsz: int, iterations: int, device: str):
    logger.info(f"\n[Benchmarking PyTorch/Ultralytics] Model: {model_path}, Device: {device}")
    from ultralytics import YOLO

    model = YOLO(model_path)
    dummy_frame = np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8)

    # Warmup
    for _ in range(5):
        _ = model.predict(dummy_frame, verbose=False, device=None if device == "auto" else device)

    latencies = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        _ = model.predict(dummy_frame, verbose=False, device=None if device == "auto" else device)
        latencies.append((time.perf_counter() - t0) * 1000.0)

    report_metrics("PyTorch (Ultralytics)", latencies)


def benchmark_onnx(model_path: str, imgsz: int, iterations: int):
    logger.info(f"\n[Benchmarking ONNX Runtime] Model: {model_path}")
    import onnxruntime as ort

    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(model_path, sess_options, providers=ort.get_available_providers())

    input_name = session.get_inputs()[0].name
    dummy_input = np.random.randn(1, 3, imgsz, imgsz).astype(np.float32)

    # Warmup
    for _ in range(5):
        _ = session.run(None, {input_name: dummy_input})

    latencies = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        _ = session.run(None, {input_name: dummy_input})
        latencies.append((time.perf_counter() - t0) * 1000.0)

    report_metrics("ONNX Runtime", latencies)


def report_metrics(backend_name: str, latencies: list):
    arr = np.array(latencies)
    mean_ms = np.mean(arr)
    p50_ms = np.percentile(arr, 50)
    p95_ms = np.percentile(arr, 95)
    p99_ms = np.percentile(arr, 99)
    min_ms = np.min(arr)
    max_ms = np.max(arr)
    fps = 1000.0 / mean_ms

    logger.info(f"--- Results for {backend_name} ---")
    logger.info(f"Mean Latency: {mean_ms:.2f} ms | FPS: {fps:.1f}")
    logger.info(f"P50: {p50_ms:.2f} ms | P95: {p95_ms:.2f} ms | P99: {p99_ms:.2f} ms")
    logger.info(f"Min: {min_ms:.2f} ms | Max: {max_ms:.2f} ms")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Edge AI Inference Benchmark Tool")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Path to .pt or .onnx model")
    parser.add_argument("--imgsz", type=int, default=640, help="Image resolution")
    parser.add_argument("--iterations", type=int, default=50, help="Number of benchmark iterations")
    parser.add_argument("--device", type=str, default="cpu", help="Device: 'cpu' or 'cuda'")
    args = parser.parse_args()

    if args.model.endswith(".onnx"):
        benchmark_onnx(args.model, args.imgsz, args.iterations)
    else:
        benchmark_pytorch(args.model, args.imgsz, args.iterations, args.device)
