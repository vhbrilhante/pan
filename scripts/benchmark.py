"""
Edge AI Inference Benchmark Suite.

Benchmarks YOLO models using the same Ultralytics inference pipeline,
allowing fair comparisons between PyTorch (.pt) and ONNX (.onnx).

Measures:
- Model loading time
- Preprocessing latency
- Inference latency
- Postprocessing latency
- Total wall-clock latency
- P50 / P95 / P99
- FPS
- CPU utilization
- RAM usage

Exports:
- Raw iteration results to CSV
- Benchmark summary to CSV
- Full report to JSON
"""

from __future__ import annotations

from pathlib import Path
from datetime import datetime
import argparse
import csv
import json
import platform
import sys
import time

import cv2
import numpy as np
import psutil


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


def load_images(
    source: str | Path | None,
    imgsz: int,
) -> list[np.ndarray]:
    if source is None:
        print(
            "[WARNING] Nenhuma fonte real informada. "
            "Usando imagem aleatória."
        )

        return [
            np.random.randint(
                0,
                255,
                (imgsz, imgsz, 3),
                dtype=np.uint8,
            )
        ]

    source_path = Path(source)

    if source_path.is_file():
        image = cv2.imread(str(source_path))

        if image is None:
            raise ValueError(
                f"Não foi possível ler a imagem: {source_path}"
            )

        return [image]

    if source_path.is_dir():
        image_paths = sorted(
            path
            for path in source_path.rglob("*")
            if (
                path.is_file()
                and path.suffix.lower() in IMAGE_EXTENSIONS
            )
        )

        if not image_paths:
            raise ValueError(
                f"Nenhuma imagem encontrada em: {source_path}"
            )

        images: list[np.ndarray] = []

        for image_path in image_paths:
            image = cv2.imread(str(image_path))

            if image is not None:
                images.append(image)

        if not images:
            raise ValueError(
                "Nenhuma imagem válida pôde ser carregada."
            )

        return images

    raise FileNotFoundError(
        f"Fonte não encontrada: {source_path}"
    )


def percentile(
    values: list[float],
    value: float,
) -> float:
    return float(np.percentile(values, value))


def calculate_metrics(
    values: list[float],
) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)

    return {
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
        "p50": percentile(values, 50),
        "p95": percentile(values, 95),
        "p99": percentile(values, 99),
    }


def get_environment_info() -> dict:
    environment = {
        "os": platform.platform(),
        "python": sys.version.split()[0],
        "cpu": platform.processor(),
        "logical_cpus": psutil.cpu_count(logical=True),
        "physical_cpus": psutil.cpu_count(logical=False),
        "ram_total_gb": round(
            psutil.virtual_memory().total / (1024 ** 3),
            2,
        ),
    }

    try:
        import torch

        environment["pytorch"] = torch.__version__
        environment["cuda_available"] = torch.cuda.is_available()
        environment["cuda_version"] = torch.version.cuda

        if torch.cuda.is_available():
            environment["gpu"] = torch.cuda.get_device_name(0)

    except ImportError:
        environment["pytorch"] = None

    try:
        import ultralytics

        environment["ultralytics"] = ultralytics.__version__

    except ImportError:
        environment["ultralytics"] = None

    try:
        import onnxruntime as ort

        environment["onnxruntime"] = ort.__version__
        environment["onnx_providers"] = (
            ort.get_available_providers()
        )

    except ImportError:
        environment["onnxruntime"] = None
        environment["onnx_providers"] = []

    return environment


def benchmark_model(
    model_path: str | Path,
    images: list[np.ndarray],
    imgsz: int,
    iterations: int,
    warmup: int,
    device: str,
    conf: float,
    iou: float,
) -> tuple[dict, list[dict]]:
    from ultralytics import YOLO

    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(
            f"Modelo não encontrado: {model_path}"
        )

    backend = (
        "ONNX"
        if model_path.suffix.lower() == ".onnx"
        else "PyTorch"
    )

    print()
    print("=" * 60)
    print(f"Modelo:   {model_path}")
    print(f"Backend:  {backend}")
    print(f"Device:   {device}")
    print(f"Imagens:  {len(images)}")
    print(f"Warmup:   {warmup}")
    print(f"Iterações: {iterations}")
    print("=" * 60)

    load_start = time.perf_counter()

    model = YOLO(str(model_path))

    load_time_ms = (
        time.perf_counter() - load_start
    ) * 1000

    predict_device = (
        None if device == "auto" else device
    )

    # Warmup
    print("\nWarmup...")

    for index in range(warmup):
        image = images[index % len(images)]

        model.predict(
            image,
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            verbose=False,
            device=predict_device,
        )

    process = psutil.Process()

    raw_results: list[dict] = []

    print("Benchmark...")

    for iteration in range(iterations):
        image = images[iteration % len(images)]

        cpu_before = process.cpu_times()
        ram_before = process.memory_info().rss

        wall_start = time.perf_counter()

        results = model.predict(
            image,
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            verbose=False,
            device=predict_device,
        )

        wall_seconds = time.perf_counter() - wall_start

        cpu_after = process.cpu_times()
        ram_after = process.memory_info().rss

        total_ms = wall_seconds * 1000

        cpu_seconds = (
            (cpu_after.user - cpu_before.user)
            + (cpu_after.system - cpu_before.system)
        )

        logical_cpus = psutil.cpu_count(logical=True) or 1

        cpu_percent = (
            (cpu_seconds / wall_seconds)
            / logical_cpus
            * 100
            if wall_seconds > 0
            else 0
        )

        result = results[0]

        speed = result.speed or {}

        preprocess_ms = float(
            speed.get("preprocess", 0)
        )

        inference_ms = float(
            speed.get("inference", 0)
        )

        postprocess_ms = float(
            speed.get("postprocess", 0)
        )

        detections = (
            len(result.boxes)
            if result.boxes is not None
            else 0
        )

        raw_results.append(
            {
                "iteration": iteration + 1,
                "model": model_path.name,
                "backend": backend,
                "preprocess_ms": preprocess_ms,
                "inference_ms": inference_ms,
                "postprocess_ms": postprocess_ms,
                "total_ms": total_ms,
                "cpu_percent": cpu_percent,
                "ram_before_mb": (
                    ram_before / (1024 ** 2)
                ),
                "ram_after_mb": (
                    ram_after / (1024 ** 2)
                ),
                "detections": detections,
            }
        )

    total_values = [
        item["total_ms"]
        for item in raw_results
    ]

    preprocess_values = [
        item["preprocess_ms"]
        for item in raw_results
    ]

    inference_values = [
        item["inference_ms"]
        for item in raw_results
    ]

    postprocess_values = [
        item["postprocess_ms"]
        for item in raw_results
    ]

    cpu_values = [
        item["cpu_percent"]
        for item in raw_results
    ]

    ram_values = [
        item["ram_after_mb"]
        for item in raw_results
    ]

    total_metrics = calculate_metrics(total_values)

    summary = {
        "model": model_path.name,
        "model_path": str(model_path.resolve()),
        "backend": backend,
        "model_size_mb": round(
            model_path.stat().st_size / (1024 ** 2),
            2,
        ),
        "device": device,
        "imgsz": imgsz,
        "conf": conf,
        "iou": iou,
        "iterations": iterations,
        "warmup": warmup,
        "images_used": len(images),
        "load_time_ms": load_time_ms,
        "preprocess": calculate_metrics(
            preprocess_values
        ),
        "inference": calculate_metrics(
            inference_values
        ),
        "postprocess": calculate_metrics(
            postprocess_values
        ),
        "total": total_metrics,
        "fps": (
            1000 / total_metrics["mean"]
            if total_metrics["mean"] > 0
            else 0
        ),
        "cpu_percent_mean": float(
            np.mean(cpu_values)
        ),
        "ram_mean_mb": float(
            np.mean(ram_values)
        ),
        "ram_peak_mb": float(
            np.max(ram_values)
        ),
    }

    print_summary(summary)

    return summary, raw_results


def print_summary(summary: dict) -> None:
    print()
    print(
        f"--- Resultado: "
        f"{summary['model']} ---"
    )

    print(
        f"Backend: {summary['backend']} | "
        f"Device: {summary['device']}"
    )

    print(
        f"Tamanho: {summary['model_size_mb']:.2f} MB | "
        f"Load: {summary['load_time_ms']:.2f} ms"
    )

    print(
        f"Preprocess: "
        f"{summary['preprocess']['mean']:.2f} ms"
    )

    print(
        f"Inference:  "
        f"{summary['inference']['mean']:.2f} ms"
    )

    print(
        f"Postprocess: "
        f"{summary['postprocess']['mean']:.2f} ms"
    )

    print(
        f"Total:      "
        f"{summary['total']['mean']:.2f} ms"
    )

    print(
        f"P50: {summary['total']['p50']:.2f} ms | "
        f"P95: {summary['total']['p95']:.2f} ms | "
        f"P99: {summary['total']['p99']:.2f} ms"
    )

    print(
        f"FPS: {summary['fps']:.2f}"
    )

    print(
        f"CPU médio: "
        f"{summary['cpu_percent_mean']:.2f}%"
    )

    print(
        f"RAM média: "
        f"{summary['ram_mean_mb']:.2f} MB | "
        f"Pico: {summary['ram_peak_mb']:.2f} MB"
    )


def save_results(
    summaries: list[dict],
    raw_results: list[dict],
    environment: dict,
    output_dir: str | Path,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    raw_csv_path = (
        output_dir
        / f"benchmark_raw_{timestamp}.csv"
    )

    summary_csv_path = (
        output_dir
        / f"benchmark_summary_{timestamp}.csv"
    )

    json_path = (
        output_dir
        / f"benchmark_report_{timestamp}.json"
    )

    if raw_results:
        with raw_csv_path.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=raw_results[0].keys(),
            )

            writer.writeheader()
            writer.writerows(raw_results)

    summary_rows = []

    for summary in summaries:
        summary_rows.append(
            {
                "model": summary["model"],
                "backend": summary["backend"],
                "device": summary["device"],
                "model_size_mb": summary[
                    "model_size_mb"
                ],
                "load_time_ms": summary[
                    "load_time_ms"
                ],
                "preprocess_mean_ms": summary[
                    "preprocess"
                ]["mean"],
                "inference_mean_ms": summary[
                    "inference"
                ]["mean"],
                "postprocess_mean_ms": summary[
                    "postprocess"
                ]["mean"],
                "total_mean_ms": summary[
                    "total"
                ]["mean"],
                "total_std_ms": summary[
                    "total"
                ]["std"],
                "p50_ms": summary["total"]["p50"],
                "p95_ms": summary["total"]["p95"],
                "p99_ms": summary["total"]["p99"],
                "fps": summary["fps"],
                "cpu_mean_percent": summary[
                    "cpu_percent_mean"
                ],
                "ram_mean_mb": summary[
                    "ram_mean_mb"
                ],
                "ram_peak_mb": summary[
                    "ram_peak_mb"
                ],
            }
        )

    if summary_rows:
        with summary_csv_path.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=summary_rows[0].keys(),
            )

            writer.writeheader()
            writer.writerows(summary_rows)

    report = {
        "generated_at": datetime.now().isoformat(),
        "environment": environment,
        "results": summaries,
    }

    with json_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            report,
            file,
            indent=4,
            ensure_ascii=False,
        )

    print()
    print("--- Arquivos gerados ---")
    print(raw_csv_path)
    print(summary_csv_path)
    print(json_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "YOLO Edge AI inference benchmark suite."
        )
    )

    parser.add_argument(
        "--models",
        nargs="+",
        required=True,
        help=(
            "Modelos que serão comparados "
            "(.pt, .onnx)"
        ),
    )

    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help=(
            "Imagem ou pasta de imagens usada "
            "no benchmark"
        ),
    )

    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
    )

    parser.add_argument(
        "--iterations",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--warmup",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=0.45,
    )

    parser.add_argument(
        "--iou",
        type=float,
        default=0.45,
    )

    parser.add_argument(
        "--output",
        type=str,
        default="benchmark_results",
    )

    args = parser.parse_args()

    if args.iterations <= 0:
        parser.error(
            "--iterations precisa ser maior que zero"
        )

    if args.warmup < 0:
        parser.error(
            "--warmup não pode ser negativo"
        )

    if args.imgsz <= 0:
        parser.error(
            "--imgsz precisa ser maior que zero"
        )

    environment = get_environment_info()

    print("\n--- Ambiente ---")

    for key, value in environment.items():
        print(f"{key}: {value}")

    images = load_images(
        source=args.source,
        imgsz=args.imgsz,
    )

    summaries: list[dict] = []
    all_raw_results: list[dict] = []

    for model_path in args.models:
        summary, raw_results = benchmark_model(
            model_path=model_path,
            images=images,
            imgsz=args.imgsz,
            iterations=args.iterations,
            warmup=args.warmup,
            device=args.device,
            conf=args.conf,
            iou=args.iou,
        )

        summaries.append(summary)
        all_raw_results.extend(raw_results)

    save_results(
        summaries=summaries,
        raw_results=all_raw_results,
        environment=environment,
        output_dir=args.output,
    )


if __name__ == "__main__":
    main()