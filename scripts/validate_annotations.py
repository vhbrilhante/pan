from pathlib import Path
import argparse
import random

import cv2
import numpy as np


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
}


def find_image(
    images_dir: Path,
    stem: str,
) -> Path | None:
    for extension in IMAGE_EXTENSIONS:
        candidate = images_dir / f"{stem}{extension}"

        if candidate.exists():
            return candidate

    return None


def load_polygons(
    label_path: Path,
    width: int,
    height: int,
) -> list[np.ndarray]:
    polygons: list[np.ndarray] = []

    if not label_path.exists():
        return polygons

    with label_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        for line_number, line in enumerate(
            file,
            start=1,
        ):
            parts = line.strip().split()

            if not parts:
                continue

            class_id = int(parts[0])

            if class_id != 0:
                continue

            coordinates = [
                float(value)
                for value in parts[1:]
            ]

            if len(coordinates) < 6:
                print(
                    f"[WARNING] Polígono inválido em "
                    f"{label_path.name}:{line_number}"
                )
                continue

            if len(coordinates) % 2 != 0:
                print(
                    f"[WARNING] Quantidade ímpar de "
                    f"coordenadas em "
                    f"{label_path.name}:{line_number}"
                )
                continue

            points = []

            for index in range(
                0,
                len(coordinates),
                2,
            ):
                normalized_x = coordinates[index]
                normalized_y = coordinates[index + 1]

                x = int(
                    normalized_x * width
                )

                y = int(
                    normalized_y * height
                )

                points.append(
                    [x, y]
                )

            polygons.append(
                np.array(
                    points,
                    dtype=np.int32,
                )
            )

    return polygons


def draw_segmentation(
    image: np.ndarray,
    polygons: list[np.ndarray],
    alpha: float = 0.4,
) -> np.ndarray:
    result = image.copy()
    overlay = image.copy()

    for polygon in polygons:
        cv2.fillPoly(
            overlay,
            [polygon],
            (0, 0, 255),
        )

        cv2.polylines(
            result,
            [polygon],
            isClosed=True,
            color=(0, 255, 0),
            thickness=2,
        )

    result = cv2.addWeighted(
        overlay,
        alpha,
        result,
        1 - alpha,
        0,
    )

    return result


def validate_annotations(
    dataset_dir: str | Path,
    output_dir: str | Path,
    sample_size: int,
    seed: int,
) -> None:
    dataset_dir = Path(dataset_dir)
    output_dir = Path(output_dir)

    images_dir = (
        dataset_dir
        / "images"
        / "all"
    )

    labels_dir = (
        dataset_dir
        / "labels"
        / "all"
    )

    if not images_dir.exists():
        raise FileNotFoundError(
            f"Pasta não encontrada: {images_dir}"
        )

    if not labels_dir.exists():
        raise FileNotFoundError(
            f"Pasta não encontrada: {labels_dir}"
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    labels = sorted(
        labels_dir.glob("*.txt")
    )

    if not labels:
        raise ValueError(
            "Nenhuma label encontrada."
        )

    rng = random.Random(seed)

    sample_size = min(
        sample_size,
        len(labels),
    )

    selected_labels = rng.sample(
        labels,
        sample_size,
    )

    valid = 0
    failed = 0

    for index, label_path in enumerate(
        selected_labels,
        start=1,
    ):
        image_path = find_image(
            images_dir,
            label_path.stem,
        )

        if image_path is None:
            print(
                f"[ERRO] Imagem não encontrada para "
                f"{label_path.name}"
            )

            failed += 1
            continue

        image = cv2.imread(
            str(image_path)
        )

        if image is None:
            print(
                f"[ERRO] Não foi possível ler "
                f"{image_path.name}"
            )

            failed += 1
            continue

        height, width = image.shape[:2]

        polygons = load_polygons(
            label_path=label_path,
            width=width,
            height=height,
        )

        if not polygons:
            print(
                f"[WARNING] Nenhum polígono em "
                f"{label_path.name}"
            )

            failed += 1
            continue

        visualized = draw_segmentation(
            image=image,
            polygons=polygons,
        )

        output_path = (
            output_dir
            / image_path.name
        )

        cv2.imwrite(
            str(output_path),
            visualized,
        )

        valid += 1

        print(
            f"[{index}/{sample_size}] "
            f"{image_path.name} -> "
            f"{len(polygons)} polígono(s)"
        )

    print()
    print("--- Validação concluída ---")
    print(f"Geradas: {valid}")
    print(f"Falhas: {failed}")
    print(
        f"Saída: {output_dir.resolve()}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Visualiza labels YOLO Segment "
            "sobre as imagens."
        )
    )

    parser.add_argument(
        "dataset",
        type=str,
        help=(
            "Diretório aifloodsense_yolo"
        ),
    )

    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="annotation_validation",
    )

    parser.add_argument(
        "-n",
        "--samples",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    if args.samples <= 0:
        parser.error(
            "--samples precisa ser maior que zero"
        )

    validate_annotations(
        dataset_dir=args.dataset,
        output_dir=args.output,
        sample_size=args.samples,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()