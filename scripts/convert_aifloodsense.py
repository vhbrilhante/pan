from pathlib import Path
import argparse
import csv
import shutil

import cv2
import numpy as np


FLOOD_VALUE = 255
CLASS_ID = 0

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
}


def mask_to_polygons(
    mask_path: Path,
    min_area: float = 100.0,
    epsilon_factor: float = 0.002,
) -> list[list[float]]:
    mask = cv2.imread(
        str(mask_path),
        cv2.IMREAD_GRAYSCALE,
    )

    if mask is None:
        raise ValueError(
            f"Não foi possível abrir a máscara: {mask_path}"
        )

    height, width = mask.shape

    # Só Flood = 255 interessa.
    flood_mask = np.where(
        mask == FLOOD_VALUE,
        255,
        0,
    ).astype(np.uint8)

    contours, _ = cv2.findContours(
        flood_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    polygons: list[list[float]] = []

    for contour in contours:
        area = cv2.contourArea(contour)

        if area < min_area:
            continue

        perimeter = cv2.arcLength(
            contour,
            True,
        )

        epsilon = (
            epsilon_factor
            * perimeter
        )

        approximated = cv2.approxPolyDP(
            contour,
            epsilon,
            True,
        )

        # Polígono precisa de pelo menos 3 pontos.
        if len(approximated) < 3:
            continue

        polygon: list[float] = []

        for point in approximated:
            x, y = point[0]

            normalized_x = x / width
            normalized_y = y / height

            polygon.extend([
                normalized_x,
                normalized_y,
            ])

        polygons.append(polygon)

    return polygons


def save_yolo_label(
    polygons: list[list[float]],
    label_path: Path,
) -> None:
    label_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with label_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        for polygon in polygons:
            coordinates = " ".join(
                f"{value:.6f}"
                for value in polygon
            )

            file.write(
                f"{CLASS_ID} {coordinates}\n"
            )


def convert_split(
    source_root: Path,
    source_split: str,
    output_root: Path,
    manifest_rows: list[dict],
    min_area: float,
) -> tuple[int, int]:
    images_dir = (
        source_root
        / source_split
        / "images"
    )

    masks_dir = (
        source_root
        / source_split
        / "masks"
    )

    if not images_dir.exists():
        raise FileNotFoundError(
            f"Imagens não encontradas: {images_dir}"
        )

    if not masks_dir.exists():
        raise FileNotFoundError(
            f"Máscaras não encontradas: {masks_dir}"
        )

    output_images = (
        output_root
        / "images"
        / "all"
    )

    output_labels = (
        output_root
        / "labels"
        / "all"
    )

    output_images.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_labels.mkdir(
        parents=True,
        exist_ok=True,
    )

    images = sorted(
        path
        for path in images_dir.iterdir()
        if (
            path.is_file()
            and path.suffix.lower()
            in IMAGE_EXTENSIONS
        )
    )

    converted = 0
    empty = 0

    for index, image_path in enumerate(
        images,
        start=1,
    ):
        image_id = image_path.stem

        mask_path = (
            masks_dir
            / f"{image_id}.png"
        )

        if not mask_path.exists():
            print(
                f"[WARNING] Máscara ausente: "
                f"{mask_path.name}"
            )
            continue

        polygons = mask_to_polygons(
            mask_path=mask_path,
            min_area=min_area,
        )

        output_name = (
            f"aifloodsense_{image_id}"
        )

        output_image_path = (
            output_images
            / f"{output_name}"
            f"{image_path.suffix.lower()}"
        )

        output_label_path = (
            output_labels
            / f"{output_name}.txt"
        )

        shutil.copy2(
            image_path,
            output_image_path,
        )

        save_yolo_label(
            polygons=polygons,
            label_path=output_label_path,
        )

        if polygons:
            converted += 1
        else:
            empty += 1

        manifest_rows.append(
            {
                "image_id": image_id,
                "dataset_filename": (
                    output_image_path.name
                ),
                "source_split": source_split,
                "polygon_count": len(polygons),
            }
        )

        print(
            f"[{source_split}] "
            f"{index}/{len(images)} "
            f"{image_path.name} -> "
            f"{len(polygons)} região(ões)"
        )

    return converted, empty


def save_manifest(
    rows: list[dict],
    output_root: Path,
) -> None:
    manifest_path = (
        output_root
        / "manifest.csv"
    )

    with manifest_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "image_id",
                "dataset_filename",
                "source_split",
                "polygon_count",
            ],
        )

        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Converte máscaras do AIFloodSense "
            "para labels YOLO Segment."
        )
    )

    parser.add_argument(
        "source",
        type=str,
        help=(
            "Diretório raiz do AIFloodSense"
        ),
    )

    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="aifloodsense_yolo",
        help="Diretório de saída",
    )

    parser.add_argument(
        "--min-area",
        type=float,
        default=100.0,
        help=(
            "Área mínima em pixels para manter "
            "um contorno"
        ),
    )

    args = parser.parse_args()

    source_root = Path(args.source)
    output_root = Path(args.output)

    manifest_rows: list[dict] = []

    total_converted = 0
    total_empty = 0

    for split in ("train", "test"):
        converted, empty = convert_split(
            source_root=source_root,
            source_split=split,
            output_root=output_root,
            manifest_rows=manifest_rows,
            min_area=args.min_area,
        )

        total_converted += converted
        total_empty += empty

    save_manifest(
        rows=manifest_rows,
        output_root=output_root,
    )

    print()
    print("--- Conversão concluída ---")
    print(
        f"Imagens com Flood: "
        f"{total_converted}"
    )
    print(
        f"Imagens sem polígono: "
        f"{total_empty}"
    )
    print(
        f"Total processado: "
        f"{len(manifest_rows)}"
    )
    print(
        f"Saída: "
        f"{output_root.resolve()}"
    )


if __name__ == "__main__":
    main()