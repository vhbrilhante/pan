from pathlib import Path
from collections import defaultdict
import argparse
import csv
import random
import shutil


def read_metadata(
    csv_path: Path,
    source_split: str,
) -> list[dict]:
    rows: list[dict] = []

    with csv_path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        reader = csv.DictReader(file)

        for row in reader:
            row["SOURCE_SPLIT"] = source_split
            rows.append(row)

    return rows


def group_by_event(
    metadata: list[dict],
) -> dict[str, list[dict]]:
    events: dict[str, list[dict]] = defaultdict(list)

    for row in metadata:
        event_name = row["FLOOD_NAME"].strip()
        events[event_name].append(row)

    return dict(events)


def split_events(
    events: dict[str, list[dict]],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> dict[str, list[dict]]:
    test_ratio = 1.0 - train_ratio - val_ratio

    total_images = sum(
        len(images)
        for images in events.values()
    )

    targets = {
        "train": total_images * train_ratio,
        "val": total_images * val_ratio,
        "test": total_images * test_ratio,
    }

    result = {
        "train": [],
        "val": [],
        "test": [],
    }

    counts = {
        "train": 0,
        "val": 0,
        "test": 0,
    }

    event_items = list(events.items())

    rng = random.Random(seed)
    rng.shuffle(event_items)

    # Eventos maiores primeiro.
    # O shuffle anterior resolve empates de forma
    # reproduzível.
    event_items.sort(
        key=lambda item: len(item[1]),
        reverse=True,
    )

    for event_name, rows in event_items:
        # Escolhe o split proporcionalmente menos cheio.
        split_name = min(
            counts,
            key=lambda split: (
                counts[split] / targets[split]
            ),
        )

        for row in rows:
            row["SPLIT"] = split_name
            result[split_name].append(row)

        counts[split_name] += len(rows)

    return result


def find_image(
    images_dir: Path,
    image_id: str,
) -> Path | None:
    stem = f"aifloodsense_{image_id}"

    for extension in (
        ".jpg",
        ".jpeg",
        ".png",
    ):
        candidate = images_dir / f"{stem}{extension}"

        if candidate.exists():
            return candidate

    return None


def copy_dataset(
    splits: dict[str, list[dict]],
    converted_dataset: Path,
    output_dir: Path,
) -> list[dict]:
    source_images = (
        converted_dataset
        / "images"
        / "all"
    )

    source_labels = (
        converted_dataset
        / "labels"
        / "all"
    )

    manifest: list[dict] = []

    for split_name, rows in splits.items():
        destination_images = (
            output_dir
            / "images"
            / split_name
        )

        destination_labels = (
            output_dir
            / "labels"
            / split_name
        )

        destination_images.mkdir(
            parents=True,
            exist_ok=True,
        )

        destination_labels.mkdir(
            parents=True,
            exist_ok=True,
        )

        for row in rows:
            image_id = row["IMAGE_ID"].strip()

            image_path = find_image(
                source_images,
                image_id,
            )

            label_path = (
                source_labels
                / f"aifloodsense_{image_id}.txt"
            )

            if image_path is None:
                print(
                    f"[WARNING] Imagem ausente: "
                    f"{image_id}"
                )
                continue

            if not label_path.exists():
                print(
                    f"[WARNING] Label ausente: "
                    f"{image_id}"
                )
                continue

            shutil.copy2(
                image_path,
                destination_images / image_path.name,
            )

            shutil.copy2(
                label_path,
                destination_labels / label_path.name,
            )

            manifest.append({
                "image_id": image_id,
                "split": split_name,
                "flood_name": row["FLOOD_NAME"],
                "year": row["YEAR"],
                "country": row["COUNTRY"],
                "continent": row["CONTINENT"],
                "environment": row["ENVIRONMENT"],
                "sky": row["SKY"],
                "latitude": row["LATITUDE"],
                "longitude": row["LONGITUDE"],
                "source": row["SOURCE"],
                "original_split": row["SOURCE_SPLIT"],
            })

    return manifest


def save_manifest(
    manifest: list[dict],
    output_dir: Path,
) -> None:
    path = output_dir / "split_manifest.csv"

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=manifest[0].keys(),
        )

        writer.writeheader()
        writer.writerows(manifest)


def save_yaml(
    output_dir: Path,
) -> None:
    yaml_path = output_dir / "data.yaml"

    content = f"""path: {output_dir.resolve().as_posix()}

train: images/train
val: images/val
test: images/test

names:
  0: flooded_area
"""

    yaml_path.write_text(
        content,
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Cria split train/val/test do "
            "AIFloodSense por evento."
        )
    )

    parser.add_argument(
        "dataset",
        type=str,
        help="Diretório aifloodsense_yolo",
    )

    parser.add_argument(
        "train_csv",
        type=str,
    )

    parser.add_argument(
        "test_csv",
        type=str,
    )

    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="aifloodsense_dataset",
    )

    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.70,
    )

    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    dataset = Path(args.dataset)
    output = Path(args.output)

    train_metadata = read_metadata(
        Path(args.train_csv),
        "original_train",
    )

    test_metadata = read_metadata(
        Path(args.test_csv),
        "original_test",
    )

    metadata = (
        train_metadata
        + test_metadata
    )

    events = group_by_event(metadata)

    print(
        f"Imagens com metadata: {len(metadata)}"
    )

    print(
        f"Eventos únicos: {len(events)}"
    )

    splits = split_events(
        events=events,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    print("\n--- Novo split ---")

    for split_name, rows in splits.items():
        event_names = {
            row["FLOOD_NAME"]
            for row in rows
        }

        print(
            f"{split_name}: "
            f"{len(rows)} imagens | "
            f"{len(event_names)} eventos"
        )

    # Garantia contra vazamento de eventos.
    train_events = {
        row["FLOOD_NAME"]
        for row in splits["train"]
    }

    val_events = {
        row["FLOOD_NAME"]
        for row in splits["val"]
    }

    test_events = {
        row["FLOOD_NAME"]
        for row in splits["test"]
    }

    assert train_events.isdisjoint(val_events)
    assert train_events.isdisjoint(test_events)
    assert val_events.isdisjoint(test_events)

    manifest = copy_dataset(
        splits=splits,
        converted_dataset=dataset,
        output_dir=output,
    )

    save_manifest(
        manifest,
        output,
    )

    save_yaml(output)

    print()
    print("--- Dataset finalizado ---")
    print(f"Copiadas: {len(manifest)}")
    print(f"Saída: {output.resolve()}")


if __name__ == "__main__":
    main()