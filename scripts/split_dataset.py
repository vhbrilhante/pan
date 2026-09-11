from pathlib import Path
import argparse
import csv
import random
import shutil


PROJECT_ROOT = Path(__file__).resolve().parent.parent

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
}


def get_images(video_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in video_dir.iterdir()
        if (
            path.is_file()
            and path.suffix.lower() in IMAGE_EXTENSIONS
        )
    )


def get_video_directories(
    frames_dir: Path,
) -> list[Path]:
    return sorted(
        directory
        for directory in frames_dir.iterdir()
        if (
            directory.is_dir()
            and get_images(directory)
        )
    )


def split_videos(
    video_dirs: list[Path],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> dict[str, list[Path]]:
    rng = random.Random(seed)

    videos = video_dirs.copy()
    rng.shuffle(videos)

    total = len(videos)

    train_count = int(total * train_ratio)
    val_count = int(total * val_ratio)

    train_videos = videos[:train_count]

    val_videos = videos[
        train_count:train_count + val_count
    ]

    test_videos = videos[
        train_count + val_count:
    ]

    return {
        "train": train_videos,
        "val": val_videos,
        "test": test_videos,
    }


def balanced_sample(
    video_dirs: list[Path],
    target_images: int,
    rng: random.Random,
) -> list[tuple[Path, Path]]:
    """
    Seleciona frames de forma aproximadamente equilibrada
    entre os vídeos.

    Retorna:
        [
            (video_directory, image_path),
            ...
        ]
    """

    pools: dict[Path, list[Path]] = {}

    for video_dir in video_dirs:
        images = get_images(video_dir)

        rng.shuffle(images)

        pools[video_dir] = images

    videos = list(video_dirs)
    rng.shuffle(videos)

    selected: list[tuple[Path, Path]] = []

    while len(selected) < target_images:
        added_something = False

        for video_dir in videos:
            if len(selected) >= target_images:
                break

            pool = pools[video_dir]

            if not pool:
                continue

            image = pool.pop()

            selected.append(
                (video_dir, image)
            )

            added_something = True

        if not added_something:
            break

    return selected


def copy_split(
    split_name: str,
    selected_images: list[tuple[Path, Path]],
    output_dir: Path,
    manifest_rows: list[dict],
) -> None:
    split_dir = (
        output_dir
        / "images"
        / split_name
    )

    split_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    for video_dir, image_path in selected_images:
        # Evita colisão:
        #
        # frame_000005.jpeg existe em praticamente
        # todos os vídeos.
        #
        # Resultado:
        # 001_xxx__frame_000005.jpeg

        output_filename = (
            f"{video_dir.name}"
            f"__{image_path.name}"
        )

        destination = (
            split_dir
            / output_filename
        )

        shutil.copy2(
            image_path,
            destination,
        )

        manifest_rows.append(
            {
                "split": split_name,
                "video": video_dir.name,
                "original_frame": image_path.name,
                "source_path": str(
                    image_path.resolve()
                ),
                "dataset_file": str(
                    destination.resolve()
                ),
            }
        )


def save_manifest(
    rows: list[dict],
    output_dir: Path,
) -> None:
    manifest_path = (
        output_dir
        / "split_manifest.csv"
    )

    with manifest_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "split",
                "video",
                "original_frame",
                "source_path",
                "dataset_file",
            ],
        )

        writer.writeheader()
        writer.writerows(rows)

    print(
        f"\nManifest salvo em: "
        f"{manifest_path}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Divide frames em train/val/test "
            "mantendo vídeos inteiros no mesmo split."
        )
    )

    parser.add_argument(
        "--frames",
        type=str,
        default=str(
            PROJECT_ROOT / "frames"
        ),
        help="Diretório contendo os frames",
    )

    parser.add_argument(
        "--output",
        type=str,
        default=str(
            PROJECT_ROOT / "dataset_raw"
        ),
        help="Diretório do dataset gerado",
    )

    parser.add_argument(
        "--max-images",
        type=int,
        default=1500,
        help=(
            "Número máximo de imagens "
            "selecionadas"
        ),
    )

    parser.add_argument(
        "--train",
        type=float,
        default=0.70,
    )

    parser.add_argument(
        "--val",
        type=float,
        default=0.15,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    frames_dir = Path(args.frames)
    output_dir = Path(args.output)

    if not frames_dir.exists():
        parser.error(
            f"Diretório não encontrado: "
            f"{frames_dir}"
        )

    if args.max_images <= 0:
        parser.error(
            "--max-images precisa ser maior que zero"
        )

    if args.train <= 0 or args.val <= 0:
        parser.error(
            "Train e val precisam ser maiores que zero"
        )

    test_ratio = (
        1.0
        - args.train
        - args.val
    )

    if test_ratio <= 0:
        parser.error(
            "Train + val precisam somar menos que 1"
        )

    video_dirs = get_video_directories(
        frames_dir
    )

    if len(video_dirs) < 3:
        parser.error(
            "São necessários pelo menos 3 vídeos."
        )

    print(
        f"Vídeos encontrados: "
        f"{len(video_dirs)}"
    )

    total_frames = sum(
        len(get_images(video))
        for video in video_dirs
    )

    print(
        f"Frames disponíveis: "
        f"{total_frames}"
    )

    splits = split_videos(
        video_dirs=video_dirs,
        train_ratio=args.train,
        val_ratio=args.val,
        seed=args.seed,
    )

    print("\n--- Vídeos por split ---")

    for name, videos in splits.items():
        print(
            f"{name}: {len(videos)}"
        )

    train_target = round(
        args.max_images
        * args.train
    )

    val_target = round(
        args.max_images
        * args.val
    )

    test_target = (
        args.max_images
        - train_target
        - val_target
    )

    targets = {
        "train": train_target,
        "val": val_target,
        "test": test_target,
    }

    rng = random.Random(
        args.seed
    )

    manifest_rows: list[dict] = []

    print(
        "\n--- Selecionando imagens ---"
    )

    for split_name, videos in splits.items():
        selected = balanced_sample(
            video_dirs=videos,
            target_images=targets[
                split_name
            ],
            rng=rng,
        )

        print(
            f"{split_name}: "
            f"{len(selected)} imagens"
        )

        copy_split(
            split_name=split_name,
            selected_images=selected,
            output_dir=output_dir,
            manifest_rows=manifest_rows,
        )

    save_manifest(
        rows=manifest_rows,
        output_dir=output_dir,
    )

    print(
        "\n--- Dataset preparado ---"
    )

    print(
        f"Total selecionado: "
        f"{len(manifest_rows)}"
    )


if __name__ == "__main__":
    main()