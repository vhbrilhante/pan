from pathlib import Path
import argparse

import cv2


PROJECT_ROOT = Path(__file__).resolve().parent.parent

VIDEO_EXTENSIONS = {
    ".mp4",
    ".avi",
    ".mov",
    ".mkv",
    ".webm",
}


def extract_frames(
    video_path: str | Path,
    interval_seconds: int,
    output_dir: str | Path | None = None,
) -> None:
    if interval_seconds <= 0:
        raise ValueError("O intervalo deve ser maior que 0.")

    video_path = Path(video_path)

    if output_dir is None:
        output_dir = PROJECT_ROOT / "frames" / video_path.stem
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    video = cv2.VideoCapture(str(video_path))

    if not video.isOpened():
        raise ValueError(
            f"Não foi possível abrir o vídeo: {video_path}"
        )

    try:
        fps = video.get(cv2.CAP_PROP_FPS)

        if fps <= 0:
            raise ValueError(
                f"Não foi possível obter o FPS: {video_path}"
            )

        frame_count = video.get(cv2.CAP_PROP_FRAME_COUNT)
        duration = frame_count / fps

        current_time = 0

        extracted_frames = 0
        skipped_similar = 0

 
        last_saved_frame = None

        while current_time < duration:
            video.set(
                cv2.CAP_PROP_POS_MSEC,
                current_time * 1000,
            )

            success, frame = video.read()

            if not success:
                break

            should_save = (
                last_saved_frame is None
                or not frames_are_similar(
                    last_saved_frame,
                    frame,
                )
            )

            if should_save:
                filename = f"frame_{current_time:06d}.jpeg"
                output_path = output_dir / filename

                saved = cv2.imwrite(
                    str(output_path),
                    frame,
                )

                if saved:
                    extracted_frames += 1

                    last_saved_frame = frame.copy()

            else:
                skipped_similar += 1

            current_time += interval_seconds

        print(
            f"[OK] {video_path.name}: "
            f"{extracted_frames} salvos | "
            f"{skipped_similar} similares descartados."
        )

    finally:
        video.release()


def extract_directory(
    videos_dir: str | Path,
    interval_seconds: int,
    output_dir: str | Path,
) -> None:
    videos_dir = Path(videos_dir)
    output_dir = Path(output_dir)

    if not videos_dir.exists():
        raise FileNotFoundError(
            f"Pasta não encontrada: {videos_dir}"
        )

    if not videos_dir.is_dir():
        raise NotADirectoryError(
            f"O caminho não é uma pasta: {videos_dir}"
        )

    videos = [
        path
        for path in videos_dir.iterdir()
        if (
            path.is_file()
            and path.suffix.lower() in VIDEO_EXTENSIONS
        )
    ]

    videos.sort()

    if not videos:
        print(
            f"Nenhum vídeo encontrado em: {videos_dir}"
        )
        return

    print(f"\nVídeos encontrados: {len(videos)}\n")

    successful = 0
    failed = 0

    for index, video_path in enumerate(
        videos,
        start=1,
    ):
        print(
            f"[{index}/{len(videos)}] "
            f"Processando {video_path.name}..."
        )

        video_output_dir = (
            output_dir / video_path.stem
        )

        try:
            extract_frames(
                video_path=video_path,
                interval_seconds=interval_seconds,
                output_dir=video_output_dir,
            )

            successful += 1

        except Exception as error:
            failed += 1

            print(
                f"[ERRO] {video_path.name}: "
                f"{error}"
            )

    print("\n--- Extração concluída ---")
    print(f"Sucesso: {successful}")
    print(f"Falhas: {failed}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Extrai frames de um vídeo ou "
            "de uma pasta contendo vídeos."
        )
    )

    parser.add_argument(
        "input_path",
        type=str,
        help="Caminho para um vídeo ou pasta",
    )

    parser.add_argument(
        "-i",
        "--interval",
        type=int,
        default=5,
        help="Intervalo entre frames em segundos",
    )

    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=str(PROJECT_ROOT / "frames"),
        help="Diretório de saída",
    )

    args = parser.parse_args()

    input_path = Path(args.input_path)

    if args.interval <= 0:
        parser.error(
            "--interval precisa ser maior que zero"
        )

    if input_path.is_file():
        output_dir = (
            Path(args.output) / input_path.stem
        )

        extract_frames(
            video_path=input_path,
            interval_seconds=args.interval,
            output_dir=output_dir,
        )

    elif input_path.is_dir():
        extract_directory(
            videos_dir=input_path,
            interval_seconds=args.interval,
            output_dir=args.output,
        )

    else:
        parser.error(
            f"Caminho não encontrado: {input_path}"
        )

def frames_are_similar(
    frame_a,
    frame_b,
    mean_threshold: float = 0.02,
    changed_ratio_threshold: float = 0.02,
) -> bool:
    comparison_size = (320, 180)

    frame_a = cv2.resize(frame_a, comparison_size)
    frame_b = cv2.resize(frame_b, comparison_size)

    frame_a = cv2.cvtColor(
        frame_a,
        cv2.COLOR_BGR2GRAY,
    )
    frame_b = cv2.cvtColor(
        frame_b,
        cv2.COLOR_BGR2GRAY,
    )

    frame_a = cv2.GaussianBlur(frame_a, (5, 5), 0)
    frame_b = cv2.GaussianBlur(frame_b, (5, 5), 0)

    difference = cv2.absdiff(frame_a, frame_b)

    mean_difference = difference.mean() / 255

    changed_pixels = difference > 20
    changed_ratio = changed_pixels.mean()

    return (
        mean_difference < mean_threshold
        and changed_ratio < changed_ratio_threshold
    )

if __name__ == "__main__":
    main()