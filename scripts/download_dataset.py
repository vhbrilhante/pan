from pathlib import Path
import argparse

import yt_dlp

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def get_playlist_info(playlist_url: str, limit: int | None = None) -> None:
    ydl_opts = {
        "extract_flat": True,
        "skip_download": True,
    }

    if limit is not None:
        ydl_opts["playlistend"] = limit

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        playlist_dict = ydl.extract_info(
            playlist_url,
            download=False,
        )

    entries = playlist_dict.get("entries", [])
    playlist_title = playlist_dict.get("title", "Unknown Playlist")

    entries_to_show = entries[:limit] if limit is not None else entries

    print(f"\n--- Playlist: {playlist_title} ---")
    print(f"Total de vídeos: {len(entries)}")
    print(f"Vídeos selecionados: {len(entries_to_show)}\n")

    for index, video in enumerate(entries_to_show, start=1):
        if video is None:
            continue

        video_title = video.get("title", "Unknown title")
        video_id = video.get("id", "Unknown ID")

        print(
            f"[{index:03d}] {video_title} | "
            f"https://youtu.be/{video_id}"
        )


def download_playlist(
    playlist_url: str,
    output_dir: str | Path,
    limit: int | None = None,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    archive_path = output_dir / "download_archive.txt"

    ydl_opts = {
        "format": (
            "bestvideo[height<=720][ext=mp4][vcodec^=avc1]"
            "/bestvideo[height<=720][ext=mp4]"
            "/bestvideo[height<=720]"
        ),

        "outtmpl": str(
            output_dir / "%(playlist_index)03d_%(id)s.%(ext)s"
        ),

        "download_archive": str(archive_path),

        "writeinfojson": True,

        "ignoreerrors": True,
    }

    if limit is not None:
        ydl_opts["playlistend"] = limit

    print("\n--- Iniciando download ---")

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        error_code = ydl.download([playlist_url])

    if error_code:
        print("\nAlguns vídeos não puderam ser baixados.")
    else:
        print("\nDownload concluído.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="List and download videos from a YouTube playlist."
    )

    parser.add_argument(
        "playlist_url",
        type=str,
        help="URL da playlist",
    )

    parser.add_argument(
        "-l",
        "--limit",
        type=int,
        default=None,
        help="Limite de vídeos que serão listados/baixados",
    )

    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=str(PROJECT_ROOT / "videos"),
        help="Diretório onde os vídeos serão salvos",
    )

    parser.add_argument(
        "--download",
        action="store_true",
        help="Baixa os vídeos selecionados",
    )

    args = parser.parse_args()

    if args.limit is not None and args.limit <= 0:
        parser.error("--limit precisa ser maior que zero")

    get_playlist_info(
        playlist_url=args.playlist_url,
        limit=args.limit,
    )

    if args.download:
        download_playlist(
            playlist_url=args.playlist_url,
            output_dir=args.output,
            limit=args.limit,
        )


if __name__ == "__main__":
    main()