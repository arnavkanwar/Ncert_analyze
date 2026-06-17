"""Convert extracted raw text files into paragraph-level chunk JSON."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline.chunking import TextChunkBuilder


def configure_logging(level: str) -> None:
    Path("logs").mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("logs/text_to_chunks.log", encoding="utf-8"),
        ],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert raw text files to chunks")
    parser.add_argument("--raw-text-dir", type=Path, default=Path("output/raw_text"))
    parser.add_argument("--chunks-dir", type=Path, default=Path("output/chunks"))
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    configure_logging(args.log_level)

    try:
        builder = TextChunkBuilder(raw_text_dir=args.raw_text_dir, chunk_output_dir=args.chunks_dir)
        chunks = builder.build_chunks()
        logging.info("Chunk generation complete with %d total chunks", len(chunks))
        return 0
    except Exception as exc:  # pylint: disable=broad-except
        logging.exception("Chunk generation failed: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
