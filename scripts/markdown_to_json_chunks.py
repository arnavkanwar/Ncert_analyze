"""CLI runner to convert chapter markdown into paragraph-level JSON chunks."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.processing.markdown_chunk_converter import MarkdownChunkConverter


def setup_logging(level: str) -> None:
    """Configure logging for console and file output."""
    Path("logs").mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("logs/markdown_to_json_chunks.log", encoding="utf-8"),
        ],
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(
        description="Convert chapter markdown to paragraph-level JSON chunks for NCERT retrieval."
    )
    parser.add_argument("--input-md", type=Path, required=True, help="Path to chapter markdown file")
    parser.add_argument("--output-json", type=Path, required=True, help="Path for generated chunk JSON")
    parser.add_argument("--board", type=str, default="NCERT", help="Board name (default: NCERT)")
    parser.add_argument("--class", dest="class_name", type=str, default=None, help="Class override, e.g. 10")
    parser.add_argument("--subject", type=str, default=None, help="Subject override")
    parser.add_argument("--book", type=str, default=None, help="Book override")
    parser.add_argument("--chapter", type=str, default=None, help="Chapter override")
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity",
    )
    return parser.parse_args()


def main() -> int:
    """Entry point for markdown to chunk conversion."""
    args = parse_args()
    setup_logging(args.log_level)

    logger = logging.getLogger(__name__)

    try:
        converter = MarkdownChunkConverter()
        metadata = converter.build_base_metadata(
            markdown_file=args.input_md,
            board=args.board,
            class_name=args.class_name,
            subject=args.subject,
            book=args.book,
            chapter=args.chapter,
        )

        chunks = converter.convert(markdown_file=args.input_md, base_metadata=metadata)
        converter.write_output(chunks=chunks, output_file=args.output_json)

        logger.info("Chunking complete: %d paragraph chunks", len(chunks))
        logger.info("Output written to %s", args.output_json)
        return 0

    except Exception as exc:  # pylint: disable=broad-except
        logger.exception("Failed to convert markdown file: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
