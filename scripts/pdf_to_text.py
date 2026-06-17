"""Extract text from NCERT and PYQ PDFs using PyMuPDF."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline.pdf_text import PDFTextExtractor


def configure_logging(level: str) -> None:
    Path("logs").mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("logs/pdf_to_text.log", encoding="utf-8"),
        ],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract text from PDFs to output/raw_text")
    parser.add_argument("--media-dir", type=Path, default=Path("media"))
    parser.add_argument("--raw-text-dir", type=Path, default=Path("output/raw_text"))
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    configure_logging(args.log_level)

    try:
        extractor = PDFTextExtractor(media_dir=args.media_dir, raw_text_dir=args.raw_text_dir)
        manifest = extractor.extract_all()
        logging.info("Completed extraction for %d PDFs", len(manifest))
        return 0
    except Exception as exc:  # pylint: disable=broad-except
        logging.exception("PDF extraction failed: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
