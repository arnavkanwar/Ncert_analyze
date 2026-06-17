"""Batch ingestion utility for PYQ PDF files."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.ingest_pipeline import run_ingestion


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch ingest all PYQ PDFs from a directory")
    parser.add_argument("--pyq-dir", type=Path, default=Path("data/pyqs"))
    parser.add_argument("--class", dest="class_name", type=str, default=None)
    parser.add_argument("--subject", type=str, default=None)
    parser.add_argument("--year", type=str, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    pyq_dir = Path(args.pyq_dir)

    if not pyq_dir.exists():
        print(f"Directory not found: {pyq_dir}")
        return 1

    pdf_paths = sorted(pyq_dir.glob("*.pdf"))
    if not pdf_paths:
        print(f"No PDFs found in {pyq_dir}")
        return 1

    for pdf_path in pdf_paths:
        print(f"Ingesting: {pdf_path.name}")
        run_ingestion(
            data_path=pdf_path,
            source="pyq",
            class_name=args.class_name,
            subject=args.subject,
            year=args.year,
        )

    print(f"Completed batch ingestion for {len(pdf_paths)} file(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
