"""Print basic NCERT/PYQ collection statistics from ChromaDB."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline.chroma_db import ChromaIndexer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Show ChromaDB collection stats")
    parser.add_argument("--chroma-dir", type=Path, default=Path("chroma_db"))
    parser.add_argument("--collection", type=str, default="ncert_chemistry")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    indexer = ChromaIndexer(persist_dir=args.chroma_dir, collection_name=args.collection)

    total = indexer.collection.count()
    ncert = indexer.collection.get(where={"source": "ncert"}, include=["metadatas"], limit=100000)
    pyq = indexer.collection.get(where={"source": "pyq"}, include=["metadatas"], limit=100000)

    ncert_count = len(ncert.get("ids", []))
    pyq_count = len(pyq.get("ids", []))

    print(f"Total chunks: {total}")
    print(f"NCERT chunks: {ncert_count}")
    print(f"PYQ chunks: {pyq_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
