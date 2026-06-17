"""Load embeddings JSON and index into ChromaDB."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline.chroma_db import ChromaIndexer


def configure_logging(level: str) -> None:
    Path("logs").mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("logs/chroma_store.log", encoding="utf-8"),
        ],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Index embeddings into ChromaDB")
    parser.add_argument("--embeddings-json", type=Path, default=Path("output/embeddings/all_embeddings.json"))
    parser.add_argument("--chroma-dir", type=Path, default=Path("chroma_db"))
    parser.add_argument("--collection", type=str, default="ncert_chemistry")
    parser.add_argument("--reset", action="store_true")
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    configure_logging(args.log_level)

    try:
        indexer = ChromaIndexer(persist_dir=args.chroma_dir, collection_name=args.collection)
        if args.reset:
            indexer.reset_collection()
            logging.info("Collection reset complete")

        count = indexer.index_embeddings(embeddings_json_path=args.embeddings_json)
        stored = indexer.collection.count()
        logging.info("Indexed %d records", count)
        logging.info("Stored records in collection '%s': %d", args.collection, stored)
        print(f"Stored records in collection '{args.collection}': {stored}")
        return 0
    except Exception as exc:  # pylint: disable=broad-except
        logging.exception("Chroma indexing failed: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
