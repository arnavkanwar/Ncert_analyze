"""Generate embeddings for chunk JSON files."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline.embedding_pipeline import EmbeddingGenerator


def configure_logging(level: str) -> None:
    Path("logs").mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("logs/generate_embeddings.log", encoding="utf-8"),
        ],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate embeddings from chunk JSON.")
    parser.add_argument("--chunks-json", type=Path, default=Path("output/chunks/all_chunks.json"))
    parser.add_argument("--output-json", type=Path, default=Path("output/embeddings/all_embeddings.json"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    configure_logging(args.log_level)

    try:
        generator = EmbeddingGenerator(batch_size=args.batch_size, device=args.device)
        generator.generate(chunks_json_path=args.chunks_json, embeddings_output_path=args.output_json)
        logging.info("Embedding generation complete.")
        return 0
    except Exception as exc:  # pylint: disable=broad-except
        logging.exception("Embedding generation failed: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
