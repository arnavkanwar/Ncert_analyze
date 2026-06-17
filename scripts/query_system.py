"""PYQ-driven NCERT retrieval CLI with reranking diagnostics."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.retrieval.intelligent_query import IntelligentQueryEngine


def configure_logging(level: str) -> None:
    Path("logs").mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("logs/query_system.log", encoding="utf-8"),
        ],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PYQ to NCERT paragraph retrieval")
    parser.add_argument("--list-pyqs", action="store_true", help="List available PYQs from chunk dataset")
    parser.add_argument("--pyq-id", type=str, default=None, help="Select PYQ by id")
    parser.add_argument("--pyq-text", type=str, default=None, help="Ad-hoc PYQ text input")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--chroma-dir", type=Path, default=Path("chroma_db"))
    parser.add_argument("--collection", type=str, default="ncert_chemistry")
    parser.add_argument("--chunks-path", type=Path, default=Path("output/chunks/all_chunks.json"))
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser.parse_args()


def print_pyq_list(pyqs: list[dict]) -> None:
    print("\nAvailable PYQs:\n")
    for row in pyqs:
        snippet = row["text"][:120].replace("\n", " ")
        print(f"- {row['pyq_id']} | {row['file_name']} | {snippet}...")


def print_result(payload: dict) -> None:
    print("\nSelected PYQ:\n")
    print(payload["selected_pyq"]["text"])

    print("\nBest Matching NCERT Paragraph:\n")
    best = payload["best_matching_ncert_paragraph"]
    if best:
        print(best["text"])
        print(
            f"\nSource: {best['metadata'].get('file_name', '')} | "
            f"Paragraph: {best['metadata'].get('paragraph_number', '')} | "
            f"Score: {best['score']:.4f}"
        )
    else:
        print("No NCERT match found.")

    print("\nOptional Second Supporting Paragraph:\n")
    second = payload["second_supporting_paragraph"]
    if second:
        print(second["text"])
        print(
            f"\nSource: {second['metadata'].get('file_name', '')} | "
            f"Paragraph: {second['metadata'].get('paragraph_number', '')} | "
            f"Score: {second['score']:.4f}"
        )
    else:
        print("None selected.")

    print("\nDebug:\n")
    print(f"Second paragraph status: {payload['debug']['second_paragraph_status']}")
    print("Top cross-encoder scores:")
    for row in payload["debug"]["top_cross_encoder_scores"]:
        print(
            f"  {row['chunk_id']} | {row['score']:.4f} | "
            f"{row['file_name']} | p{row['paragraph_number']}"
        )


def main() -> int:
    args = parse_args()
    configure_logging(args.log_level)

    try:
        engine = IntelligentQueryEngine(
            chroma_dir=args.chroma_dir,
            collection_name=args.collection,
            chunks_path=args.chunks_path,
            device=args.device,
        )

        if args.list_pyqs:
            pyqs = engine.get_pyq_list()
            print_pyq_list(pyqs)
            return 0

        if not args.pyq_id and not args.pyq_text:
            print("Use --list-pyqs to inspect IDs, then run with --pyq-id <id>.")
            return 1

        result = engine.query_from_pyq(pyq_id=args.pyq_id, pyq_text=args.pyq_text, top_k=args.top_k)
        if args.json:
            print(json.dumps(result, indent=2, ensure_ascii=False))
        else:
            print_result(result)
        return 0

    except Exception as exc:  # pylint: disable=broad-except
        logging.exception("Query system failed: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
