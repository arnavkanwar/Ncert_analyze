<<<<<<< HEAD
"""Master runner for full NCERT + PYQ semantic search pipeline."""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from pathlib import Path


def run_step(name: str, command: list[str]) -> None:
    logging.info("Running step: %s", name)
    logging.info("Command: %s", " ".join(command))
    result = subprocess.run(command, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Step failed: {name} (exit code {result.returncode})")
    logging.info("Completed step: %s", name)


def validate_outputs() -> None:
    manifest = Path("output/raw_text/manifest.json")
    chunks = Path("output/chunks/all_chunks.json")
    embeddings = Path("output/embeddings/all_embeddings.json")

    if not manifest.exists():
        raise RuntimeError("Validation failed: output/raw_text/manifest.json not found")

    if not chunks.exists():
        raise RuntimeError("Validation failed: output/chunks/all_chunks.json not found")

    if not embeddings.exists():
        raise RuntimeError("Validation failed: output/embeddings/all_embeddings.json not found")

    chunks_data = json.loads(chunks.read_text(encoding="utf-8"))
    emb_data = json.loads(embeddings.read_text(encoding="utf-8"))

    logging.info("Validation: chunk count = %d", len(chunks_data))
    logging.info("Validation: embedding record_count = %d", int(emb_data.get("record_count", 0)))

    if len(chunks_data) == 0:
        raise RuntimeError("Validation failed: all_chunks.json is empty")

    if int(emb_data.get("record_count", 0)) <= 0:
        raise RuntimeError("Validation failed: embedding record_count is zero")


def main() -> int:
    Path("logs").mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler("logs/run_pipeline.log", encoding="utf-8")],
    )

    python_exe = sys.executable

    try:
        run_step("PDF to text", [python_exe, "scripts/pdf_to_text.py", "--media-dir", "media", "--raw-text-dir", "output/raw_text"])
        run_step("Text to chunks", [python_exe, "scripts/text_to_chunks.py", "--raw-text-dir", "output/raw_text", "--chunks-dir", "output/chunks"])
        run_step("Chunks to embeddings", [python_exe, "scripts/generate_embeddings.py", "--chunks-json", "output/chunks/all_chunks.json", "--output-json", "output/embeddings/all_embeddings.json", "--device", "cpu"])
        run_step("Embeddings to ChromaDB", [python_exe, "scripts/chroma_store.py", "--embeddings-json", "output/embeddings/all_embeddings.json", "--chroma-dir", "chroma_db", "--collection", "ncert_chemistry", "--reset"])

        validate_outputs()
        logging.info("Full pipeline completed successfully.")
        return 0

    except Exception as exc:  # pylint: disable=broad-except
        logging.exception("Pipeline failed: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
=======
"""Master runner for full NCERT + PYQ semantic search pipeline."""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from pathlib import Path


def run_step(name: str, command: list[str]) -> None:
    logging.info("Running step: %s", name)
    logging.info("Command: %s", " ".join(command))
    result = subprocess.run(command, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Step failed: {name} (exit code {result.returncode})")
    logging.info("Completed step: %s", name)


def validate_outputs() -> None:
    manifest = Path("output/raw_text/manifest.json")
    chunks = Path("output/chunks/all_chunks.json")
    embeddings = Path("output/embeddings/all_embeddings.json")

    if not manifest.exists():
        raise RuntimeError("Validation failed: output/raw_text/manifest.json not found")

    if not chunks.exists():
        raise RuntimeError("Validation failed: output/chunks/all_chunks.json not found")

    if not embeddings.exists():
        raise RuntimeError("Validation failed: output/embeddings/all_embeddings.json not found")

    chunks_data = json.loads(chunks.read_text(encoding="utf-8"))
    emb_data = json.loads(embeddings.read_text(encoding="utf-8"))

    logging.info("Validation: chunk count = %d", len(chunks_data))
    logging.info("Validation: embedding record_count = %d", int(emb_data.get("record_count", 0)))

    if len(chunks_data) == 0:
        raise RuntimeError("Validation failed: all_chunks.json is empty")

    if int(emb_data.get("record_count", 0)) <= 0:
        raise RuntimeError("Validation failed: embedding record_count is zero")


def main() -> int:
    Path("logs").mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler("logs/run_pipeline.log", encoding="utf-8")],
    )

    python_exe = sys.executable

    try:
        run_step("PDF to text", [python_exe, "scripts/pdf_to_text.py", "--media-dir", "media", "--raw-text-dir", "output/raw_text"])
        run_step("Text to chunks", [python_exe, "scripts/text_to_chunks.py", "--raw-text-dir", "output/raw_text", "--chunks-dir", "output/chunks"])
        run_step("Chunks to embeddings", [python_exe, "scripts/generate_embeddings.py", "--chunks-json", "output/chunks/all_chunks.json", "--output-json", "output/embeddings/all_embeddings.json", "--device", "cpu"])
        run_step("Embeddings to ChromaDB", [python_exe, "scripts/chroma_store.py", "--embeddings-json", "output/embeddings/all_embeddings.json", "--chroma-dir", "chroma_db", "--collection", "ncert_chemistry", "--reset"])

        validate_outputs()
        logging.info("Full pipeline completed successfully.")
        return 0

    except Exception as exc:  # pylint: disable=broad-except
        logging.exception("Pipeline failed: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
>>>>>>> b87913a (Initial commit)
