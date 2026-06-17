"""Input/output helpers for Phase 3 embedding generation."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

from jsonschema import Draft7Validator

from .phase3_models import ChunkRecord, EmbeddedChunkRecord

logger = logging.getLogger(__name__)


_INPUT_ITEM_SCHEMA = {
    "type": "object",
    "required": ["chunk_id", "text"],
    "properties": {
        "chunk_id": {"type": "string", "minLength": 1},
        "text": {"type": "string", "minLength": 1},
        "board": {"type": ["string", "null"]},
        "class": {"type": ["string", "null"]},
        "subject": {"type": ["string", "null"]},
        "book": {"type": ["string", "null"]},
        "chapter": {"type": ["string", "null"]},
        "heading": {"type": ["string", "null"]},
        "paragraph_number": {"type": ["integer", "null"]}
    },
    "additionalProperties": True
}


_INPUT_SCHEMA = {
    "type": "array",
    "items": _INPUT_ITEM_SCHEMA
}


class Phase3IOError(Exception):
    """Raised when Phase 3 input/output operations fail."""


class Phase3DataLoader:
    """Load and validate processed chunk JSON from Phase 2 outputs."""

    def __init__(self) -> None:
        self.validator = Draft7Validator(_INPUT_SCHEMA)

    def load_chunks(self, input_json_path: Path) -> List[ChunkRecord]:
        """Load chunk records from a JSON file and validate the structure."""
        if not input_json_path.exists():
            raise Phase3IOError(f"Input JSON file not found: {input_json_path}")

        try:
            with input_json_path.open("r", encoding="utf-8") as infile:
                payload = json.load(infile)
        except json.JSONDecodeError as exc:
            raise Phase3IOError(f"Invalid JSON in input file: {exc}") from exc
        except OSError as exc:
            raise Phase3IOError(f"Failed to read input file: {exc}") from exc

        errors = sorted(self.validator.iter_errors(payload), key=lambda err: err.path)
        if errors:
            first_error = errors[0]
            error_path = ".".join(str(part) for part in first_error.path)
            raise Phase3IOError(
                f"Input schema validation failed at '{error_path}': {first_error.message}"
            )

        chunks: List[ChunkRecord] = []
        for item in payload:
            chunk_id = item["chunk_id"].strip()
            text = item["text"].strip()

            if not chunk_id:
                raise Phase3IOError("chunk_id cannot be empty after stripping whitespace")
            if not text:
                raise Phase3IOError(f"text cannot be empty for chunk_id='{chunk_id}'")

            metadata = {
                key: value
                for key, value in item.items()
                if key not in {"chunk_id", "text"}
            }

            chunks.append(ChunkRecord(chunk_id=chunk_id, text=text, metadata=metadata))

        logger.info("Loaded %d validated chunk records from %s", len(chunks), input_json_path)
        return chunks


class Phase3DataWriter:
    """Write embedding output in a clean structure for later indexing."""

    def write_embeddings(
        self,
        output_json_path: Path,
        records: List[EmbeddedChunkRecord],
        model_info: Dict[str, Any],
        schema_version: str,
    ) -> None:
        """Persist embeddings and traceability fields as JSON."""
        output_json_path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "pipeline_phase": "phase3_embedding_generation",
            "schema_version": schema_version,
            "model": model_info,
            "record_count": len(records),
            "records": [
                {
                    "chunk_id": record.chunk_id,
                    "text": record.text,
                    "metadata": record.metadata,
                    "embedding": record.embedding,
                }
                for record in records
            ],
        }

        try:
            with output_json_path.open("w", encoding="utf-8") as outfile:
                json.dump(payload, outfile, indent=2, ensure_ascii=False)
        except OSError as exc:
            raise Phase3IOError(f"Failed to write output JSON: {exc}") from exc

        logger.info("Saved %d embedding records to %s", len(records), output_json_path)
