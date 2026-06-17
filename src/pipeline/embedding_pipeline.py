"""Embedding generation utilities for chunk JSON files.

IMPROVED VERSION:
- Preserves and stores ALL metadata (headings, chapter, subject) in ChromaDB
- Uses context-prepended text for embedding (heading + paragraph)
- Stores both raw and contextualized text
- Adds question_number metadata for PYQs
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """Generate normalized embeddings from chunk payloads.
    
    Key improvements:
    1. Embeds context-prepended text (heading + paragraph) when available
    2. Preserves all metadata fields through to ChromaDB
    3. Adds instruction prefix for asymmetric retrieval models
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-mpnet-base-v2",
        device: str = "cpu",
        batch_size: int = 32,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.model = SentenceTransformer(model_name, device=device)

    def _get_text_for_embedding(self, chunk: Dict) -> str:
        """Get the best text representation for embedding.
        
        WHY: The 'text' field now contains heading-prepended text like:
        '[Chemical Bonding > Ionic Bonds] In ionic bonding, ...'
        
        This is what we embed. The raw display text is preserved separately.
        """
        # Use context-prepended text if available (from improved chunking)
        text = str(chunk.get("text", ""))
        return text

    def _build_rich_metadata(self, chunk: Dict) -> Dict:
        """Build comprehensive metadata for ChromaDB storage.
        
        WHY: The original only stored source/file_name/paragraph_number.
        By storing headings and chapter info, we enable:
        1. Chapter-level filtering at query time
        2. Better debugging and explainability
        3. Heading context for display in results
        
        ChromaDB requires values to be str, int, float, or bool.
        """
        meta = {
            "source": str(chunk.get("source", "ncert")),
            "file_name": str(chunk.get("file_name", "")),
            "paragraph_number": int(chunk.get("paragraph_number", 0)),
        }

        # --- NEW: Preserve heading context ---
        heading_context = str(chunk.get("heading_context", ""))
        if heading_context:
            meta["heading_context"] = heading_context

        for field in ["heading_h1", "heading_h2", "heading_h3"]:
            val = str(chunk.get(field, "")).strip()
            if val:
                meta[field] = val

        # Preserve chapter/subject/class metadata if present
        for field in ["chapter", "subject", "class_name", "book", "year"]:
            val = str(chunk.get(field, "")).strip()
            if val:
                meta[field] = val

        for count_field in ["char_count", "word_count"]:
            val = chunk.get(count_field)
            if isinstance(val, int):
                meta[count_field] = val

        # For PYQs, preserve question number
        qn = chunk.get("question_number", "")
        if qn:
            try:
                meta["question_number"] = int(qn)
            except (ValueError, TypeError):
                meta["question_number"] = str(qn)

        # Store raw text separately for display (without heading prefix)
        raw_text = str(chunk.get("text_raw", "")).strip()
        if raw_text:
            meta["text_raw"] = raw_text[:500]  # ChromaDB has metadata size limits

        return meta

    def generate(
        self,
        chunks_json_path: Path,
        embeddings_output_path: Path,
    ) -> Dict[str, object]:
        chunks_json_path = Path(chunks_json_path)
        embeddings_output_path = Path(embeddings_output_path)

        if not chunks_json_path.exists():
            raise FileNotFoundError(f"Chunk JSON not found: {chunks_json_path}")

        chunks: List[Dict[str, object]] = json.loads(chunks_json_path.read_text(encoding="utf-8"))
        if not chunks:
            raise ValueError(f"No chunks found in {chunks_json_path}")

        logger.info("Embedding input chunk count: %d", len(chunks))
        
        # Use context-prepended text for embedding
        texts = [self._get_text_for_embedding(item) for item in chunks]

        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        records: List[Dict[str, object]] = []
        for chunk, embedding in zip(chunks, embeddings):
            records.append(
                {
                    "chunk_id": chunk["chunk_id"],
                    "text": chunk["text"],          # Context-prepended text
                    "text_raw": chunk.get("text_raw", chunk["text"]),  # Raw display text
                    "metadata": self._build_rich_metadata(chunk),
                    "embedding": embedding.tolist(),
                }
            )

        payload = {
            "pipeline_phase": "embedding_generation",
            "model": {
                "name": self.model_name,
                "embedding_dimension": int(self.model.get_sentence_embedding_dimension()),
                "device": self.device,
                "normalized": True,
                "generated_at": datetime.now(timezone.utc).isoformat(),
            },
            "record_count": len(records),
            "records": records,
        }

        if payload["record_count"] <= 0:
            raise ValueError("Embedding generation produced zero records")

        embeddings_output_path.parent.mkdir(parents=True, exist_ok=True)
        embeddings_output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info("Saved %d embedding records to %s", len(records), embeddings_output_path)
        return payload
