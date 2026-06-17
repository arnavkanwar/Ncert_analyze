"""ChromaDB indexing and retrieval helpers for semantic search.

IMPROVED VERSION:
- Stores ALL metadata fields (headings, chapter, subject, class)
- Supports richer metadata for filtering at query time
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import chromadb
from chromadb.config import Settings

logger = logging.getLogger(__name__)


class ChromaIndexer:
    """Index embeddings and run vector search against persistent ChromaDB.
    
    Improvements:
    1. Stores full metadata including heading_context, chapter, subject
    2. Metadata sanitization ensures ChromaDB compatibility
    """

    def __init__(
        self,
        persist_dir: Path,
        collection_name: str = "ncert_chemistry",
    ) -> None:
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        self.client = chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection_name = collection_name
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def reset_collection(self) -> None:
        try:
            self.client.delete_collection(self.collection_name)
        except Exception:  # pylint: disable=broad-except
            pass
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    @staticmethod
    def _sanitize_metadata(meta: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure all metadata values are ChromaDB-compatible types.
        
        ChromaDB only accepts: str, int, float, bool.
        """
        sanitized = {}
        for key, value in meta.items():
            if value is None:
                continue  # Skip None values entirely
            if isinstance(value, (str, int, float, bool)):
                # Skip empty strings to keep metadata clean
                if isinstance(value, str) and not value.strip():
                    continue
                sanitized[key] = value
            else:
                str_val = str(value).strip()
                if str_val:
                    sanitized[key] = str_val
        return sanitized

    def index_embeddings(self, embeddings_json_path: Path, batch_size: int = 128) -> int:
        payload = json.loads(Path(embeddings_json_path).read_text(encoding="utf-8"))
        records: List[Dict[str, Any]] = payload.get("records", [])
        total = 0

        for idx in range(0, len(records), batch_size):
            batch = records[idx : idx + batch_size]
            ids = [row["chunk_id"] for row in batch]
            docs = [row["text"] for row in batch]
            embs = [row["embedding"] for row in batch]
            
            # --- IMPROVED: Store ALL metadata, not just 3 fields ---
            metas = [
                self._sanitize_metadata(row.get("metadata", {}))
                for row in batch
            ]
            
            self.collection.upsert(ids=ids, documents=docs, embeddings=embs, metadatas=metas)
            total += len(batch)

        logger.info("Indexed %d records into collection '%s'", total, self.collection_name)
        logger.info("Collection '%s' total records now: %d", self.collection_name, self.collection.count())
        return total

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        filter_metadata: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        return self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=filter_metadata,
            include=["documents", "metadatas", "distances", "embeddings"],
        )
