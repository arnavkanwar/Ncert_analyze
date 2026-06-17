"""Phase 3 pipeline for batch embedding generation from processed JSON chunks."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from config.config import Config
from src.embedding.embedder import TextEmbedder

from .phase3_io import Phase3DataLoader, Phase3DataWriter
from .phase3_models import EmbeddedChunkRecord

logger = logging.getLogger(__name__)


class Phase3EmbeddingPipeline:
    """Generate embeddings from pre-processed chunk JSON files."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        batch_size: Optional[int] = None,
        normalize_embeddings: Optional[bool] = None,
    ) -> None:
        self.model_name = model_name or Config.EMBEDDING_MODEL
        self.device = device or Config.DEVICE
        self.batch_size = batch_size or int(Config.PHASE3["batch_size"])
        self.normalize_embeddings = (
            normalize_embeddings
            if normalize_embeddings is not None
            else bool(Config.PHASE3["normalize_embeddings"])
        )

        self.loader = Phase3DataLoader()
        self.writer = Phase3DataWriter()
        self.embedder = TextEmbedder(model_name=self.model_name, device=self.device)

    def run(self, input_json_path: Path, output_json_path: Path) -> Path:
        """Execute Phase 3 end-to-end and return output path."""
        start_time = datetime.now(timezone.utc)
        logger.info("Phase 3 started at %s", start_time.isoformat())

        chunks = self.loader.load_chunks(input_json_path=input_json_path)
        texts = [chunk.text for chunk in chunks]

        embeddings = self.embedder.model.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            show_progress_bar=True,
            normalize_embeddings=self.normalize_embeddings,
        )

        embedded_records: List[EmbeddedChunkRecord] = []
        for chunk, embedding in zip(chunks, embeddings):
            embedded_records.append(
                EmbeddedChunkRecord(
                    chunk_id=chunk.chunk_id,
                    text=chunk.text,
                    metadata=chunk.metadata,
                    embedding=embedding.tolist(),
                )
            )

        model_info = {
            "name": self.embedder.model_name,
            "embedding_dimension": self.embedder.get_embedding_dimension(),
            "device": self.embedder.device,
            "normalize_embeddings": self.normalize_embeddings,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

        self.writer.write_embeddings(
            output_json_path=output_json_path,
            records=embedded_records,
            model_info=model_info,
            schema_version=str(Config.PHASE3["schema_version"]),
        )

        end_time = datetime.now(timezone.utc)
        logger.info(
            "Phase 3 finished at %s (duration: %.2fs)",
            end_time.isoformat(),
            (end_time - start_time).total_seconds(),
        )

        return output_json_path
