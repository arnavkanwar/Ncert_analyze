"""Data models for Phase 3 embedding generation."""

from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class ChunkRecord:
    """Canonical input chunk record used by the Phase 3 pipeline."""

    chunk_id: str
    text: str
    metadata: Dict[str, Any]


@dataclass
class EmbeddedChunkRecord:
    """Output record that preserves traceability from text to embedding."""

    chunk_id: str
    text: str
    metadata: Dict[str, Any]
    embedding: List[float]
