"""Pipeline helpers for PDF ingestion, chunking, embeddings, and Chroma indexing."""

# Lazy import for PDFTextExtractor since it requires PyMuPDF (fitz)
# which may not be installed in all environments.
try:
    from .pdf_text import PDFTextExtractor
except ImportError:
    PDFTextExtractor = None

from .chunking import TextChunkBuilder
from .embedding_pipeline import EmbeddingGenerator
from .chroma_db import ChromaIndexer

__all__ = [
    "PDFTextExtractor",
    "TextChunkBuilder",
    "EmbeddingGenerator",
    "ChromaIndexer",
]
