"""Embedding and vector storage modules."""

__all__ = ['TextEmbedder', 'Phase3EmbeddingPipeline', 'VectorStore']


def __getattr__(name):
	"""Lazy import heavy modules to keep package import lightweight."""
	if name == 'TextEmbedder':
		from .embedder import TextEmbedder

		return TextEmbedder
	if name == 'Phase3EmbeddingPipeline':
		from .phase3_pipeline import Phase3EmbeddingPipeline

		return Phase3EmbeddingPipeline
	if name == 'VectorStore':
		from .vector_store import VectorStore

		return VectorStore
	raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
