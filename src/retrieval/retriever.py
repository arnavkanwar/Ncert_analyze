"""
Semantic Retriever Module

Provides high-level interface for semantic search and retrieval.
Includes support for filtering, reranking (future), and result formatting.
"""

from typing import List, Dict, Optional
import logging
from dataclasses import dataclass

from config.config import Config
from src.embedding.embedder import TextEmbedder
from src.embedding.vector_store import VectorStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Represents a single search result."""
    chunk_id: str
    text: str
    similarity_score: float
    metadata: Dict
    rank: int
    
    def get_display_context(self) -> str:
        """Get formatted context for display."""
        parts = []
        
        if self.metadata.get('class_name'):
            parts.append(self.metadata['class_name'].replace('class', 'Class '))
        
        if self.metadata.get('subject'):
            parts.append(self.metadata['subject'].title())
        
        if self.metadata.get('chapter'):
            parts.append(self.metadata['chapter'].replace('ch', 'Ch '))
        
        # Add headings
        headings = []
        for level in ['heading_h1', 'heading_h2', 'heading_h3']:
            if self.metadata.get(level):
                headings.append(self.metadata[level])
        
        if headings:
            parts.append(' > '.join(headings))
        
        return ' | '.join(parts)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "similarity_score": self.similarity_score,
            "rank": self.rank,
            "metadata": self.metadata,
            "display_context": self.get_display_context()
        }


class SemanticRetriever:
    """
    High-level semantic retrieval interface.
    
    Features:
    - Natural language search
    - Metadata filtering
    - Result ranking and formatting
    - Ready for cross-encoder reranking integration
    """
    
    def __init__(
        self,
        embedder: TextEmbedder = None,
        vector_store: VectorStore = None,
        config: Dict = None
    ):
        """
        Initialize SemanticRetriever.
        
        Args:
            embedder: TextEmbedder instance (created if not provided)
            vector_store: VectorStore instance (created if not provided)
            config: Optional configuration dict
        """
        self.config = config or Config.RETRIEVAL
        
        # Initialize components
        self.embedder = embedder or TextEmbedder()
        self.vector_store = vector_store or VectorStore()
        
        # Reranking configuration (for future implementation)
        self.rerank_enabled = self.config.get('rerank', False)
        self.rerank_model = self.config.get('rerank_model', None)
        
        logger.info("SemanticRetriever initialized")
    
    def search(
        self,
        query: str,
        top_k: int = None,
        filters: Dict = None,
        score_threshold: float = None,
        class_filter: str = None,
        subject_filter: str = None,
        chapter_filter: str = None
    ) -> List[SearchResult]:
        """
        Semantic search with optional filtering.
        
        Args:
            query: Natural language query
            top_k: Number of results to return
            filters: Custom metadata filters
            score_threshold: Minimum similarity score
            class_filter: Filter by class (e.g., "class10")
            subject_filter: Filter by subject (e.g., "science")
            chapter_filter: Filter by chapter (e.g., "ch1")
            
        Returns:
            List of SearchResult objects, ranked by relevance
        """
        if not query or not query.strip():
            logger.warning("Empty query provided")
            return []
        
        # Set defaults
        top_k = top_k or self.config.get('top_k', 5)
        score_threshold = score_threshold or self.config.get('score_threshold', 0.0)
        
        # Build metadata filters
        metadata_filters = self._build_filters(
            filters, class_filter, subject_filter, chapter_filter
        )
        
        logger.info(f"Searching for: '{query}' (top_k={top_k}, filters={metadata_filters})")
        
        # Generate query embedding
        query_embedding = self.embedder.embed_text(query)
        
        # Search vector store
        raw_results = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k * 2 if self.rerank_enabled else top_k,  # Get more for reranking
            filter_metadata=metadata_filters,
            score_threshold=score_threshold
        )
        
        # Apply reranking if enabled
        if self.rerank_enabled and self.rerank_model:
            raw_results = self._rerank_results(query, raw_results)
            raw_results = raw_results[:top_k]  # Take top K after reranking
        
        # Format results
        results = []
        for rank, result in enumerate(raw_results, 1):
            search_result = SearchResult(
                chunk_id=result['chunk_id'],
                text=result['text'],
                similarity_score=result['similarity_score'],
                metadata=result['metadata'],
                rank=rank
            )
            results.append(search_result)
        
        logger.info(f"Returning {len(results)} results")
        return results
    
    def search_by_topic(
        self,
        topic: str,
        class_name: str = None,
        subject: str = None,
        top_k: int = 10
    ) -> List[SearchResult]:
        """
        Search for content related to a specific topic.
        
        Convenience method for topic-based retrieval.
        
        Args:
            topic: Topic to search for
            class_name: Optional class filter
            subject: Optional subject filter
            top_k: Number of results
            
        Returns:
            List of SearchResult objects
        """
        return self.search(
            query=topic,
            top_k=top_k,
            class_filter=class_name,
            subject_filter=subject
        )
    
    def get_chapter_summary(
        self,
        class_name: str,
        subject: str,
        chapter: str,
        top_k: int = 5
    ) -> List[SearchResult]:
        """
        Get representative chunks from a specific chapter.
        
        Args:
            class_name: Class identifier (e.g., "class10")
            subject: Subject name (e.g., "science")
            chapter: Chapter identifier (e.g., "ch1")
            top_k: Number of chunks to retrieve
            
        Returns:
            List of SearchResult objects
        """
        # Use a general query to get diverse chunks
        query = f"main topics and concepts in {subject} chapter"
        
        return self.search(
            query=query,
            top_k=top_k,
            class_filter=class_name,
            subject_filter=subject,
            chapter_filter=chapter
        )
    
    def _build_filters(
        self,
        custom_filters: Dict = None,
        class_filter: str = None,
        subject_filter: str = None,
        chapter_filter: str = None
    ) -> Optional[Dict]:
        """
        Build ChromaDB metadata filters.
        
        Args:
            custom_filters: Custom filter dictionary
            class_filter: Class filter value
            subject_filter: Subject filter value
            chapter_filter: Chapter filter value
            
        Returns:
            Filter dictionary for ChromaDB or None
        """
        filters = custom_filters.copy() if custom_filters else {}
        
        if class_filter:
            filters['class_name'] = class_filter
        
        if subject_filter:
            filters['subject'] = subject_filter
        
        if chapter_filter:
            filters['chapter'] = chapter_filter
        
        # ChromaDB requires specific filter format
        # For simple equality filters, we can use direct key-value pairs
        if filters:
            # Convert to ChromaDB where clause format
            where_clause = {}
            for key, value in filters.items():
                where_clause[key] = value
            return where_clause
        
        return None
    
    def _rerank_results(self, query: str, results: List[Dict]) -> List[Dict]:
        """
        Rerank results using cross-encoder model.
        
        This is a placeholder for future cross-encoder integration.
        
        Args:
            query: Original query
            results: Initial search results
            
        Returns:
            Reranked results
        """
        # TODO: Implement cross-encoder reranking
        # Example with sentence-transformers cross-encoder:
        # from sentence_transformers import CrossEncoder
        # model = CrossEncoder(self.rerank_model)
        # pairs = [(query, result['text']) for result in results]
        # scores = model.predict(pairs)
        # sorted_results = [r for _, r in sorted(zip(scores, results), reverse=True)]
        # return sorted_results
        
        logger.info("Reranking not yet implemented, returning original results")
        return results
    
    def format_results_for_display(
        self,
        results: List[SearchResult],
        include_metadata: bool = True,
        max_text_length: int = 200
    ) -> List[Dict]:
        """
        Format search results for user display.
        
        Args:
            results: List of SearchResult objects
            include_metadata: Whether to include full metadata
            max_text_length: Maximum length of text to display
            
        Returns:
            List of formatted result dictionaries
        """
        formatted = []
        
        for result in results:
            text_display = result.text
            if len(text_display) > max_text_length:
                text_display = text_display[:max_text_length] + "..."
            
            formatted_result = {
                "rank": result.rank,
                "text": text_display,
                "similarity_score": round(result.similarity_score, 4),
                "context": result.get_display_context()
            }
            
            if include_metadata:
                formatted_result["metadata"] = result.metadata
            
            formatted.append(formatted_result)
        
        return formatted
    
    def get_statistics(self) -> Dict:
        """
        Get statistics about the retrieval system.
        
        Returns:
            Dictionary with system statistics
        """
        return {
            "total_chunks": self.vector_store.get_count(),
            "embedding_model": self.embedder.model_name,
            "embedding_dimension": self.embedder.embedding_dimension,
            "collection_name": self.vector_store.collection_name,
            "reranking_enabled": self.rerank_enabled,
            "default_top_k": self.config.get('top_k', 5)
        }


if __name__ == "__main__":
    # Example usage
    retriever = SemanticRetriever()
    
    # Get system statistics
    stats = retriever.get_statistics()
    print("Retrieval System Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Example search (will only work if data is already indexed)
    if stats['total_chunks'] > 0:
        print("\nPerforming sample search...")
        results = retriever.search(
            query="What is photosynthesis?",
            top_k=3,
            subject_filter="science"
        )
        
        print(f"\nFound {len(results)} results:")
        for result in results:
            print(f"\n{result.rank}. [{result.similarity_score:.4f}] {result.get_display_context()}")
            print(f"   {result.text[:100]}...")
    else:
        print("\nNo data indexed yet. Run the ingestion pipeline first.")
