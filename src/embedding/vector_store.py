"""
Vector Store Module

Manages vector storage and retrieval using ChromaDB.
Provides interface for indexing and querying text embeddings.
"""

from typing import List, Dict, Optional, Tuple
import logging
from pathlib import Path
import json
import numpy as np

import chromadb
from chromadb.config import Settings

from config.config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VectorStore:
    """
    Vector database interface using ChromaDB.
    
    Features:
    - Persistent storage of embeddings and metadata
    - Efficient similarity search
    - Metadata filtering
    - Batch operations
    """
    
    def __init__(
        self,
        collection_name: str = None,
        persist_directory: str = None,
        distance_metric: str = None
    ):
        """
        Initialize VectorStore.
        
        Args:
            collection_name: Name of the ChromaDB collection
            persist_directory: Directory for persistent storage
            distance_metric: Distance metric ('cosine', 'l2', 'ip')
        """
        self.collection_name = collection_name or Config.COLLECTION_NAME
        self.persist_directory = Path(persist_directory or Config.CHROMA_DB_DIR)
        self.distance_metric = distance_metric or Config.DISTANCE_METRIC
        
        # Map distance metric names
        metric_map = {
            "cosine": "cosine",
            "l2": "l2",
            "ip": "ip",  # Inner product
            "euclidean": "l2"
        }
        self.chroma_metric = metric_map.get(self.distance_metric, "cosine")
        
        # Ensure directory exists
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        
        # Initialize ChromaDB client
        logger.info(f"Initializing ChromaDB at {self.persist_directory}")
        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Get or create collection
        self._initialize_collection()
    
    def _initialize_collection(self):
        """Initialize or get existing collection."""
        try:
            # Try to get existing collection
            self.collection = self.client.get_collection(
                name=self.collection_name
            )
            count = self.collection.count()
            logger.info(f"Loaded existing collection '{self.collection_name}' with {count} items")
            
        except Exception:
            # Create new collection
            logger.info(f"Creating new collection '{self.collection_name}'")
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": self.chroma_metric}
            )
    
    def add_chunks(
        self,
        chunks: List,
        embeddings: np.ndarray = None,
        batch_size: int = 100
    ) -> int:
        """
        Add text chunks to the vector store.
        
        Args:
            chunks: List of TextChunk objects with embeddings
            embeddings: Optional pre-computed embeddings array
            batch_size: Number of chunks to add at once
            
        Returns:
            Number of chunks successfully added
        """
        if not chunks:
            logger.warning("No chunks to add")
            return 0
        
        total_added = 0
        
        # Process in batches
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            
            try:
                # Extract data from chunks
                ids = [chunk.chunk_id for chunk in batch]
                texts = [chunk.text for chunk in batch]
                metadatas = [self._prepare_metadata(chunk.metadata) for chunk in batch]
                
                # Get embeddings
                if embeddings is not None:
                    batch_embeddings = embeddings[i:i + batch_size].tolist()
                else:
                    batch_embeddings = [chunk.embedding.tolist() for chunk in batch]
                
                # Add to collection
                self.collection.add(
                    ids=ids,
                    embeddings=batch_embeddings,
                    documents=texts,
                    metadatas=metadatas
                )
                
                total_added += len(batch)
                logger.info(f"Added batch {i//batch_size + 1}: {len(batch)} chunks")
                
            except Exception as e:
                logger.error(f"Error adding batch {i//batch_size + 1}: {e}")
                continue
        
        logger.info(f"Successfully added {total_added} chunks to vector store")
        return total_added
    
    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = None,
        filter_metadata: Dict = None,
        score_threshold: float = None
    ) -> List[Dict]:
        """
        Search for similar chunks using query embedding.
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            filter_metadata: Optional metadata filters
            score_threshold: Minimum similarity score
            
        Returns:
            List of result dictionaries with chunk data and scores
        """
        top_k = top_k or Config.RETRIEVAL['top_k']
        score_threshold = score_threshold or Config.RETRIEVAL.get('score_threshold', 0.0)
        
        try:
            # Convert embedding to list
            query_embedding_list = query_embedding.tolist() if isinstance(query_embedding, np.ndarray) else query_embedding
            
            # Query collection
            results = self.collection.query(
                query_embeddings=[query_embedding_list],
                n_results=top_k,
                where=filter_metadata,
                include=["documents", "metadatas", "distances"]
            )
            
            # Format results
            formatted_results = []
            
            if results['ids'] and results['ids'][0]:
                for idx, chunk_id in enumerate(results['ids'][0]):
                    # Calculate similarity score from distance
                    distance = results['distances'][0][idx]
                    
                    # Convert distance to similarity based on metric
                    if self.chroma_metric == "cosine":
                        # Cosine distance is 1 - cosine similarity
                        similarity = 1 - distance
                    elif self.chroma_metric == "l2":
                        # Convert L2 distance to similarity (inverse relationship)
                        similarity = 1 / (1 + distance)
                    elif self.chroma_metric == "ip":
                        # Inner product (higher is more similar)
                        similarity = distance
                    else:
                        similarity = -distance
                    
                    # Apply threshold filter
                    if similarity < score_threshold:
                        continue
                    
                    result = {
                        "chunk_id": chunk_id,
                        "text": results['documents'][0][idx],
                        "metadata": results['metadatas'][0][idx],
                        "similarity_score": float(similarity),
                        "distance": float(distance)
                    }
                    
                    formatted_results.append(result)
            
            logger.info(f"Found {len(formatted_results)} results above threshold {score_threshold}")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error during search: {e}")
            return []
    
    def search_by_text(
        self,
        query_text: str,
        embedder,
        top_k: int = None,
        filter_metadata: Dict = None,
        score_threshold: float = None
    ) -> List[Dict]:
        """
        Search using text query (convenience method).
        
        Args:
            query_text: Text query
            embedder: TextEmbedder instance to generate query embedding
            top_k: Number of results to return
            filter_metadata: Optional metadata filters
            score_threshold: Minimum similarity score
            
        Returns:
            List of result dictionaries
        """
        # Generate embedding for query
        query_embedding = embedder.embed_text(query_text)
        
        # Perform search
        return self.search(
            query_embedding=query_embedding,
            top_k=top_k,
            filter_metadata=filter_metadata,
            score_threshold=score_threshold
        )
    
    def get_by_id(self, chunk_id: str) -> Optional[Dict]:
        """
        Retrieve a specific chunk by ID.
        
        Args:
            chunk_id: Chunk identifier
            
        Returns:
            Chunk dictionary or None if not found
        """
        try:
            result = self.collection.get(
                ids=[chunk_id],
                include=["documents", "metadatas", "embeddings"]
            )
            
            if result['ids']:
                return {
                    "chunk_id": result['ids'][0],
                    "text": result['documents'][0],
                    "metadata": result['metadatas'][0],
                    "embedding": result['embeddings'][0] if result['embeddings'] else None
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error retrieving chunk {chunk_id}: {e}")
            return None
    
    def delete_chunks(self, chunk_ids: List[str]) -> int:
        """
        Delete chunks by ID.
        
        Args:
            chunk_ids: List of chunk IDs to delete
            
        Returns:
            Number of chunks deleted
        """
        try:
            self.collection.delete(ids=chunk_ids)
            logger.info(f"Deleted {len(chunk_ids)} chunks")
            return len(chunk_ids)
            
        except Exception as e:
            logger.error(f"Error deleting chunks: {e}")
            return 0
    
    def clear_collection(self):
        """Delete all items in the collection."""
        try:
            self.client.delete_collection(name=self.collection_name)
            logger.info(f"Deleted collection '{self.collection_name}'")
            
            # Recreate empty collection
            self._initialize_collection()
            
        except Exception as e:
            logger.error(f"Error clearing collection: {e}")
    
    def get_count(self) -> int:
        """
        Get the number of chunks in the store.
        
        Returns:
            Number of stored chunks
        """
        return self.collection.count()
    
    def _prepare_metadata(self, metadata: Dict) -> Dict:
        """
        Prepare metadata for ChromaDB storage.
        
        ChromaDB requires metadata values to be strings, ints, floats, or bools.
        
        Args:
            metadata: Original metadata dictionary
            
        Returns:
            Prepared metadata dictionary
        """
        prepared = {}
        
        for key, value in metadata.items():
            if value is None:
                prepared[key] = ""
            elif isinstance(value, (str, int, float, bool)):
                prepared[key] = value
            else:
                # Convert other types to string
                prepared[key] = str(value)
        
        return prepared
    
    def get_stats(self) -> Dict:
        """
        Get statistics about the vector store.
        
        Returns:
            Dictionary with statistics
        """
        return {
            "collection_name": self.collection_name,
            "total_chunks": self.get_count(),
            "distance_metric": self.distance_metric,
            "persist_directory": str(self.persist_directory)
        }


if __name__ == "__main__":
    # Example usage
    from src.processing.chunker import TextChunk
    import numpy as np
    
    # Initialize vector store
    store = VectorStore()
    
    print(f"Vector store initialized")
    print(f"Current count: {store.get_count()}")
    
    # Create sample chunks with embeddings
    sample_chunks = [
        TextChunk(
            text="Photosynthesis is the process by which plants make food.",
            chunk_id="class10_science_ch1_p001",
            metadata={
                "class_name": "class10",
                "subject": "science",
                "chapter": "ch1",
                "heading_h1": "Plant Biology"
            },
            char_count=56,
            word_count=10
        )
    ]
    
    # Add fake embedding
    sample_chunks[0].embedding = np.random.rand(768)
    
    # Add to store
    # added_count = store.add_chunks(sample_chunks)
    # print(f"Added {added_count} chunks")
    
    # Get stats
    stats = store.get_stats()
    print("\nVector Store Stats:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
