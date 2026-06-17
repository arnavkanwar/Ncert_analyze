"""
Text Embedder Module

Generates dense vector embeddings using sentence-transformers.
Supports batch processing for efficient embedding generation.
"""

from typing import List, Union
import logging
import numpy as np
from sentence_transformers import SentenceTransformer
import torch

from config.config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TextEmbedder:
    """
    Text embedding generator using sentence-transformers.
    
    Uses all-mpnet-base-v2 model by default for high-quality embeddings.
    Supports batch processing and GPU acceleration.
    """
    
    def __init__(self, model_name: str = None, device: str = None):
        """
        Initialize TextEmbedder with model.
        
        Args:
            model_name: Name of sentence-transformers model. Defaults to Config.EMBEDDING_MODEL
            device: Device to run model on ('cpu' or 'cuda'). Defaults to Config.DEVICE
        """
        self.model_name = model_name or Config.EMBEDDING_MODEL
        self.device = device or Config.DEVICE
        
        logger.info(f"Loading embedding model: {self.model_name}")
        logger.info(f"Using device: {self.device}")
        
        try:
            self.model = SentenceTransformer(self.model_name, device=self.device)
            self.embedding_dimension = self.model.get_sentence_embedding_dimension()
            logger.info(f"Model loaded successfully. Embedding dimension: {self.embedding_dimension}")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def embed_text(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text to embed
            
        Returns:
            Numpy array of embedding vector
        """
        if not text or not text.strip():
            logger.warning("Empty text provided for embedding")
            return np.zeros(self.embedding_dimension)
        
        try:
            embedding = self.model.encode(
                text,
                convert_to_numpy=True,
                show_progress_bar=False,
                normalize_embeddings=True  # Normalize for cosine similarity
            )
            return embedding
            
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return np.zeros(self.embedding_dimension)
    
    def embed_batch(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Generate embeddings for multiple texts efficiently.
        
        Args:
            texts: List of texts to embed
            batch_size: Number of texts to process at once
            show_progress: Whether to show progress bar
            
        Returns:
            Numpy array of shape (n_texts, embedding_dimension)
        """
        if not texts:
            logger.warning("Empty text list provided for batch embedding")
            return np.array([])
        
        # Filter out empty texts but keep track of indices
        valid_texts = []
        valid_indices = []
        
        for i, text in enumerate(texts):
            if text and text.strip():
                valid_texts.append(text)
                valid_indices.append(i)
        
        if not valid_texts:
            logger.warning("No valid texts in batch")
            return np.zeros((len(texts), self.embedding_dimension))
        
        logger.info(f"Generating embeddings for {len(valid_texts)} texts (batch size: {batch_size})")
        
        try:
            embeddings = self.model.encode(
                valid_texts,
                batch_size=batch_size,
                convert_to_numpy=True,
                show_progress_bar=show_progress,
                normalize_embeddings=True
            )
            
            # Create full embedding array with zeros for invalid texts
            full_embeddings = np.zeros((len(texts), self.embedding_dimension))
            full_embeddings[valid_indices] = embeddings
            
            logger.info(f"Successfully generated {len(embeddings)} embeddings")
            return full_embeddings
            
        except Exception as e:
            logger.error(f"Error generating batch embeddings: {e}")
            return np.zeros((len(texts), self.embedding_dimension))
    
    def embed_chunks(self, chunks: List, batch_size: int = 32) -> List:
        """
        Generate embeddings for TextChunk objects.
        
        Modifies chunks in place by adding 'embedding' field.
        
        Args:
            chunks: List of TextChunk objects
            batch_size: Batch size for embedding generation
            
        Returns:
            List of chunks with embeddings added
        """
        if not chunks:
            return []
        
        # Extract texts
        texts = [chunk.text for chunk in chunks]
        
        # Generate embeddings
        embeddings = self.embed_batch(texts, batch_size=batch_size, show_progress=True)
        
        # Add embeddings to chunks
        for chunk, embedding in zip(chunks, embeddings):
            chunk.embedding = embedding
        
        logger.info(f"Added embeddings to {len(chunks)} chunks")
        return chunks
    
    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of embeddings produced by this model.
        
        Returns:
            Embedding dimension as integer
        """
        return self.embedding_dimension
    
    def compute_similarity(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray,
        metric: str = "cosine"
    ) -> float:
        """
        Compute similarity between two embeddings.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            metric: Similarity metric ('cosine', 'dot', 'euclidean')
            
        Returns:
            Similarity score
        """
        if metric == "cosine":
            # Cosine similarity (assumes normalized vectors)
            return float(np.dot(embedding1, embedding2))
        
        elif metric == "dot":
            # Dot product
            return float(np.dot(embedding1, embedding2))
        
        elif metric == "euclidean":
            # Negative Euclidean distance (negative so higher is more similar)
            return float(-np.linalg.norm(embedding1 - embedding2))
        
        else:
            raise ValueError(f"Unknown metric: {metric}")
    
    def get_model_info(self) -> dict:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary with model information
        """
        return {
            "model_name": self.model_name,
            "embedding_dimension": self.embedding_dimension,
            "device": self.device,
            "max_seq_length": self.model.max_seq_length
        }


if __name__ == "__main__":
    # Example usage
    embedder = TextEmbedder()
    
    # Single text embedding
    text = "Photosynthesis is the process by which plants make food."
    embedding = embedder.embed_text(text)
    print(f"Single embedding shape: {embedding.shape}")
    print(f"First 5 values: {embedding[:5]}")
    
    # Batch embedding
    texts = [
        "Photosynthesis occurs in chloroplasts.",
        "Plants use sunlight to convert CO2 and water into glucose.",
        "The chemical equation for photosynthesis involves oxygen production."
    ]
    
    embeddings = embedder.embed_batch(texts, batch_size=2, show_progress=False)
    print(f"\nBatch embeddings shape: {embeddings.shape}")
    
    # Compute similarity
    similarity = embedder.compute_similarity(embeddings[0], embeddings[1], metric="cosine")
    print(f"\nSimilarity between first two texts: {similarity:.4f}")
    
    # Model info
    print("\nModel info:")
    for key, value in embedder.get_model_info().items():
        print(f"  {key}: {value}")
