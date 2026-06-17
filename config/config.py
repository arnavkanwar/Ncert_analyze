"""
Configuration settings for NCERT Retrieval System.

This module contains all configuration parameters for the pipeline.
Modify these values to customize the system behavior.
"""

import os
from pathlib import Path
from typing import Dict, Any

class Config:
    """Central configuration for the NCERT retrieval pipeline."""
    
    # Project paths
    BASE_DIR = Path(__file__).parent.parent
    DATA_DIR = BASE_DIR / "data" / "NCERT"
    OUTPUT_DIR = BASE_DIR / "output" / "chunks"
    EMBEDDING_OUTPUT_DIR = BASE_DIR / "output" / "embeddings"
    CHROMA_DB_DIR = BASE_DIR / "chroma_db"
    
    # Input data structure
    # Expected: data/NCERT/Class_XX/Subject/Book_X/Chapter_XX/chapter.md
    DATA_STRUCTURE = {
        "root": "data/NCERT",
        "levels": ["class", "subject", "book", "chapter"],
        "file_pattern": "*.md"
    }
    
    # Embedding model configuration
    # OPTION A (current): Good general-purpose model
    EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
    EMBEDDING_DIMENSION = 768  # Dimension for all-mpnet-base-v2
    # OPTION B (recommended upgrade): Better retrieval-optimized model
    # EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
    # EMBEDDING_DIMENSION = 768
    DEVICE = "cpu"  # Change to "cuda" if GPU available

    # Phase 3: Embedding generation configuration
    PHASE3 = {
        "input_json": BASE_DIR / "data" / "sample_phase3_input.json",
        "output_json": EMBEDDING_OUTPUT_DIR / "phase3_embeddings.json",
        "batch_size": 32,
        "normalize_embeddings": True,
        "schema_version": "1.0",
        "required_input_fields": [
            "chunk_id",
            "text"
        ]
    }
    
    # ChromaDB configuration
    COLLECTION_NAME = "ncert_textbook_chunks"
    DISTANCE_METRIC = "cosine"  # Options: cosine, l2, ip
    
    # Chunking configuration
    CHUNKING = {
        "strategy": "heading_aware",  # IMPROVED: heading-context-aware chunking
        "min_chunk_size": 60,         # Minimum characters per chunk
        "max_chunk_size": 1500,       # Maximum characters per chunk
        "merge_threshold": 200,       # Merge paragraphs shorter than this
        "overlap_sentences": 1,       # Sentence overlap between split chunks
        "preserve_headings": True,    # Keep heading context
        "prepend_heading": True,      # Prepend heading to chunk text for embedding
        "split_on": ["\n\n", "\n", ". "]  # Split delimiters in order of preference
    }
    
    # Text cleaning configuration
    TEXT_CLEANING = {
        "remove_extra_whitespace": True,
        "normalize_unicode": True,
        "remove_special_chars": False,  # Keep for math/science content
        "lowercase": False,              # Preserve case for proper nouns
        "remove_urls": True,
        "remove_html": True,
        "min_words": 3                   # Minimum words to keep a chunk
    }
    
    # Metadata configuration
    METADATA_FIELDS = [
        "chunk_id",
        "source_file",
        "class_name",
        "subject",
        "book",
        "chapter",
        "heading_h1",
        "heading_h2",
        "heading_h3",
        "paragraph_number",
        "char_count",
        "word_count",
        "created_at"
    ]
    
    # Chunk ID format: class10_science_ch1_p003
    CHUNK_ID_FORMAT = "{class_name}_{subject}_{chapter}_p{paragraph:03d}"
    
    # Retrieval configuration
    RETRIEVAL = {
        "top_k": 10,                   # Initial retrieval pool
        "score_threshold": 0.15,       # Minimum hybrid score to return
        "rerank": True,                # ENABLED: cross-encoder reranking
        "rerank_model": "cross-encoder/ms-marco-MiniLM-L-12-v2",  # L-12 is much better than L-6
        "rerank_top_k": 2,             # Return top 2 after reranking
        "vector_candidates": 30,       # Candidates from vector search
        "bm25_candidates": 20,         # Candidates from BM25
        "hybrid_weights": {            # Fusion weights
            "rerank": 0.55,
            "vector": 0.20,
            "bm25": 0.15,
            "keyword": 0.10,
        }
    }
    
    # JSON output schema version
    SCHEMA_VERSION = "1.0"
    
    # Logging configuration
    LOGGING = {
        "level": "INFO",
        "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        "file": "logs/pipeline.log"
    }
    
    @classmethod
    def ensure_directories(cls) -> None:
        """Create necessary directories if they don't exist."""
        directories = [
            cls.OUTPUT_DIR,
            cls.EMBEDDING_OUTPUT_DIR,
            cls.CHROMA_DB_DIR,
            cls.BASE_DIR / "logs"
        ]
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def get_config_dict(cls) -> Dict[str, Any]:
        """Return configuration as a dictionary."""
        return {
            "embedding_model": cls.EMBEDDING_MODEL,
            "embedding_dimension": cls.EMBEDDING_DIMENSION,
            "collection_name": cls.COLLECTION_NAME,
            "chunking": cls.CHUNKING,
            "text_cleaning": cls.TEXT_CLEANING,
            "retrieval": cls.RETRIEVAL,
            "schema_version": cls.SCHEMA_VERSION
        }
    
    @classmethod
    def display_config(cls) -> None:
        """Display current configuration settings."""
        print("=" * 60)
        print("NCERT Retrieval System Configuration")
        print("=" * 60)
        print(f"Data Directory: {cls.DATA_DIR}")
        print(f"Output Directory: {cls.OUTPUT_DIR}")
        print(f"ChromaDB Directory: {cls.CHROMA_DB_DIR}")
        print(f"Embedding Model: {cls.EMBEDDING_MODEL}")
        print(f"Collection Name: {cls.COLLECTION_NAME}")
        print(f"Chunking Strategy: {cls.CHUNKING['strategy']}")
        print(f"Max Chunk Size: {cls.CHUNKING['max_chunk_size']}")
        print(f"Top-K Retrieval: {cls.RETRIEVAL['top_k']}")
        print("=" * 60)


# Initialize directories on import
Config.ensure_directories()
