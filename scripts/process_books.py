"""
NCERT Books Preprocessing Pipeline

This script performs the complete offline preprocessing workflow:
  1. Reads NCERT PDFs from data/books/
  2. Converts PDF → clean text
  3. Splits into paragraph chunks
  4. Generates embeddings using all-mpnet-base-v2
  5. Stores embeddings in ChromaDB

Usage:
  python scripts/process_books.py [--clear-chroma]

This is run ONCE as an offline preprocessing step.
Query pipeline does NOT process PDFs in real-time.
"""

import sys
import json
import logging
import re
from pathlib import Path
from typing import List, Dict, Any
import argparse

# Add main codebase to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.embedding.embedder import TextEmbedder
from src.pipeline.chroma_db import ChromaIndexer

# ============================================================================
# Configuration
# ============================================================================

BOOKS_DIR = Path("data/books")
PYQS_DIR = Path("data/pyqs")
OUTPUT_DIR = Path("output/embeddings")
CHROMA_DIR = Path("chroma_db")
COLLECTION_NAME = "ncert_chemistry"
EMBEDDING_DIMENSION = 768
BATCH_SIZE = 32

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/process_books.log")
    ]
)
logger = logging.getLogger(__name__)

# ============================================================================
# Helper Functions
# ============================================================================

def is_pdf(file_path: Path) -> bool:
    """Check if file is a PDF."""
    return file_path.suffix.lower() == '.pdf'


def is_text(file_path: Path) -> bool:
    """Check if file is a text file."""
    return file_path.suffix.lower() in ['.txt', '.md']


def extract_text_from_pdf(pdf_path: Path) -> str:
    """
    Extract text from PDF file.
    
    Args:
        pdf_path: Path to PDF file
        
    Returns:
        Extracted text
        
    Note:
        Requires PyPDF2 or pdfplumber. Install with:
        pip install PyPDF2
    """
    try:
        import fitz  # PyMuPDF

        text = []
        doc = fitz.open(pdf_path)
        for page in doc:
            text.append(page.get_text("text"))
        doc.close()

        return '\n\n'.join(text)

    except ImportError:
        # Fallback for environments where PyMuPDF is unavailable.
        try:
            import PyPDF2

            text = []
            with open(pdf_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    text.append(page.extract_text())

            return '\n\n'.join(text)
        except ImportError:
            logger.error("Neither PyMuPDF nor PyPDF2 is installed. Install with: pip install PyMuPDF")
            raise
        except Exception as e:
            logger.error(f"Error extracting text from {pdf_path}: {e}")
            raise
    except Exception as e:
        logger.error(f"Error extracting text from {pdf_path}: {e}")
        raise


def extract_text_from_file(file_path: Path) -> str:
    """Extract text from file (PDF or text)."""
    if is_pdf(file_path):
        return extract_text_from_pdf(file_path)
    elif is_text(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    else:
        raise ValueError(f"Unsupported file type: {file_path.suffix}")


def split_into_paragraphs(text: str, min_length: int = 50) -> List[str]:
    """
    Split text into paragraphs.
    
    Args:
        text: Raw text
        min_length: Minimum paragraph length
        
    Returns:
        List of paragraphs
    """
    # Split by double newlines (paragraph breaks)
    paragraphs = text.split('\n\n')
    
    # Filter empty and short paragraphs
    paragraphs = [p.strip() for p in paragraphs if len(p.strip()) >= min_length]
    
    return paragraphs


def split_into_questions(text: str, min_length: int = 12) -> List[str]:
    """Split PYQ content into question blocks, preserving MCQ options."""
    cleaned = text.replace('\r', '\n')
    lines = [ln.strip() for ln in cleaned.splitlines() if ln.strip()]

    question_start = re.compile(r'^(?:Q(?:uestion)?\s*)?\d+\s*[:.)-]\s*', flags=re.IGNORECASE)
    option_start = re.compile(r'^[A-Da-d]\s*[:.)-]\s+')
    noise_line = re.compile(
        r'^(?:section\s*[-A-Za-z0-9]*|questions?\s*no\.?|marks?\s*each|reprint\s*\d{4}|page\s*\d+)$',
        flags=re.IGNORECASE,
    )

    questions: List[str] = []
    current: List[str] = []

    def flush_current() -> None:
        nonlocal current
        if not current:
            return
        merged = ' '.join(current)
        merged = ' '.join(merged.split())
        merged = re.sub(r'\s+', ' ', merged).strip()
        if len(merged) >= min_length:
            questions.append(merged)
        current = []

    for line in lines:
        if noise_line.match(line):
            continue

        # Handle cases like "CHEMISTRY ... 1. The role ..." by trimming to first question marker.
        inline_match = re.search(r'(?:Q(?:uestion)?\s*)?\d+\s*[:.)-]\s*', line, flags=re.IGNORECASE)
        if inline_match and not question_start.match(line):
            line = line[inline_match.start():]

        if question_start.match(line):
            flush_current()
            current.append(question_start.sub('', line).strip())
            continue

        if option_start.match(line):
            if current:
                current.append(line)
            continue

        if current:
            current.append(line)

    flush_current()

    # Fallback for very noisy OCR where numbering is lost.
    if not questions:
        parts = re.split(r'\n(?=(?:Q(?:uestion)?\s*\d+[:.)-]?|\d+[).]))', cleaned, flags=re.IGNORECASE)
        for part in parts:
            normalized = ' '.join(part.split())
            if len(normalized) >= min_length:
                questions.append(normalized)

    return questions


def create_chunks_from_books(books_dir: Path) -> List[Dict[str, Any]]:
    """
    Create chunks from all books in directory.
    
    Args:
        books_dir: Directory containing PDF/text books
        
    Returns:
        List of chunk dictionaries with metadata
    """
    chunks = []
    chunk_id_counter = 0
    
    if not books_dir.exists():
        logger.warning(f"Books directory not found: {books_dir}")
        return chunks
    
    # Get all PDF and text files
    book_files = list(books_dir.glob("**/*.pdf")) + list(books_dir.glob("**/*.txt")) + list(books_dir.glob("**/*.md"))
    
    if not book_files:
        logger.warning(f"No book files found in {books_dir}")
        return chunks
    
    logger.info(f"Found {len(book_files)} book files")
    
    for book_file in sorted(book_files):
        try:
            logger.info(f"Processing: {book_file.name}")
            
            # Extract text
            text = extract_text_from_file(book_file)
            logger.info(f"Extracted {len(text)} characters from {book_file.name}")
            
            # Split into paragraphs
            paragraphs = split_into_paragraphs(text)
            logger.info(f"Created {len(paragraphs)} paragraphs from {book_file.name}")
            
            # Create chunks
            for para_idx, paragraph in enumerate(paragraphs):
                chunk_id_counter += 1
                chunk = {
                    "chunk_id": f"ncert_book_{chunk_id_counter:05d}",
                    "text": paragraph,
                    "metadata": {
                        "source": "ncert",
                        "file_name": book_file.name,
                        "paragraph_number": para_idx + 1,
                        "source_type": "book"
                    }
                }
                chunks.append(chunk)
        
        except Exception as e:
            logger.error(f"Error processing {book_file}: {e}")
            continue
    
    logger.info(f"Created total {len(chunks)} chunks from books")
    return chunks


def add_pyq_questions(pyqs_dir: Path, chunks: List[Dict]) -> List[Dict]:
    """
    Add PYQ questions as searchable chunks (optional).
    
    Args:
        pyqs_dir: Directory containing PYQ files
        chunks: Existing chunks list
        
    Returns:
        Updated chunks list
    """
    if not pyqs_dir.exists():
        logger.warning(f"PYQs directory not found: {pyqs_dir}")
        return chunks
    
    pyq_files = (
        list(pyqs_dir.glob("**/*.json"))
        + list(pyqs_dir.glob("**/*.txt"))
        + list(pyqs_dir.glob("**/*.md"))
        + list(pyqs_dir.glob("**/*.pdf"))
    )
    
    if not pyq_files:
        logger.info("No PYQ files found")
        return chunks
    
    logger.info(f"Found {len(pyq_files)} PYQ files")
    
    for pyq_file in sorted(pyq_files):
        try:
            if pyq_file.suffix == '.json':
                with open(pyq_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        pyqs = data
                    else:
                        pyqs = [data]
            else:
                text = extract_text_from_file(pyq_file)
                pyqs = split_into_questions(text)
            
            for idx, pyq in enumerate(pyqs):
                if isinstance(pyq, dict):
                    text = pyq.get('question', pyq.get('text', ''))
                    metadata = pyq.get('metadata', {})
                else:
                    text = str(pyq)
                    metadata = {}
                
                if text:
                    chunk_id = f"pyq_{len(chunks):05d}"
                    chunk = {
                        "chunk_id": chunk_id,
                        "text": text,
                        "metadata": {
                            "source": "pyq",
                            "file_name": pyq_file.name,
                            "question_number": idx + 1,
                            "source_type": "question",
                            **metadata
                        }
                    }
                    chunks.append(chunk)
            
            logger.info(f"Added {len(pyqs)} PYQs from {pyq_file.name}")
        
        except Exception as e:
            logger.error(f"Error processing {pyq_file}: {e}")
            continue
    
    return chunks


def generate_embeddings(chunks: List[Dict], batch_size: int = 32, device: str = "cpu") -> None:
    """
    Generate embeddings for chunks and store in ChromaDB.
    
    Args:
        chunks: List of chunks to embed
        batch_size: Batch size for embedding
        device: 'cpu' or 'cuda'
    """
    if not chunks:
        logger.error("No chunks to embed")
        return
    
    logger.info(f"Generating embeddings for {len(chunks)} chunks...")
    
    # Initialize embedder
    embedder = TextEmbedder(device=device)
    logger.info(f"Using device: {device}")
    
    # Initialize ChromaDB indexer
    indexer = ChromaIndexer(persist_dir=CHROMA_DIR, collection_name=COLLECTION_NAME)
    logger.info(f"Initialized ChromaDB at {CHROMA_DIR}")
    
    # Generate embeddings in batches
    embeddings_list = []
    texts = [chunk["text"] for chunk in chunks]
    
    logger.info(f"Batch size: {batch_size}")
    embeddings = embedder.embed_batch(texts, batch_size=batch_size, show_progress=True)
    
    # Add to ChromaDB
    total_indexed = 0
    for idx, chunk in enumerate(chunks):
        chunk_copy = chunk.copy()
        chunk_copy["embedding"] = embeddings[idx].tolist()
        embeddings_list.append(chunk_copy)
        total_indexed += 1
    
    # Batch add to ChromaDB
    batch_size_chroma = 100
    for i in range(0, len(embeddings_list), batch_size_chroma):
        batch = embeddings_list[i:i + batch_size_chroma]
        
        ids = [c["chunk_id"] for c in batch]
        docs = [c["text"] for c in batch]
        embs = [c["embedding"] for c in batch]
        metas = [c["metadata"] for c in batch]
        
        indexer.collection.upsert(ids=ids, documents=docs, embeddings=embs, metadatas=metas)
        logger.info(f"Indexed batch {i//batch_size_chroma + 1}: {len(batch)} chunks")
    
    # Save embeddings to JSON for backup
    output_file = OUTPUT_DIR / "ncert_embeddings.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            "record_count": len(embeddings_list),
            "embedding_dimension": EMBEDDING_DIMENSION,
            "records": embeddings_list
        }, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Saved embeddings to {output_file}")
    logger.info(f"Total indexed in ChromaDB: {total_indexed} chunks")
    logger.info(f"ChromaDB collection now has: {indexer.collection.count()} items")


# ============================================================================
# Main Pipeline
# ============================================================================

def main():
    """Run the complete preprocessing pipeline."""
    parser = argparse.ArgumentParser(
        description="Process NCERT books and PYQs for semantic search"
    )
    parser.add_argument(
        "--clear-chroma",
        action="store_true",
        help="Clear existing ChromaDB collection before processing"
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device for embedding generation"
    )
    args = parser.parse_args()
    
    logger.info("=" * 70)
    logger.info("NCERT Books Preprocessing Pipeline")
    logger.info("=" * 70)
    
    try:
        # Create output directory
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        Path("logs").mkdir(parents=True, exist_ok=True)
        
        # Clear ChromaDB if requested
        if args.clear_chroma:
            logger.info("Clearing existing ChromaDB collection...")
            indexer = ChromaIndexer(persist_dir=CHROMA_DIR, collection_name=COLLECTION_NAME)
            indexer.reset_collection()
            logger.info("ChromaDB collection cleared")
        
        # Step 1: Create chunks from books
        logger.info("\n[Step 1] Creating chunks from books...")
        chunks = create_chunks_from_books(BOOKS_DIR)
        
        # Step 2: Add PYQ questions
        logger.info("\n[Step 2] Adding PYQ questions...")
        chunks = add_pyq_questions(PYQS_DIR, chunks)
        
        if not chunks:
            logger.error("No chunks created. Please add books to data/books/ or PYQs to data/pyqs/")
            return 1
        
        logger.info(f"\n[Summary] Total chunks created: {len(chunks)}")
        
        # Step 3: Generate embeddings and store in ChromaDB
        logger.info("\n[Step 3] Generating embeddings and indexing in ChromaDB...")
        generate_embeddings(chunks, batch_size=BATCH_SIZE, device=args.device)
        
        logger.info("\n" + "=" * 70)
        logger.info("Preprocessing pipeline completed successfully!")
        logger.info("=" * 70)
        logger.info(f"\nNext step: Start backend with:")
        logger.info(f"  cd backend")
        logger.info(f"  python -m uvicorn main:app --reload --port 8000")
        
        return 0
    
    except Exception as e:
        logger.error(f"\nPipeline failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
