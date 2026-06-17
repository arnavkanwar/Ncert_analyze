"""
NCERT Ingestion Pipeline

End-to-end pipeline for processing NCERT markdown files and
indexing them in the vector database.

Usage:
    python scripts/ingest_pipeline.py
    python scripts/ingest_pipeline.py --data-dir path/to/data
    python scripts/ingest_pipeline.py --clear-existing
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json
import logging
import shutil
import subprocess
from datetime import datetime
from typing import List, Dict, Optional
from tqdm import tqdm

from config.config import Config
from src.parser.markdown_parser import MarkdownParser
from src.processing.text_cleaner import TextCleaner
from src.processing.chunker import TextChunker
from src.processing.metadata_extractor import MetadataExtractor
from src.embedding.embedder import TextEmbedder
from src.embedding.vector_store import VectorStore

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/ingestion.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def _run_command(command: List[str], cwd: Optional[Path] = None) -> None:
    """Run a command and raise on non-zero exit status."""
    logger.info("Running command: %s", " ".join(command))
    result = subprocess.run(command, cwd=str(cwd) if cwd else None, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed ({result.returncode}): {' '.join(command)}")


def _patch_manifest_metadata(
    manifest_path: Path,
    source: str,
    staged_pdf_names: List[str],
    class_name: Optional[str] = None,
    subject: Optional[str] = None,
    year: Optional[str] = None,
    chapter: Optional[str] = None,
    book: Optional[str] = None,
) -> None:
    """Patch output/raw_text manifest entries so metadata flows into chunk/embedding stages."""
    if not manifest_path.exists():
        logger.warning("Manifest not found for metadata patching: %s", manifest_path)
        return

    rows = json.loads(manifest_path.read_text(encoding="utf-8"))
    staged_set = {name.lower() for name in staged_pdf_names}

    for row in rows:
        file_name = str(row.get("file_name", "")).lower()
        if file_name not in staged_set:
            continue

        row["source"] = source
        row["type"] = source
        if class_name:
            row["class_name"] = class_name
        if subject:
            row["subject"] = subject
        if year:
            row["year"] = str(year)
        if chapter:
            row["chapter"] = chapter
        if book:
            row["book"] = book

    manifest_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Patched manifest metadata for %d staged files", len(staged_pdf_names))


def run_ingestion(
    data_path: Path,
    source: str,
    class_name: Optional[str] = None,
    subject: Optional[str] = None,
    year: Optional[str] = None,
    chapter: Optional[str] = None,
    book: Optional[str] = None,
    clear_existing: bool = False,
) -> Dict[str, object]:
    """Run repeatable ingestion for uploaded PYQ/NCERT PDF files.

    This path is used by Prompt-04 style ingestion and the upload endpoint.
    """
    project_root = Path(__file__).resolve().parent.parent
    python_exe = sys.executable

    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Input path not found: {data_path}")

    if data_path.is_file():
        pdf_files = [data_path] if data_path.suffix.lower() == ".pdf" else []
    else:
        pdf_files = sorted(data_path.rglob("*.pdf"))

    if not pdf_files:
        raise ValueError("No PDF files found. Prompt 04 workflow expects PDF uploads for this path.")

    stage_root = project_root / "media" / ("pyqs" if source == "pyq" else "ncert_uploads")
    stage_root.mkdir(parents=True, exist_ok=True)

    staged_names: List[str] = []
    for pdf in pdf_files:
        target = stage_root / pdf.name
        if pdf.resolve() != target.resolve():
            shutil.copy2(pdf, target)
        staged_names.append(target.name)

    raw_text_dir = project_root / "output" / "raw_text_upload"
    chunks_dir = project_root / "output" / "chunks_upload"
    chunks_file = chunks_dir / "all_chunks.json"
    embeddings_file = project_root / "output" / "embeddings" / "upload_embeddings.json"

    if raw_text_dir.exists():
        shutil.rmtree(raw_text_dir)
    if chunks_dir.exists():
        shutil.rmtree(chunks_dir)

    _run_command(
        [python_exe, "scripts/pdf_to_text.py", "--media-dir", str(stage_root), "--raw-text-dir", str(raw_text_dir)],
        cwd=project_root,
    )

    _patch_manifest_metadata(
        manifest_path=raw_text_dir / "manifest.json",
        source=source,
        staged_pdf_names=staged_names,
        class_name=class_name,
        subject=subject,
        year=year,
        chapter=chapter,
        book=book,
    )

    _run_command(
        [python_exe, "scripts/text_to_chunks.py", "--raw-text-dir", str(raw_text_dir), "--chunks-dir", str(chunks_dir)],
        cwd=project_root,
    )
    _run_command(
        [python_exe, "scripts/generate_embeddings.py", "--chunks-json", str(chunks_file), "--output-json", str(embeddings_file), "--device", "cpu"],
        cwd=project_root,
    )

    chroma_cmd = [
        python_exe,
        "scripts/chroma_store.py",
        "--embeddings-json",
        str(embeddings_file),
        "--chroma-dir",
        str(project_root / "chroma_db"),
        "--collection",
        "ncert_chemistry",
    ]
    if clear_existing:
        chroma_cmd.append("--reset")
    _run_command(chroma_cmd, cwd=project_root)

    return {
        "status": "indexed",
        "source": source,
        "files": [str(p.name) for p in pdf_files],
        "count": len(pdf_files),
    }


class IngestionPipeline:
    """
    Complete ingestion pipeline for NCERT textbooks.
    
    Pipeline stages:
    1. Parse markdown files
    2. Clean and normalize text
    3. Chunk into paragraphs
    4. Extract metadata
    5. Generate embeddings
    6. Store in vector database
    7. Export JSON outputs
    """
    
    def __init__(self, data_dir: Path = None, clear_existing: bool = False):
        """
        Initialize ingestion pipeline.
        
        Args:
            data_dir: Root directory containing NCERT markdown files
            clear_existing: Whether to clear existing data in vector store
        """
        self.data_dir = Path(data_dir) if data_dir else Config.DATA_DIR
        self.output_dir = Config.OUTPUT_DIR
        self.clear_existing = clear_existing
        
        # Initialize components
        logger.info("Initializing pipeline components...")
        self.parser = MarkdownParser()
        self.cleaner = TextCleaner()
        self.chunker = TextChunker()
        self.metadata_extractor = MetadataExtractor()
        self.embedder = TextEmbedder()
        self.vector_store = VectorStore()
        
        # Statistics
        self.stats = {
            "files_processed": 0,
            "sections_parsed": 0,
            "chunks_created": 0,
            "chunks_indexed": 0,
            "start_time": None,
            "end_time": None
        }
        
        # Clear existing data if requested
        if self.clear_existing:
            logger.warning("Clearing existing vector store data...")
            self.vector_store.clear_collection()
    
    def run(self) -> Dict:
        """
        Run the complete ingestion pipeline.
        
        Returns:
            Dictionary with pipeline statistics
        """
        self.stats["start_time"] = datetime.now()
        
        logger.info("=" * 70)
        logger.info("Starting NCERT Ingestion Pipeline")
        logger.info("=" * 70)
        logger.info(f"Data directory: {self.data_dir}")
        logger.info(f"Output directory: {self.output_dir}")
        
        # Find all markdown files
        markdown_files = self.parser.find_markdown_files(self.data_dir)
        
        if not markdown_files:
            logger.error(f"No markdown files found in {self.data_dir}")
            return self.stats
        
        logger.info(f"Found {len(markdown_files)} markdown files to process")
        
        # Process each file
        all_chunks = []
        
        for file_path in tqdm(markdown_files, desc="Processing files"):
            chunks = self._process_file(file_path)
            all_chunks.extend(chunks)
            self.stats["files_processed"] += 1
        
        # Generate embeddings in batch
        if all_chunks:
            logger.info(f"Generating embeddings for {len(all_chunks)} chunks...")
            self.embedder.embed_chunks(all_chunks, batch_size=32)
            
            # Index in vector store
            logger.info("Indexing chunks in vector database...")
            indexed_count = self.vector_store.add_chunks(all_chunks, batch_size=100)
            self.stats["chunks_indexed"] = indexed_count
            
            # Export JSON outputs
            logger.info("Exporting JSON chunks...")
            self._export_json_chunks(all_chunks)
        
        # Finalize statistics
        self.stats["end_time"] = datetime.now()
        duration = (self.stats["end_time"] - self.stats["start_time"]).total_seconds()
        self.stats["duration_seconds"] = duration
        
        # Print summary
        self._print_summary()
        
        return self.stats
    
    def _process_file(self, file_path: Path) -> List:
        """
        Process a single markdown file.
        
        Args:
            file_path: Path to markdown file
            
        Returns:
            List of processed chunks
        """
        try:
            # Parse markdown
            sections = self.parser.parse_file(file_path)
            self.stats["sections_parsed"] += len(sections)
            
            if not sections:
                logger.warning(f"No sections found in {file_path}")
                return []
            
            # Extract path metadata
            path_metadata = self.metadata_extractor.extract_from_path(file_path)
            
            # Process each section
            chunks = []
            paragraph_counter = 1
            
            for section in sections:
                # Clean text
                cleaned_text = self.cleaner.clean(section.content)
                
                # Skip if not valid
                if not self.cleaner.is_valid_chunk(cleaned_text):
                    continue
                
                # Combine metadata
                section_metadata = {
                    **path_metadata,
                    "heading_h1": section.heading_h1 or "",
                    "heading_h2": section.heading_h2 or "",
                    "heading_h3": section.heading_h3 or ""
                }
                
                # Chunk text
                text_chunks = self.chunker.chunk_text(cleaned_text, section_metadata)
                
                # Update chunk IDs to be globally unique
                for chunk in text_chunks:
                    # Regenerate chunk ID with continuous paragraph numbering
                    chunk.chunk_id = self.chunker._generate_chunk_id(
                        section_metadata,
                        paragraph_counter
                    )
                    chunk.metadata["paragraph_number"] = paragraph_counter
                    paragraph_counter += 1
                
                chunks.extend(text_chunks)
            
            self.stats["chunks_created"] += len(chunks)
            logger.info(f"Processed {file_path.name}: {len(sections)} sections → {len(chunks)} chunks")
            
            return chunks
            
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")
            return []
    
    def _export_json_chunks(self, chunks: List) -> None:
        """
        Export chunks as JSON files.
        
        Args:
            chunks: List of TextChunk objects
        """
        try:
            # Ensure output directory exists
            self.output_dir.mkdir(parents=True, exist_ok=True)
            
            # Export all chunks to a single JSON file
            all_chunks_file = self.output_dir / f"all_chunks_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            chunks_data = []
            for chunk in chunks:
                chunk_dict = {
                    "chunk_id": chunk.chunk_id,
                    "text": chunk.text,
                    "char_count": chunk.char_count,
                    "word_count": chunk.word_count,
                    "metadata": chunk.metadata
                }
                chunks_data.append(chunk_dict)
            
            with open(all_chunks_file, 'w', encoding='utf-8') as f:
                json.dump(chunks_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Exported {len(chunks)} chunks to {all_chunks_file}")
            
            # Also export a summary file
            summary_file = self.output_dir / f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            summary = {
                "total_chunks": len(chunks),
                "schema_version": Config.SCHEMA_VERSION,
                "created_at": datetime.now().isoformat(),
                "statistics": self.stats
            }
            
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Exported summary to {summary_file}")
            
        except Exception as e:
            logger.error(f"Error exporting JSON chunks: {e}")
    
    def _print_summary(self) -> None:
        """Print pipeline execution summary."""
        logger.info("\n" + "=" * 70)
        logger.info("Pipeline Execution Summary")
        logger.info("=" * 70)
        logger.info(f"Files processed: {self.stats['files_processed']}")
        logger.info(f"Sections parsed: {self.stats['sections_parsed']}")
        logger.info(f"Chunks created: {self.stats['chunks_created']}")
        logger.info(f"Chunks indexed: {self.stats['chunks_indexed']}")
        logger.info(f"Duration: {self.stats['duration_seconds']:.2f} seconds")
        logger.info(f"Vector store total: {self.vector_store.get_count()} chunks")
        logger.info("=" * 70)


def main():
    """Main entry point for ingestion pipeline."""
    parser = argparse.ArgumentParser(
        description="NCERT Textbook Ingestion Pipeline"
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default=None,
        help='Root directory containing NCERT markdown files'
    )
    parser.add_argument(
        '--clear-existing',
        action='store_true',
        help='Clear existing data in vector store before ingestion'
    )
    parser.add_argument(
        '--source',
        type=str,
        choices=['pyq', 'ncert'],
        default=None,
        help='Source type for Prompt-04 upload ingestion workflow'
    )
    parser.add_argument('--class', dest='class_name', type=str, default=None, help='Class metadata, e.g. class12')
    parser.add_argument('--subject', type=str, default=None, help='Subject metadata, e.g. chemistry')
    parser.add_argument('--year', type=str, default=None, help='Year metadata for PYQ files, e.g. 2024')
    parser.add_argument('--chapter', type=str, default=None, help='Chapter metadata, e.g. ch5')
    parser.add_argument('--book', type=str, default=None, help='Book metadata, e.g. book1')
    
    args = parser.parse_args()
    
    try:
        if args.source:
            if not args.data_dir:
                raise ValueError("--data-dir is required when --source is provided")

            result = run_ingestion(
                data_path=Path(args.data_dir),
                source=args.source,
                class_name=args.class_name,
                subject=args.subject,
                year=args.year,
                chapter=args.chapter,
                book=args.book,
                clear_existing=args.clear_existing,
            )
            logger.info("\n✓ Upload ingestion completed successfully: %s", result)
            sys.exit(0)

        # Default behavior: legacy markdown ingestion pipeline.
        pipeline = IngestionPipeline(
            data_dir=args.data_dir,
            clear_existing=args.clear_existing
        )
        stats = pipeline.run()

        if stats['chunks_indexed'] > 0:
            logger.info("\n✓ Pipeline completed successfully!")
            sys.exit(0)
        logger.error("\n✗ Pipeline completed but no chunks were indexed")
        sys.exit(1)
            
    except KeyboardInterrupt:
        logger.warning("\n⚠ Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n✗ Pipeline failed with error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
