"""
NCERT Text Processing Pipeline - Phase 1 & 2

Main pipeline script that processes text files and generates chunked JSON output.

Usage:
    python pipeline.py
    python pipeline.py --input-dir custom_input --output-dir custom_output
"""

import sys
from pathlib import Path

# Add modules directory to path
sys.path.insert(0, str(Path(__file__).parent / 'modules'))

import argparse
from datetime import datetime
from typing import List

from text_parser import TextParser
from text_cleaner import TextCleaner
from paragraph_chunker import ParagraphChunker, TextChunk
from json_exporter import JSONExporter


class NCERTProcessingPipeline:
    """
    Complete pipeline for processing NCERT textbook text files.
    
    Pipeline stages:
    1. Parse text files from folder structure
    2. Clean text (remove page numbers, normalize whitespace)
    3. Chunk into paragraphs
    4. Generate metadata
    5. Export to JSON
    """
    
    def __init__(self, input_dir: str = "input", output_dir: str = "output"):
        """
        Initialize pipeline.
        
        Args:
            input_dir: Directory containing input text files
            output_dir: Directory for output JSON files
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        
        # Initialize components
        print("Initializing pipeline components...")
        self.parser = TextParser(input_dir)
        self.cleaner = TextCleaner()
        self.chunker = ParagraphChunker()
        self.exporter = JSONExporter(output_dir)
        
        # Statistics
        self.stats = {
            "files_processed": 0,
            "total_chunks": 0,
            "total_words": 0,
            "start_time": None,
            "end_time": None
        }
    
    def process_single_file(self, file_path: Path) -> List[TextChunk]:
        """
        Process a single text file.
        
        Args:
            file_path: Path to text file
            
        Returns:
            List of TextChunk objects
        """
        print(f"\nProcessing: {file_path.name}")
        print("-" * 70)
        
        # Step 1: Parse file
        parsed_data = self.parser.parse_file_with_metadata(file_path)
        content = parsed_data["content"]
        metadata = parsed_data["metadata"]
        
        if not content:
            print(f"  ⚠ No content found in {file_path.name}")
            return []
        
        print(f"  ✓ Parsed file: {len(content)} characters")
        
        # Step 2: Clean text
        cleaned_content = self.cleaner.clean(content)
        print(f"  ✓ Cleaned text: {len(cleaned_content)} characters")
        
        # Step 3: Chunk into paragraphs
        chunks = self.chunker.chunk_text(cleaned_content, metadata)
        print(f"  ✓ Created {len(chunks)} chunks")
        
        # Update statistics
        self.stats["files_processed"] += 1
        self.stats["total_chunks"] += len(chunks)
        self.stats["total_words"] += sum(chunk.word_count for chunk in chunks)
        
        return chunks
    
    def process_all_files(self) -> List[TextChunk]:
        """
        Process all text files in the input directory.
        
        Returns:
            List of all TextChunk objects
        """
        # Find all files
        files = self.parser.find_all_files()
        
        if not files:
            print(f"\n⚠ No text files found in {self.input_dir}")
            return []
        
        print(f"\nFound {len(files)} file(s) to process")
        print("=" * 70)
        
        # Process each file
        all_chunks = []
        
        for file_path in files:
            chunks = self.process_single_file(file_path)
            all_chunks.extend(chunks)
        
        return all_chunks
    
    def run(self):
        """
        Run the complete pipeline.
        """
        self.stats["start_time"] = datetime.now()
        
        print("\n" + "=" * 70)
        print("NCERT Text Processing Pipeline - Phase 1 & 2")
        print("=" * 70)
        print(f"Input directory: {self.input_dir}")
        print(f"Output directory: {self.output_dir}")
        
        # Process all files
        all_chunks = self.process_all_files()
        
        if not all_chunks:
            print("\n⚠ No chunks were created. Please check your input files.")
            return
        
        # Export to JSON
        print("\n" + "=" * 70)
        print("Exporting to JSON...")
        print("-" * 70)
        
        output_file = self.exporter.export_chunks(all_chunks)
        
        # Export summary
        summary_file = self.exporter.export_summary(all_chunks)
        
        # Print final statistics
        self.stats["end_time"] = datetime.now()
        duration = (self.stats["end_time"] - self.stats["start_time"]).total_seconds()
        
        print("\n" + "=" * 70)
        print("Pipeline Execution Summary")
        print("=" * 70)
        print(f"Files processed: {self.stats['files_processed']}")
        print(f"Total chunks created: {self.stats['total_chunks']}")
        print(f"Total words: {self.stats['total_words']}")
        print(f"Execution time: {duration:.2f} seconds")
        print(f"Output file: {output_file}")
        print(f"Summary file: {summary_file}")
        print("=" * 70)
        print("\n✓ Pipeline completed successfully!")
        
        # Display sample chunks
        self._display_sample_chunks(all_chunks[:3])
    
    def _display_sample_chunks(self, chunks: List[TextChunk]):
        """Display sample chunks for verification."""
        if not chunks:
            return
        
        print("\n" + "=" * 70)
        print("Sample Chunks")
        print("=" * 70)
        
        for i, chunk in enumerate(chunks, 1):
            print(f"\nChunk {i}:")
            print(f"  ID: {chunk.chunk_id}")
            print(f"  Heading: {chunk.heading or 'N/A'}")
            print(f"  Words: {chunk.word_count}, Characters: {chunk.char_count}")
            print(f"  Text: {chunk.text[:100]}{'...' if len(chunk.text) > 100 else ''}")
            print("-" * 70)


def main():
    """Main entry point for the pipeline."""
    parser = argparse.ArgumentParser(
        description="NCERT Text Processing Pipeline - Phase 1 & 2"
    )
    parser.add_argument(
        '--input-dir',
        type=str,
        default='input',
        help='Input directory containing text files (default: input)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='output',
        help='Output directory for JSON files (default: output)'
    )
    
    args = parser.parse_args()
    
    # Create and run pipeline
    try:
        pipeline = NCERTProcessingPipeline(
            input_dir=args.input_dir,
            output_dir=args.output_dir
        )
        pipeline.run()
        
    except KeyboardInterrupt:
        print("\n\n⚠ Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Pipeline failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
