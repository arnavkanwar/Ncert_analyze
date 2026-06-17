"""
Metadata Generator and JSON Exporter - Phase 1 & 2

Generates structured metadata for chunks and exports to JSON.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict
from paragraph_chunker import TextChunk


class MetadataGenerator:
    """
    Generates and manages metadata for text chunks.
    """
    
    def __init__(self):
        """Initialize metadata generator."""
        self.schema_version = "1.0"
    
    def generate_chunk_metadata(self, chunk: TextChunk) -> Dict:
        """
        Generate complete metadata for a chunk.
        
        Args:
            chunk: TextChunk object
            
        Returns:
            Dictionary with complete metadata
        """
        metadata = {
            "chunk_id": chunk.chunk_id,
            "board": chunk.metadata.get("board", "NCERT"),
            "class": chunk.metadata.get("class", ""),
            "subject": chunk.metadata.get("subject", ""),
            "book": chunk.metadata.get("book", ""),
            "chapter": chunk.metadata.get("chapter", ""),
            "heading": chunk.heading or "",
            "paragraph_number": chunk.paragraph_number,
            "word_count": chunk.word_count,
            "char_count": chunk.char_count,
            "source_file": chunk.metadata.get("source_file", ""),
            "created_at": datetime.now().isoformat(),
            "schema_version": self.schema_version
        }
        
        return metadata
    
    def chunk_to_json_format(self, chunk: TextChunk) -> Dict:
        """
        Convert chunk to JSON format with text and metadata.
        
        Args:
            chunk: TextChunk object
            
        Returns:
            Dictionary ready for JSON export
        """
        return {
            "chunk_id": chunk.chunk_id,
            "board": chunk.metadata.get("board", "NCERT"),
            "class": chunk.metadata.get("class", ""),
            "subject": chunk.metadata.get("subject", ""),
            "book": chunk.metadata.get("book", ""),
            "chapter": chunk.metadata.get("chapter", ""),
            "heading": chunk.heading or "",
            "paragraph_number": chunk.paragraph_number,
            "text": chunk.text
        }


class JSONExporter:
    """
    Exports chunks to structured JSON files.
    """
    
    def __init__(self, output_dir: str = "output"):
        """
        Initialize JSON exporter.
        
        Args:
            output_dir: Directory for output files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def export_chunks(
        self,
        chunks: List[TextChunk],
        output_filename: str = None
    ) -> str:
        """
        Export chunks to JSON file.
        
        Args:
            chunks: List of TextChunk objects
            output_filename: Optional custom filename
            
        Returns:
            Path to the output file
        """
        if not chunks:
            print("Warning: No chunks to export")
            return ""
        
        # Generate filename if not provided
        if not output_filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            first_chunk = chunks[0]
            class_num = first_chunk.metadata.get("class", "0")
            subject = first_chunk.metadata.get("subject", "unknown")
            output_filename = f"chunks_class{class_num}_{subject}_{timestamp}.json"
        
        output_path = self.output_dir / output_filename
        
        # Convert chunks to JSON format
        metadata_gen = MetadataGenerator()
        chunks_data = []
        
        for chunk in chunks:
            chunk_dict = metadata_gen.chunk_to_json_format(chunk)
            chunks_data.append(chunk_dict)
        
        # Create output structure
        output_data = {
            "metadata": {
                "total_chunks": len(chunks),
                "schema_version": "1.0",
                "created_at": datetime.now().isoformat(),
                "board": chunks[0].metadata.get("board", "NCERT"),
                "class": chunks[0].metadata.get("class", ""),
                "subject": chunks[0].metadata.get("subject", "")
            },
            "chunks": chunks_data
        }
        
        # Write to file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Exported {len(chunks)} chunks to: {output_path}")
        return str(output_path)
    
    def export_single_chunk(self, chunk: TextChunk, output_filename: str) -> str:
        """
        Export a single chunk to JSON file.
        
        Args:
            chunk: TextChunk object
            output_filename: Output filename
            
        Returns:
            Path to the output file
        """
        return self.export_chunks([chunk], output_filename)
    
    def export_summary(self, chunks: List[TextChunk]) -> str:
        """
        Export a summary of chunks (metadata only, no text).
        
        Args:
            chunks: List of TextChunk objects
            
        Returns:
            Path to the summary file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"chunks_summary_{timestamp}.json"
        output_path = self.output_dir / output_filename
        
        metadata_gen = MetadataGenerator()
        summary_data = []
        
        for chunk in chunks:
            metadata = metadata_gen.generate_chunk_metadata(chunk)
            summary_data.append(metadata)
        
        # Create summary structure
        output_data = {
            "metadata": {
                "total_chunks": len(chunks),
                "created_at": datetime.now().isoformat()
            },
            "chunks_metadata": summary_data
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Exported summary to: {output_path}")
        return str(output_path)


if __name__ == "__main__":
    # Example usage
    from paragraph_chunker import TextChunk
    
    # Create sample chunk
    sample_chunk = TextChunk(
        chunk_id="class10_science_ch1_p001",
        text="A chemical reaction is a process in which substances are converted into different substances.",
        paragraph_number=1,
        metadata={
            "board": "NCERT",
            "class": "10",
            "subject": "Science",
            "book": "Book_1",
            "chapter": "Chapter_01",
            "source_file": "input/Class_10/Science/Book_1/Chapter_01/chapter.txt"
        },
        word_count=14,
        char_count=93,
        heading="Chemical Reactions"
    )
    
    # Export to JSON
    exporter = JSONExporter(output_dir="output")
    exporter.export_chunks([sample_chunk])
    
    print("\nSample chunk exported successfully!")
