"""
Text Chunker Module

Implements paragraph-level text chunking while preserving context
and maintaining interpretability of individual chunks.
"""

import re
from typing import List, Dict, Tuple
from dataclasses import dataclass
import logging

from config.config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TextChunk:
    """Represents a processed text chunk with metadata."""
    text: str
    chunk_id: str
    metadata: Dict
    char_count: int
    word_count: int
    
    def to_dict(self) -> Dict:
        """Convert chunk to dictionary format."""
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "char_count": self.char_count,
            "word_count": self.word_count,
            "metadata": self.metadata
        }


class TextChunker:
    """
    Paragraph-level text chunker with context preservation.
    
    Features:
    - Splits text into paragraph-level chunks
    - Respects minimum and maximum chunk sizes
    - Preserves heading context
    - Creates overlapping chunks when necessary
    - Generates structured chunk IDs
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize TextChunker with configuration.
        
        Args:
            config: Optional configuration dictionary. Uses Config.CHUNKING if not provided.
        """
        self.config = config or Config.CHUNKING
        self.min_size = self.config.get('min_chunk_size', 50)
        self.max_size = self.config.get('max_chunk_size', 1000)
        self.overlap = self.config.get('overlap', 50)
        self.split_delimiters = self.config.get('split_on', ["\n\n", "\n", ". "])
        
    def chunk_text(self, text: str, metadata: Dict) -> List[TextChunk]:
        """
        Split text into paragraph-level chunks.
        
        Args:
            text: Text to chunk
            metadata: Metadata to attach to chunks (class, subject, chapter, etc.)
            
        Returns:
            List of TextChunk objects
        """
        if not text or not text.strip():
            return []
        
        # First split by paragraphs (double newline)
        paragraphs = self._split_into_paragraphs(text)
        
        chunks = []
        paragraph_number = 1
        
        for para in paragraphs:
            para = para.strip()
            
            # Skip empty paragraphs
            if not para or len(para) < self.min_size:
                # If it's too small but not empty, combine with next or previous
                if para and chunks:
                    # Try to add to previous chunk if it won't exceed max size
                    last_chunk = chunks[-1]
                    combined_text = last_chunk.text + "\n\n" + para
                    if len(combined_text) <= self.max_size:
                        # Update the last chunk
                        chunks[-1] = TextChunk(
                            text=combined_text,
                            chunk_id=last_chunk.chunk_id,
                            metadata=last_chunk.metadata,
                            char_count=len(combined_text),
                            word_count=len(combined_text.split())
                        )
                        continue
                continue
            
            # If paragraph is within size limits, create chunk
            if len(para) <= self.max_size:
                chunk = self._create_chunk(para, metadata, paragraph_number)
                chunks.append(chunk)
                paragraph_number += 1
            else:
                # Paragraph is too large, need to split further
                sub_chunks = self._split_large_paragraph(para, metadata, paragraph_number)
                chunks.extend(sub_chunks)
                paragraph_number += len(sub_chunks)
        
        logger.info(f"Created {len(chunks)} chunks from text")
        return chunks
    
    def _split_into_paragraphs(self, text: str) -> List[str]:
        """
        Split text into paragraphs based on double newlines.
        
        Args:
            text: Text to split
            
        Returns:
            List of paragraph strings
        """
        # Split on double newlines
        paragraphs = re.split(r'\n\s*\n', text)
        
        # Clean up paragraphs
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        
        return paragraphs
    
    def _split_large_paragraph(self, text: str, metadata: Dict, 
                               base_paragraph_num: int) -> List[TextChunk]:
        """
        Split a large paragraph into smaller chunks with overlap.
        
        Args:
            text: Large paragraph text
            metadata: Metadata for chunks
            base_paragraph_num: Starting paragraph number
            
        Returns:
            List of TextChunk objects
        """
        chunks = []
        
        # Try splitting by sentences first
        sentences = self._split_by_sentences(text)
        
        current_chunk = ""
        chunk_num = base_paragraph_num
        
        for sentence in sentences:
            # Check if adding this sentence would exceed max size
            if current_chunk and len(current_chunk) + len(sentence) + 1 > self.max_size:
                # Save current chunk
                if current_chunk:
                    chunk = self._create_chunk(current_chunk, metadata, chunk_num)
                    chunks.append(chunk)
                    chunk_num += 1
                    
                    # Start new chunk with overlap from previous chunk
                    overlap_text = self._get_overlap_text(current_chunk, self.overlap)
                    current_chunk = overlap_text + " " + sentence if overlap_text else sentence
            else:
                # Add sentence to current chunk
                current_chunk = current_chunk + " " + sentence if current_chunk else sentence
        
        # Handle remaining chunk
        if current_chunk:
            chunk = self._create_chunk(current_chunk, metadata, chunk_num)
            chunks.append(chunk)
        
        return chunks
    
    def _split_by_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences.
        
        Args:
            text: Text to split
            
        Returns:
            List of sentences
        """
        # Simple sentence splitter (can be improved with NLTK or spaCy)
        # Handles common sentence endings
        sentence_pattern = re.compile(r'(?<=[.!?])\s+(?=[A-Z])')
        sentences = sentence_pattern.split(text)
        
        return [s.strip() for s in sentences if s.strip()]
    
    def _get_overlap_text(self, text: str, overlap_size: int) -> str:
        """
        Get the last N characters from text for overlap.
        
        Args:
            text: Source text
            overlap_size: Number of characters to overlap
            
        Returns:
            Overlap text
        """
        if len(text) <= overlap_size:
            return text
        
        # Try to break at word boundary
        overlap_text = text[-overlap_size:]
        first_space = overlap_text.find(' ')
        
        if first_space > 0:
            return overlap_text[first_space:].strip()
        
        return overlap_text
    
    def _create_chunk(self, text: str, metadata: Dict, paragraph_num: int) -> TextChunk:
        """
        Create a TextChunk object with metadata.
        
        Args:
            text: Chunk text
            metadata: Base metadata (class, subject, chapter, etc.)
            paragraph_num: Paragraph number
            
        Returns:
            TextChunk object
        """
        # Generate chunk ID
        chunk_id = self._generate_chunk_id(metadata, paragraph_num)
        
        # Create full metadata
        chunk_metadata = {
            **metadata,
            "paragraph_number": paragraph_num
        }
        
        # Calculate statistics
        char_count = len(text)
        word_count = len(text.split())
        
        return TextChunk(
            text=text,
            chunk_id=chunk_id,
            metadata=chunk_metadata,
            char_count=char_count,
            word_count=word_count
        )
    
    def _generate_chunk_id(self, metadata: Dict, paragraph_num: int) -> str:
        """
        Generate structured chunk ID.
        
        Format: class10_science_ch1_p003
        
        Args:
            metadata: Metadata dictionary with class, subject, chapter
            paragraph_num: Paragraph number
            
        Returns:
            Formatted chunk ID
        """
        class_name = metadata.get('class_name', 'unknown')
        subject = metadata.get('subject', 'unknown')
        chapter = metadata.get('chapter', 'ch0')
        
        # Clean values
        class_name = re.sub(r'[^a-z0-9]', '', class_name.lower())
        subject = re.sub(r'[^a-z0-9]', '', subject.lower())
        chapter = re.sub(r'[^a-z0-9]', '', chapter.lower())
        
        # Format: class10_science_ch1_p003
        chunk_id = f"{class_name}_{subject}_{chapter}_p{paragraph_num:03d}"
        
        return chunk_id
    
    def chunk_multiple_sections(self, sections: List[Tuple[str, Dict]]) -> List[TextChunk]:
        """
        Chunk multiple sections of text.
        
        Args:
            sections: List of (text, metadata) tuples
            
        Returns:
            List of all TextChunk objects
        """
        all_chunks = []
        
        for text, metadata in sections:
            chunks = self.chunk_text(text, metadata)
            all_chunks.extend(chunks)
        
        logger.info(f"Created total of {len(all_chunks)} chunks from {len(sections)} sections")
        return all_chunks


if __name__ == "__main__":
    # Example usage
    chunker = TextChunker()
    
    sample_text = """
This is the first paragraph. It contains several sentences. This helps demonstrate the chunking.

This is the second paragraph. It should be kept separate from the first paragraph.

This is a very long paragraph that might exceed the maximum chunk size limit. It contains many sentences that go on and on. We need to make sure this paragraph gets split into smaller chunks while maintaining overlap between them. The chunker should handle this gracefully. It should split at sentence boundaries when possible. This ensures that chunks remain readable and interpretable. The overlap helps maintain context across chunk boundaries.
"""
    
    metadata = {
        "class_name": "class10",
        "subject": "science",
        "chapter": "ch1",
        "heading_h1": "Introduction"
    }
    
    chunks = chunker.chunk_text(sample_text, metadata)
    
    print(f"Created {len(chunks)} chunks:\n")
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i+1}: {chunk.chunk_id}")
        print(f"  Words: {chunk.word_count}, Chars: {chunk.char_count}")
        print(f"  Text: {chunk.text[:100]}...")
        print()
