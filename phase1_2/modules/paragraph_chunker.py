"""
Paragraph Chunking Module - Phase 1 & 2

Splits cleaned text into paragraph-level chunks.
Each chunk remains independently readable.
"""

import re
from typing import List, Dict, Optional
from dataclasses import dataclass


@dataclass
class TextChunk:
    """Represents a text chunk with metadata."""
    chunk_id: str
    text: str
    paragraph_number: int
    metadata: Dict[str, str]
    word_count: int
    char_count: int
    heading: Optional[str] = None


class ParagraphChunker:
    """
    Splits text into paragraph-level chunks.
    
    Key principles:
    - Each chunk is one paragraph
    - Preserve heading context
    - Each chunk is independently readable
    - Do not merge unrelated paragraphs
    """
    
    def __init__(self):
        """Initialize paragraph chunker."""
        self.min_chunk_size = 20  # Minimum characters
        self.min_words = 5        # Minimum words per chunk
    
    def split_into_paragraphs(self, text: str) -> List[str]:
        """
        Split text into paragraphs based on double newlines.
        
        Args:
            text: Cleaned text
            
        Returns:
            List of paragraphs
        """
        # Split on double newlines (paragraph separator)
        paragraphs = text.split('\n\n')
        
        # Clean and filter paragraphs
        cleaned_paragraphs = []
        for para in paragraphs:
            para = para.strip()
            if para and len(para) >= self.min_chunk_size:
                # Check if it has minimum words
                if len(para.split()) >= self.min_words:
                    cleaned_paragraphs.append(para)
        
        return cleaned_paragraphs
    
    def is_heading(self, text: str) -> bool:
        """
        Check if a text line is likely a heading.
        
        Args:
            text: Text to check
            
        Returns:
            True if likely a heading
        """
        # Check for numbered headings (e.g., "1.1 Introduction")
        if re.match(r'^\d+(\.\d+)*\s+[A-Z]', text):
            return True
        
        # Check for all caps (if short)
        if text.isupper() and len(text) < 100:
            return True
        
        # Check if it ends with colon and is short
        if text.endswith(':') and len(text) < 100:
            return True
        
        return False
    
    def identify_headings_and_paragraphs(self, paragraphs: List[str]) -> List[Dict[str, any]]:
        """
        Identify which paragraphs are headings and which are content.
        
        Args:
            paragraphs: List of paragraph strings
            
        Returns:
            List of dictionaries with type and content
        """
        items = []
        
        for para in paragraphs:
            if self.is_heading(para):
                items.append({
                    "type": "heading",
                    "content": para
                })
            else:
                items.append({
                    "type": "paragraph",
                    "content": para
                })
        
        return items
    
    def create_chunks(
        self,
        paragraphs: List[str],
        base_metadata: Dict[str, str]
    ) -> List[TextChunk]:
        """
        Create chunks from paragraphs with metadata.
        
        Args:
            paragraphs: List of paragraph strings
            base_metadata: Base metadata (class, subject, etc.)
            
        Returns:
            List of TextChunk objects
        """
        chunks = []
        
        # Identify headings and paragraphs
        items = self.identify_headings_and_paragraphs(paragraphs)
        
        # Track current heading for context
        current_heading = None
        paragraph_number = 1
        
        for item in items:
            if item["type"] == "heading":
                # Update current heading
                current_heading = item["content"]
            
            elif item["type"] == "paragraph":
                # Create chunk for paragraph
                text = item["content"]
                
                # Generate chunk ID
                chunk_id = self._generate_chunk_id(base_metadata, paragraph_number)
                
                # Calculate statistics
                word_count = len(text.split())
                char_count = len(text)
                
                # Create chunk
                chunk = TextChunk(
                    chunk_id=chunk_id,
                    text=text,
                    paragraph_number=paragraph_number,
                    metadata=base_metadata,
                    word_count=word_count,
                    char_count=char_count,
                    heading=current_heading
                )
                
                chunks.append(chunk)
                paragraph_number += 1
        
        return chunks
    
    def _generate_chunk_id(self, metadata: Dict[str, str], paragraph_num: int) -> str:
        """
        Generate chunk ID in format: class10_science_ch1_p003
        
        Args:
            metadata: Metadata dictionary
            paragraph_num: Paragraph number
            
        Returns:
            Formatted chunk ID
        """
        # Extract and clean metadata values
        class_num = metadata.get("class", "0")
        subject = metadata.get("subject", "unknown").lower()
        chapter = metadata.get("chapter", "Chapter_00")
        
        # Clean subject name (remove spaces, special chars)
        subject = re.sub(r'[^a-z0-9]', '', subject.lower())
        
        # Extract chapter number
        chapter_match = re.search(r'(\d+)', chapter)
        chapter_num = chapter_match.group(1) if chapter_match else "0"
        
        # Format: class10_science_ch1_p003
        chunk_id = f"class{class_num}_{subject}_ch{chapter_num}_p{paragraph_num:03d}"
        
        return chunk_id
    
    def chunk_text(
        self,
        text: str,
        metadata: Dict[str, str]
    ) -> List[TextChunk]:
        """
        Main method to chunk text into paragraphs.
        
        Args:
            text: Cleaned text content
            metadata: Base metadata (class, subject, chapter, etc.)
            
        Returns:
            List of TextChunk objects
        """
        # Split into paragraphs
        paragraphs = self.split_into_paragraphs(text)
        
        if not paragraphs:
            print("Warning: No valid paragraphs found in text")
            return []
        
        # Create chunks
        chunks = self.create_chunks(paragraphs, metadata)
        
        print(f"Created {len(chunks)} chunks from {len(paragraphs)} paragraphs")
        
        return chunks


if __name__ == "__main__":
    # Example usage
    chunker = ParagraphChunker()
    
    sample_text = """
CHEMICAL REACTIONS

A chemical reaction is a process in which substances are converted into different substances. The substances that undergo change are called reactants.

The new substances formed are called products. Chemical reactions are represented by chemical equations.

1.1 Types of Reactions

There are several types of chemical reactions. Each type has specific characteristics.

Combination reactions occur when two or more substances combine to form a single new substance. This is also called a synthesis reaction.

Decomposition reactions are the opposite. A single compound breaks down into two or more simpler substances.
"""
    
    metadata = {
        "board": "NCERT",
        "class": "10",
        "subject": "Science",
        "book": "Book_1",
        "chapter": "Chapter_01"
    }
    
    chunks = chunker.chunk_text(sample_text, metadata)
    
    print(f"\nCreated {len(chunks)} chunks:\n")
    for chunk in chunks:
        print(f"ID: {chunk.chunk_id}")
        print(f"Heading: {chunk.heading}")
        print(f"Words: {chunk.word_count}, Chars: {chunk.char_count}")
        print(f"Text: {chunk.text[:80]}...")
        print("-" * 80)
