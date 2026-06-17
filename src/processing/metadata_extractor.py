"""
Metadata Extraction Module

Extracts and structures metadata from markdown sections and file paths.
Combines information from multiple sources to create rich metadata for each chunk.
"""

from datetime import datetime
from pathlib import Path
from typing import Dict, Optional
import logging

from config.config import Config
from src.parser.markdown_parser import MarkdownSection

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MetadataExtractor:
    """
    Extracts and enriches metadata for text chunks.
    
    Combines:
    - Path-based metadata (class, subject, chapter)
    - Section metadata (headings)
    - Content metadata (statistics, timestamps)
    """
    
    def __init__(self, schema_version: str = None):
        """
        Initialize MetadataExtractor.
        
        Args:
            schema_version: Version string for metadata schema
        """
        self.schema_version = schema_version or Config.SCHEMA_VERSION
        self.required_fields = Config.METADATA_FIELDS
    
    def extract_full_metadata(
        self,
        section: MarkdownSection,
        path_metadata: Dict,
        chunk_id: str,
        chunk_stats: Dict
    ) -> Dict:
        """
        Extract complete metadata for a chunk.
        
        Args:
            section: MarkdownSection object with heading context
            path_metadata: Metadata extracted from file path
            chunk_id: Generated chunk identifier
            chunk_stats: Statistics about the chunk (char_count, word_count)
            
        Returns:
            Complete metadata dictionary
        """
        metadata = {
            # Identifiers
            "chunk_id": chunk_id,
            "source_file": path_metadata.get("source_file", ""),
            
            # Educational hierarchy
            "class_name": path_metadata.get("class_name", ""),
            "subject": path_metadata.get("subject", ""),
            "book": path_metadata.get("book", ""),
            "chapter": path_metadata.get("chapter", ""),
            
            # Content hierarchy
            "heading_h1": section.heading_h1 or "",
            "heading_h2": section.heading_h2 or "",
            "heading_h3": section.heading_h3 or "",
            
            # Content statistics
            "paragraph_number": chunk_stats.get("paragraph_number", 0),
            "char_count": chunk_stats.get("char_count", 0),
            "word_count": chunk_stats.get("word_count", 0),
            
            # Temporal
            "created_at": datetime.now().isoformat(),
            
            # Schema version
            "schema_version": self.schema_version,
            
            # Optional: Line reference
            "line_number": section.line_number if hasattr(section, 'line_number') else 0
        }
        
        # Validate required fields
        self._validate_metadata(metadata)
        
        return metadata
    
    def extract_from_section(self, section: MarkdownSection) -> Dict:
        """
        Extract metadata specifically from a MarkdownSection.
        
        Args:
            section: MarkdownSection object
            
        Returns:
            Metadata dictionary with heading information
        """
        return {
            "heading_h1": section.heading_h1 or "",
            "heading_h2": section.heading_h2 or "",
            "heading_h3": section.heading_h3 or "",
            "line_number": section.line_number if hasattr(section, 'line_number') else 0
        }
    
    def extract_from_path(self, file_path: Path) -> Dict:
        """
        Extract metadata from file path structure.
        
        Expected: data/NCERT/Class_XX/Subject/Book_X/Chapter_XX/chapter.md
        
        Args:
            file_path: Path to the markdown file
            
        Returns:
            Metadata dictionary with class, subject, book, chapter
        """
        import re
        
        parts = file_path.parts
        metadata = {
            "source_file": str(file_path),
            "class_name": "",
            "subject": "",
            "book": "",
            "chapter": ""
        }
        
        try:
            # Find NCERT index
            if "NCERT" in parts:
                ncert_idx = parts.index("NCERT")
                
                # Extract class (e.g., "Class_10" -> "class10")
                if ncert_idx + 1 < len(parts):
                    class_part = parts[ncert_idx + 1]
                    class_match = re.search(r'(\d+)', class_part)
                    if class_match:
                        metadata["class_name"] = f"class{class_match.group(1)}"
                
                # Extract subject (e.g., "Science" -> "science")
                if ncert_idx + 2 < len(parts):
                    metadata["subject"] = parts[ncert_idx + 2].lower()
                
                # Extract book (e.g., "Book_1" -> "book1")
                if ncert_idx + 3 < len(parts):
                    book_part = parts[ncert_idx + 3]
                    book_match = re.search(r'(\d+)', book_part)
                    if book_match:
                        metadata["book"] = f"book{book_match.group(1)}"
                
                # Extract chapter (e.g., "Chapter_01" -> "ch1")
                if ncert_idx + 4 < len(parts):
                    chapter_part = parts[ncert_idx + 4]
                    chapter_match = re.search(r'(\d+)', chapter_part)
                    if chapter_match:
                        metadata["chapter"] = f"ch{int(chapter_match.group(1))}"
            
        except (ValueError, IndexError) as e:
            logger.warning(f"Could not extract full metadata from path {file_path}: {e}")
        
        return metadata
    
    def create_display_context(self, metadata: Dict) -> str:
        """
        Create a human-readable context string from metadata.
        
        Useful for displaying search results to users.
        
        Args:
            metadata: Metadata dictionary
            
        Returns:
            Formatted context string
        """
        parts = []
        
        # Add class and subject
        if metadata.get("class_name"):
            class_display = metadata["class_name"].replace("class", "Class ")
            parts.append(class_display)
        
        if metadata.get("subject"):
            parts.append(metadata["subject"].title())
        
        # Add chapter
        if metadata.get("chapter"):
            chapter_display = metadata["chapter"].replace("ch", "Chapter ")
            parts.append(chapter_display)
        
        # Add heading hierarchy
        headings = []
        for level in ["heading_h1", "heading_h2", "heading_h3"]:
            if metadata.get(level):
                headings.append(metadata[level])
        
        if headings:
            parts.append(" > ".join(headings))
        
        return " | ".join(parts)
    
    def create_hierarchical_context(self, metadata: Dict) -> Dict[str, str]:
        """
        Create hierarchical context dictionary for retrieval.
        
        Args:
            metadata: Metadata dictionary
            
        Returns:
            Dictionary with different levels of context
        """
        return {
            "chapter_context": f"{metadata.get('class_name', '')} {metadata.get('subject', '')} {metadata.get('chapter', '')}",
            "section_context": " > ".join(filter(None, [
                metadata.get("heading_h1", ""),
                metadata.get("heading_h2", ""),
                metadata.get("heading_h3", "")
            ])),
            "full_context": self.create_display_context(metadata)
        }
    
    def _validate_metadata(self, metadata: Dict) -> bool:
        """
        Validate that metadata contains required fields.
        
        Args:
            metadata: Metadata dictionary to validate
            
        Returns:
            True if valid, raises ValueError if not
        """
        missing_fields = []
        
        for field in self.required_fields:
            if field not in metadata:
                missing_fields.append(field)
        
        if missing_fields:
            logger.warning(f"Missing metadata fields: {missing_fields}")
            # Don't raise error, just warn - some fields might be optional in practice
        
        return True
    
    def enrich_metadata(self, metadata: Dict, additional_data: Dict) -> Dict:
        """
        Enrich existing metadata with additional information.
        
        Args:
            metadata: Base metadata dictionary
            additional_data: Additional data to merge
            
        Returns:
            Enriched metadata dictionary
        """
        enriched = metadata.copy()
        enriched.update(additional_data)
        return enriched
    
    def get_chunk_category(self, metadata: Dict) -> str:
        """
        Categorize chunk based on metadata.
        
        Useful for filtering or grouping chunks.
        
        Args:
            metadata: Metadata dictionary
            
        Returns:
            Category string
        """
        # Example categorization logic
        if metadata.get("heading_h3"):
            return "subsection"
        elif metadata.get("heading_h2"):
            return "section"
        elif metadata.get("heading_h1"):
            return "chapter"
        else:
            return "paragraph"


if __name__ == "__main__":
    # Example usage
    from src.parser.markdown_parser import MarkdownSection
    from pathlib import Path
    
    extractor = MetadataExtractor()
    
    # Create sample section
    section = MarkdownSection(
        content="This is sample content about photosynthesis.",
        heading_h1="Biology",
        heading_h2="Plant Life",
        heading_h3="Photosynthesis",
        line_number=42
    )
    
    # Sample path
    file_path = Path("data/NCERT/Class_10/Science/Book_1/Chapter_03/chapter.md")
    
    # Extract path metadata
    path_metadata = extractor.extract_from_path(file_path)
    print("Path Metadata:")
    print(path_metadata)
    
    # Extract full metadata
    chunk_stats = {"paragraph_number": 5, "char_count": 150, "word_count": 25}
    full_metadata = extractor.extract_full_metadata(
        section=section,
        path_metadata=path_metadata,
        chunk_id="class10_science_ch3_p005",
        chunk_stats=chunk_stats
    )
    
    print("\nFull Metadata:")
    for key, value in full_metadata.items():
        print(f"  {key}: {value}")
    
    print("\nDisplay Context:")
    print(extractor.create_display_context(full_metadata))
