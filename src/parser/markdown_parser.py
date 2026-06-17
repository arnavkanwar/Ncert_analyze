"""
Markdown Parser Module

Parses markdown files and extracts structured content with heading hierarchy.
Supports NCERT textbook format with hierarchical sections.
"""

import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class MarkdownSection:
    """Represents a section in the markdown document."""
    content: str
    heading_h1: Optional[str] = None
    heading_h2: Optional[str] = None
    heading_h3: Optional[str] = None
    line_number: int = 0


class MarkdownParser:
    """
    Parses markdown files and extracts content with heading context.
    
    Features:
    - Preserves heading hierarchy (h1, h2, h3)
    - Extracts paragraphs while maintaining structure
    - Handles nested sections
    - Cleans markdown formatting
    """
    
    def __init__(self):
        self.heading_pattern = re.compile(r'^(#{1,6})\s+(.+)$', re.MULTILINE)
        self.code_block_pattern = re.compile(r'```[\s\S]*?```', re.MULTILINE)
        self.inline_code_pattern = re.compile(r'`[^`]+`')
        self.bold_pattern = re.compile(r'\*\*(.+?)\*\*')
        self.italic_pattern = re.compile(r'\*(.+?)\*')
        self.link_pattern = re.compile(r'\[([^\]]+)\]\([^\)]+\)')
        self.image_pattern = re.compile(r'!\[([^\]]*)\]\([^\)]+\)')
        
    def parse_file(self, file_path: Path) -> List[MarkdownSection]:
        """
        Parse a markdown file and return structured sections.
        
        Args:
            file_path: Path to the markdown file
            
        Returns:
            List of MarkdownSection objects with content and heading context
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            logger.info(f"Parsing file: {file_path}")
            return self.parse_content(content)
            
        except Exception as e:
            logger.error(f"Error parsing file {file_path}: {e}")
            return []
    
    def parse_content(self, content: str) -> List[MarkdownSection]:
        """
        Parse markdown content and extract sections with heading hierarchy.
        
        Args:
            content: Raw markdown content string
            
        Returns:
            List of MarkdownSection objects
        """
        sections = []
        lines = content.split('\n')
        
        # Track current heading context
        current_h1 = None
        current_h2 = None
        current_h3 = None
        
        # Accumulate paragraph content
        paragraph_buffer = []
        line_number = 0
        
        for i, line in enumerate(lines):
            line_number = i + 1
            stripped = line.strip()
            
            # Skip empty lines unless we're accumulating content
            if not stripped:
                if paragraph_buffer:
                    paragraph_buffer.append('')
                continue
            
            # Check if this is a heading
            heading_match = self.heading_pattern.match(stripped)
            
            if heading_match:
                # Save accumulated paragraph before processing heading
                if paragraph_buffer:
                    paragraph_text = '\n'.join(paragraph_buffer).strip()
                    if paragraph_text:
                        sections.append(MarkdownSection(
                            content=paragraph_text,
                            heading_h1=current_h1,
                            heading_h2=current_h2,
                            heading_h3=current_h3,
                            line_number=line_number - len(paragraph_buffer)
                        ))
                    paragraph_buffer = []
                
                # Update heading context
                heading_level = len(heading_match.group(1))
                heading_text = heading_match.group(2).strip()
                
                if heading_level == 1:
                    current_h1 = heading_text
                    current_h2 = None
                    current_h3 = None
                elif heading_level == 2:
                    current_h2 = heading_text
                    current_h3 = None
                elif heading_level == 3:
                    current_h3 = heading_text
                
            else:
                # Regular content line - add to paragraph buffer
                paragraph_buffer.append(line)
                
                # Check if this is the end of a paragraph (double newline or list item)
                if i + 1 < len(lines):
                    next_line = lines[i + 1].strip()
                    
                    # End paragraph on double newline or before heading
                    if not next_line or self.heading_pattern.match(next_line):
                        paragraph_text = '\n'.join(paragraph_buffer).strip()
                        if paragraph_text:
                            sections.append(MarkdownSection(
                                content=paragraph_text,
                                heading_h1=current_h1,
                                heading_h2=current_h2,
                                heading_h3=current_h3,
                                line_number=line_number - len(paragraph_buffer) + 1
                            ))
                        paragraph_buffer = []
        
        # Handle any remaining content
        if paragraph_buffer:
            paragraph_text = '\n'.join(paragraph_buffer).strip()
            if paragraph_text:
                sections.append(MarkdownSection(
                    content=paragraph_text,
                    heading_h1=current_h1,
                    heading_h2=current_h2,
                    heading_h3=current_h3,
                    line_number=line_number - len(paragraph_buffer) + 1
                ))
        
        logger.info(f"Extracted {len(sections)} sections")
        return sections
    
    def clean_markdown_formatting(self, text: str) -> str:
        """
        Remove markdown formatting while preserving content.
        
        Args:
            text: Text with markdown formatting
            
        Returns:
            Plain text without markdown syntax
        """
        # Remove code blocks (preserve content)
        text = self.code_block_pattern.sub(lambda m: m.group(0).replace('```', ''), text)
        
        # Remove inline code formatting
        text = self.inline_code_pattern.sub(lambda m: m.group(0).replace('`', ''), text)
        
        # Remove image syntax (keep alt text)
        text = self.image_pattern.sub(r'\1', text)
        
        # Remove link syntax (keep link text)
        text = self.link_pattern.sub(r'\1', text)
        
        # Remove bold formatting (keep text)
        text = self.bold_pattern.sub(r'\1', text)
        
        # Remove italic formatting (keep text)
        text = self.italic_pattern.sub(r'\1', text)
        
        return text
    
    def extract_metadata_from_path(self, file_path: Path) -> Dict[str, str]:
        """
        Extract metadata from the file path structure.
        
        Expected structure: data/NCERT/Class_XX/Subject/Book_X/Chapter_XX/chapter.md
        
        Args:
            file_path: Path to the markdown file
            
        Returns:
            Dictionary with extracted metadata
        """
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
            ncert_idx = parts.index("NCERT")
            
            # Extract class
            if ncert_idx + 1 < len(parts):
                class_part = parts[ncert_idx + 1]
                metadata["class_name"] = class_part.lower().replace('_', '')
            
            # Extract subject
            if ncert_idx + 2 < len(parts):
                metadata["subject"] = parts[ncert_idx + 2].lower()
            
            # Extract book
            if ncert_idx + 3 < len(parts):
                book_part = parts[ncert_idx + 3]
                metadata["book"] = book_part.lower().replace('_', '')
            
            # Extract chapter
            if ncert_idx + 4 < len(parts):
                chapter_part = parts[ncert_idx + 4]
                # Extract chapter number: Chapter_01 -> ch1
                chapter_match = re.search(r'(\d+)', chapter_part)
                if chapter_match:
                    metadata["chapter"] = f"ch{int(chapter_match.group(1))}"
            
        except (ValueError, IndexError) as e:
            logger.warning(f"Could not extract full metadata from path {file_path}: {e}")
        
        return metadata
    
    def find_markdown_files(self, root_dir: Path) -> List[Path]:
        """
        Recursively find all markdown files in the directory structure.
        
        Args:
            root_dir: Root directory to search
            
        Returns:
            List of paths to markdown files
        """
        markdown_files = list(root_dir.rglob("*.md"))
        logger.info(f"Found {len(markdown_files)} markdown files in {root_dir}")
        return markdown_files


if __name__ == "__main__":
    # Example usage
    parser = MarkdownParser()
    
    # Test with sample content
    sample_content = """
# Chapter 1: Introduction to Science

This is the first paragraph introducing science.

## 1.1 What is Science?

Science is a systematic enterprise. It builds and organizes knowledge.

## 1.2 Scientific Method

The scientific method involves observation and experimentation.

### 1.2.1 Observation

Careful watching and recording of phenomena.
"""
    
    sections = parser.parse_content(sample_content)
    for i, section in enumerate(sections):
        print(f"\nSection {i+1}:")
        print(f"H1: {section.heading_h1}")
        print(f"H2: {section.heading_h2}")
        print(f"H3: {section.heading_h3}")
        print(f"Content: {section.content[:100]}...")
