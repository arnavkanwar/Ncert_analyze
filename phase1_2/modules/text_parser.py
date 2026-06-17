"""
Text Parser Module - Phase 1 & 2

Reads text files from a folder structure and extracts content.
Designed to easily support Markdown files in the future.
"""

import os
from pathlib import Path
from typing import List, Dict, Optional
import re


class TextParser:
    """
    Parser for reading text files with support for folder structures.
    
    Current: Supports .txt files
    Future: Will support .md files with heading extraction
    """
    
    def __init__(self, root_dir: str):
        """
        Initialize parser with root directory.
        
        Args:
            root_dir: Root directory containing text files
        """
        self.root_dir = Path(root_dir)
        self.supported_extensions = ['.txt', '.md']
    
    def find_all_files(self) -> List[Path]:
        """
        Recursively find all text files in the directory.
        
        Returns:
            List of file paths
        """
        files = []
        for ext in self.supported_extensions:
            files.extend(self.root_dir.rglob(f'*{ext}'))
        
        print(f"Found {len(files)} text files in {self.root_dir}")
        return files
    
    def read_file(self, file_path: Path) -> str:
        """
        Read content from a text file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            File content as string
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            return content
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
            return ""
    
    def extract_metadata_from_path(self, file_path: Path) -> Dict[str, str]:
        """
        Extract metadata from file path structure.
        
        Expected structure: input/Class_XX/Subject/Book_X/Chapter_XX/file.txt
        
        Args:
            file_path: Path to the file
            
        Returns:
            Dictionary with metadata
        """
        parts = file_path.parts
        metadata = {
            "board": "NCERT",
            "class": "",
            "subject": "",
            "book": "",
            "chapter": "",
            "source_file": str(file_path)
        }
        
        try:
            # Find Class directory
            for i, part in enumerate(parts):
                if part.startswith('Class_'):
                    # Extract class number
                    class_match = re.search(r'Class_(\d+)', part)
                    if class_match:
                        metadata["class"] = class_match.group(1)
                    
                    # Extract subject (next level)
                    if i + 1 < len(parts):
                        metadata["subject"] = parts[i + 1]
                    
                    # Extract book (next level)
                    if i + 2 < len(parts):
                        book_match = re.search(r'Book_(\d+)', parts[i + 2])
                        if book_match:
                            metadata["book"] = f"Book_{book_match.group(1)}"
                    
                    # Extract chapter (next level)
                    if i + 3 < len(parts):
                        chapter_match = re.search(r'Chapter_(\d+)', parts[i + 3])
                        if chapter_match:
                            metadata["chapter"] = f"Chapter_{chapter_match.group(1).zfill(2)}"
                    
                    break
        
        except Exception as e:
            print(f"Warning: Could not extract full metadata from path {file_path}: {e}")
        
        return metadata
    
    def extract_simple_headings(self, content: str) -> List[Dict[str, str]]:
        """
        Extract simple headings from text.
        
        For .txt files: Looks for lines in ALL CAPS or numbered sections
        For .md files (future): Will parse markdown headings
        
        Args:
            content: Text content
            
        Returns:
            List of headings with line numbers
        """
        headings = []
        lines = content.split('\n')
        
        for i, line in enumerate(lines):
            line_stripped = line.strip()
            
            # Skip empty lines
            if not line_stripped:
                continue
            
            # Check for numbered headings (e.g., "1.1 Introduction")
            if re.match(r'^\d+(\.\d+)*\s+[A-Z]', line_stripped):
                headings.append({
                    "line_number": i + 1,
                    "text": line_stripped,
                    "level": 1
                })
            
            # Check for all caps headings (if short enough)
            elif line_stripped.isupper() and len(line_stripped) < 100 and len(line_stripped) > 3:
                headings.append({
                    "line_number": i + 1,
                    "text": line_stripped,
                    "level": 2
                })
        
        return headings
    
    def parse_file_with_metadata(self, file_path: Path) -> Dict[str, any]:
        """
        Parse a file and return content with metadata.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Dictionary with content and metadata
        """
        content = self.read_file(file_path)
        metadata = self.extract_metadata_from_path(file_path)
        headings = self.extract_simple_headings(content)
        
        return {
            "content": content,
            "metadata": metadata,
            "headings": headings,
            "file_path": file_path
        }


# Future support for Markdown files
class MarkdownParser(TextParser):
    """
    Extended parser for Markdown files.
    To be implemented in future phases.
    """
    
    def extract_markdown_headings(self, content: str) -> List[Dict[str, str]]:
        """
        Extract markdown-style headings (# ## ###).
        
        Args:
            content: Markdown content
            
        Returns:
            List of headings with levels
        """
        headings = []
        lines = content.split('\n')
        
        for i, line in enumerate(lines):
            if line.strip().startswith('#'):
                # Count heading level
                level = 0
                for char in line:
                    if char == '#':
                        level += 1
                    else:
                        break
                
                heading_text = line.lstrip('#').strip()
                headings.append({
                    "line_number": i + 1,
                    "text": heading_text,
                    "level": level
                })
        
        return headings


if __name__ == "__main__":
    # Example usage
    parser = TextParser("input")
    
    # Find all files
    files = parser.find_all_files()
    
    # Parse first file if available
    if files:
        result = parser.parse_file_with_metadata(files[0])
        print(f"\nParsed: {result['file_path']}")
        print(f"Metadata: {result['metadata']}")
        print(f"Headings found: {len(result['headings'])}")
        print(f"Content length: {len(result['content'])} characters")
