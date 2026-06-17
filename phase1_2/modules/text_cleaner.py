"""
Text Cleaning Module - Phase 1 & 2

Cleans and normalizes raw text content for chunking.
Removes noise while preserving educational content.
"""

import re
from typing import List


class TextCleaner:
    """
    Text cleaning and normalization for textbook content.
    """
    
    def __init__(self):
        """Initialize text cleaner with patterns."""
        # Pattern for page numbers (common formats)
        self.page_number_patterns = [
            r'^\s*Page\s+\d+\s*$',           # "Page 12"
            r'^\s*\d+\s*$',                   # Just numbers alone on a line
            r'^\s*-\s*\d+\s*-\s*$',          # "- 12 -"
            r'^\s*\[\s*\d+\s*\]\s*$',        # "[12]"
        ]
        
        # Pattern for multiple spaces
        self.multiple_spaces = re.compile(r' {2,}')
        
        # Pattern for multiple newlines
        self.multiple_newlines = re.compile(r'\n{3,}')
    
    def remove_page_numbers(self, text: str) -> str:
        """
        Remove page numbers from text.
        
        Args:
            text: Input text
            
        Returns:
            Text with page numbers removed
        """
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            # Check if line matches any page number pattern
            is_page_number = False
            for pattern in self.page_number_patterns:
                if re.match(pattern, line, re.IGNORECASE):
                    is_page_number = True
                    break
            
            if not is_page_number:
                cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
    
    def normalize_whitespace(self, text: str) -> str:
        """
        Normalize whitespace in text.
        
        - Replace multiple spaces with single space
        - Replace multiple newlines with double newline
        - Remove trailing/leading whitespace from lines
        
        Args:
            text: Input text
            
        Returns:
            Text with normalized whitespace
        """
        # Split into lines
        lines = text.split('\n')
        
        # Clean each line
        cleaned_lines = []
        for line in lines:
            # Remove leading/trailing whitespace
            line = line.strip()
            
            # Replace multiple spaces with single space
            line = self.multiple_spaces.sub(' ', line)
            
            cleaned_lines.append(line)
        
        # Join lines
        text = '\n'.join(cleaned_lines)
        
        # Replace multiple newlines with double newline (paragraph separator)
        text = self.multiple_newlines.sub('\n\n', text)
        
        return text
    
    def remove_empty_lines(self, text: str) -> str:
        """
        Remove completely empty lines but preserve paragraph breaks.
        
        Args:
            text: Input text
            
        Returns:
            Text with empty lines removed
        """
        lines = text.split('\n')
        
        # Remove empty lines but keep structure
        cleaned_lines = []
        prev_empty = False
        
        for line in lines:
            is_empty = not line.strip()
            
            if is_empty:
                # Only add one empty line between paragraphs
                if not prev_empty and cleaned_lines:
                    cleaned_lines.append('')
                prev_empty = True
            else:
                cleaned_lines.append(line)
                prev_empty = False
        
        return '\n'.join(cleaned_lines)
    
    def remove_extra_spaces(self, text: str) -> str:
        """
        Remove extra spaces within text.
        
        Args:
            text: Input text
            
        Returns:
            Text with single spaces
        """
        return self.multiple_spaces.sub(' ', text)
    
    def clean(self, text: str) -> str:
        """
        Apply all cleaning operations to text.
        
        Args:
            text: Raw input text
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Step 1: Remove page numbers
        text = self.remove_page_numbers(text)
        
        # Step 2: Normalize whitespace
        text = self.normalize_whitespace(text)
        
        # Step 3: Remove empty lines (while preserving paragraph structure)
        text = self.remove_empty_lines(text)
        
        # Step 4: Final trim
        text = text.strip()
        
        return text
    
    def is_valid_paragraph(self, paragraph: str, min_words: int = 5) -> bool:
        """
        Check if a paragraph is valid for chunking.
        
        Args:
            paragraph: Text paragraph
            min_words: Minimum number of words
            
        Returns:
            True if valid, False otherwise
        """
        if not paragraph or not paragraph.strip():
            return False
        
        # Count words
        words = paragraph.split()
        if len(words) < min_words:
            return False
        
        # Check if it's not just numbers or special characters
        alphanumeric_count = sum(c.isalnum() for c in paragraph)
        if alphanumeric_count < 10:
            return False
        
        return True


if __name__ == "__main__":
    # Example usage
    cleaner = TextCleaner()
    
    sample_text = """
Page 12

CHEMICAL REACTIONS


A chemical reaction is    a process   in which substances are converted.


Page 13

Reactants are substances that undergo change.    Products are the newly formed substances.



Multiple      spaces      here.
"""
    
    print("Original text:")
    print(repr(sample_text))
    
    cleaned = cleaner.clean(sample_text)
    
    print("\n\nCleaned text:")
    print(repr(cleaned))
    
    print("\n\nFormatted output:")
    print(cleaned)
