"""
Text Cleaner Module

Provides text cleaning and normalization utilities for processing
NCERT textbook content before chunking and embedding.
"""

import re
import unicodedata
from typing import Dict
import logging

from config.config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TextCleaner:
    """
    Text cleaning and normalization for educational content.
    
    Handles:
    - Unicode normalization
    - Whitespace cleaning
    - URL and HTML removal
    - Special character handling (preserves educational symbols)
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize TextCleaner with configuration.
        
        Args:
            config: Optional configuration dictionary. Uses Config.TEXT_CLEANING if not provided.
        """
        self.config = config or Config.TEXT_CLEANING
        
        # Compile regex patterns for efficiency
        self.url_pattern = re.compile(
            r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        )
        self.html_pattern = re.compile(r'<[^>]+>')
        self.extra_whitespace_pattern = re.compile(r'\s+')
        self.multiple_newlines_pattern = re.compile(r'\n{3,}')
        
        # Patterns for educational content (preserve these)
        self.math_symbols = set('×÷±≠≤≥∞∑∫√π°')
        
    def clean(self, text: str) -> str:
        """
        Apply all configured cleaning operations to text.
        
        Args:
            text: Raw text to clean
            
        Returns:
            Cleaned and normalized text
        """
        if not text or not isinstance(text, str):
            return ""
        
        # Normalize unicode
        if self.config.get('normalize_unicode', True):
            text = self._normalize_unicode(text)
        
        # Remove URLs
        if self.config.get('remove_urls', True):
            text = self._remove_urls(text)
        
        # Remove HTML tags
        if self.config.get('remove_html', True):
            text = self._remove_html(text)
        
        # Clean whitespace
        if self.config.get('remove_extra_whitespace', True):
            text = self._clean_whitespace(text)
        
        # Optional: lowercase (not recommended for NCERT content)
        if self.config.get('lowercase', False):
            text = text.lower()
        
        # Final trim
        text = text.strip()
        
        return text
    
    def _normalize_unicode(self, text: str) -> str:
        """
        Normalize unicode characters to standard form.
        
        Args:
            text: Text to normalize
            
        Returns:
            Normalized text
        """
        # Use NFKC normalization (compatibility composition)
        # This handles most unicode inconsistencies while preserving meaning
        return unicodedata.normalize('NFKC', text)
    
    def _remove_urls(self, text: str) -> str:
        """
        Remove URLs from text.
        
        Args:
            text: Text containing URLs
            
        Returns:
            Text with URLs removed
        """
        return self.url_pattern.sub('', text)
    
    def _remove_html(self, text: str) -> str:
        """
        Remove HTML tags from text.
        
        Args:
            text: Text containing HTML
            
        Returns:
            Text with HTML removed
        """
        return self.html_pattern.sub('', text)
    
    def _clean_whitespace(self, text: str) -> str:
        """
        Normalize whitespace in text.
        
        - Replaces multiple spaces with single space
        - Replaces excessive newlines with double newline
        - Removes trailing/leading whitespace from lines
        
        Args:
            text: Text with irregular whitespace
            
        Returns:
            Text with cleaned whitespace
        """
        # Clean each line
        lines = text.split('\n')
        cleaned_lines = [line.strip() for line in lines]
        
        # Join lines and normalize
        text = '\n'.join(cleaned_lines)
        
        # Replace multiple newlines with max 2
        text = self.multiple_newlines_pattern.sub('\n\n', text)
        
        # Replace multiple spaces with single space (but preserve newlines)
        lines = text.split('\n')
        cleaned_lines = [self.extra_whitespace_pattern.sub(' ', line) for line in lines]
        text = '\n'.join(cleaned_lines)
        
        return text
    
    def is_valid_chunk(self, text: str) -> bool:
        """
        Check if a text chunk meets minimum quality requirements.
        
        Args:
            text: Text chunk to validate
            
        Returns:
            True if chunk meets requirements, False otherwise
        """
        if not text or not text.strip():
            return False
        
        # Check minimum words
        min_words = self.config.get('min_words', 3)
        word_count = len(text.split())
        
        if word_count < min_words:
            return False
        
        # Check if chunk is not just special characters
        alphanumeric_chars = sum(c.isalnum() for c in text)
        if alphanumeric_chars < 5:  # At least 5 alphanumeric characters
            return False
        
        return True
    
    def clean_heading(self, heading: str) -> str:
        """
        Clean heading text specifically.
        
        Args:
            heading: Raw heading text
            
        Returns:
            Cleaned heading
        """
        if not heading:
            return ""
        
        # Remove markdown heading symbols
        heading = re.sub(r'^#{1,6}\s*', '', heading)
        
        # Remove numbering if present (e.g., "1.1 Introduction" -> "Introduction")
        # But preserve it for context - actually keep it
        
        # Clean whitespace
        heading = heading.strip()
        
        return heading
    
    def extract_keywords(self, text: str, top_n: int = 10) -> list:
        """
        Extract potential keywords from text (simple word frequency approach).
        
        This is a basic implementation. For production, consider using
        more sophisticated NLP techniques or the embedding model.
        
        Args:
            text: Text to extract keywords from
            top_n: Number of keywords to extract
            
        Returns:
            List of keywords
        """
        # Convert to lowercase for keyword extraction
        text_lower = text.lower()
        
        # Remove punctuation
        text_clean = re.sub(r'[^\w\s]', ' ', text_lower)
        
        # Split into words
        words = text_clean.split()
        
        # Filter stopwords (basic list for educational content)
        stopwords = {
            'the', 'is', 'are', 'was', 'were', 'a', 'an', 'and', 'or', 'but',
            'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'as',
            'this', 'that', 'these', 'those', 'it', 'its', 'be', 'been', 'being'
        }
        
        words_filtered = [w for w in words if w not in stopwords and len(w) > 3]
        
        # Count frequency
        word_freq = {}
        for word in words_filtered:
            word_freq[word] = word_freq.get(word, 0) + 1
        
        # Sort by frequency
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        
        # Return top N
        return [word for word, freq in sorted_words[:top_n]]


if __name__ == "__main__":
    # Example usage
    cleaner = TextCleaner()
    
    sample_text = """
    # This is a heading
    
    This   is    a    paragraph   with     extra     spaces.
    
    It contains a URL: https://example.com and some <b>HTML</b>.
    
    
    
    Multiple newlines above.
    """
    
    cleaned = cleaner.clean(sample_text)
    print("Original:")
    print(repr(sample_text))
    print("\nCleaned:")
    print(repr(cleaned))
    print("\nIs valid:", cleaner.is_valid_chunk(cleaned))
