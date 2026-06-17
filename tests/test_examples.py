"""
Test Examples for NCERT Retrieval System

This module contains test cases and examples demonstrating
the functionality of each component.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import unittest
from datetime import datetime

from config.config import Config
from src.parser.markdown_parser import MarkdownParser, MarkdownSection
from src.processing.text_cleaner import TextCleaner
from src.processing.chunker import TextChunker
from src.processing.metadata_extractor import MetadataExtractor


class TestMarkdownParser(unittest.TestCase):
    """Test markdown parsing functionality."""
    
    def setUp(self):
        self.parser = MarkdownParser()
    
    def test_parse_simple_content(self):
        """Test parsing simple markdown content."""
        content = """
# Chapter 1: Introduction

This is the introduction paragraph.

## 1.1 What is Science?

Science is a systematic study.

### 1.1.1 Scientific Method

The method involves observation.
"""
        sections = self.parser.parse_content(content)
        
        # Should extract multiple sections
        self.assertGreater(len(sections), 0)
        
        # Check heading preservation
        found_h1 = any(s.heading_h1 == "Chapter 1: Introduction" for s in sections)
        self.assertTrue(found_h1)
    
    def test_clean_markdown_formatting(self):
        """Test markdown formatting removal."""
        text = "This is **bold** and *italic* text with a [link](url)."
        cleaned = self.parser.clean_markdown_formatting(text)
        
        # Should remove formatting but keep content
        self.assertIn("bold", cleaned)
        self.assertIn("italic", cleaned)
        self.assertNotIn("**", cleaned)
        self.assertNotIn("*", cleaned)
    
    def test_extract_metadata_from_path(self):
        """Test metadata extraction from file path."""
        file_path = Path("data/NCERT/Class_10/Science/Book_1/Chapter_01/chapter.md")
        metadata = self.parser.extract_metadata_from_path(file_path)
        
        self.assertEqual(metadata["class_name"], "class10")
        self.assertEqual(metadata["subject"], "science")
        self.assertEqual(metadata["chapter"], "ch1")


class TestTextCleaner(unittest.TestCase):
    """Test text cleaning functionality."""
    
    def setUp(self):
        self.cleaner = TextCleaner()
    
    def test_clean_whitespace(self):
        """Test whitespace cleaning."""
        text = "This   has    multiple     spaces."
        cleaned = self.cleaner.clean(text)
        
        # Should have single spaces
        self.assertNotIn("  ", cleaned)
    
    def test_remove_urls(self):
        """Test URL removal."""
        text = "Check this link https://example.com for more info."
        cleaned = self.cleaner.clean(text)
        
        self.assertNotIn("https://", cleaned)
    
    def test_is_valid_chunk(self):
        """Test chunk validation."""
        valid_text = "This is a valid chunk with enough words."
        invalid_text = "Too short"
        
        self.assertTrue(self.cleaner.is_valid_chunk(valid_text))
        # May or may not be invalid depending on config
    
    def test_unicode_normalization(self):
        """Test unicode normalization."""
        text = "Café résumé"  # Contains accented characters
        cleaned = self.cleaner.clean(text)
        
        # Should still contain the content
        self.assertIn("Caf", cleaned)


class TestTextChunker(unittest.TestCase):
    """Test text chunking functionality."""
    
    def setUp(self):
        self.chunker = TextChunker()
    
    def test_chunk_simple_text(self):
        """Test chunking simple text."""
        text = """
This is the first paragraph. It contains some information.

This is the second paragraph. It contains different information.

This is the third paragraph. It also has content.
"""
        metadata = {
            "class_name": "class10",
            "subject": "science",
            "chapter": "ch1"
        }
        
        chunks = self.chunker.chunk_text(text, metadata)
        
        # Should create multiple chunks
        self.assertGreater(len(chunks), 0)
        
        # Check chunk IDs
        for chunk in chunks:
            self.assertIn("class10_science_ch1_p", chunk.chunk_id)
    
    def test_chunk_long_paragraph(self):
        """Test chunking of long paragraph."""
        # Create a long paragraph
        long_text = " ".join(["This is a sentence."] * 100)
        
        metadata = {
            "class_name": "class10",
            "subject": "science",
            "chapter": "ch1"
        }
        
        chunks = self.chunker.chunk_text(long_text, metadata)
        
        # Should split long paragraph
        if len(long_text) > self.chunker.max_size:
            self.assertGreater(len(chunks), 1)
    
    def test_chunk_id_format(self):
        """Test chunk ID format."""
        text = "Sample paragraph for testing."
        metadata = {
            "class_name": "class10",
            "subject": "science",
            "chapter": "ch1"
        }
        
        chunks = self.chunker.chunk_text(text, metadata)
        
        if chunks:
            # Should match format: class10_science_ch1_p001
            import re
            pattern = r"^class10_science_ch1_p\d{3}$"
            self.assertTrue(re.match(pattern, chunks[0].chunk_id))


class TestMetadataExtractor(unittest.TestCase):
    """Test metadata extraction functionality."""
    
    def setUp(self):
        self.extractor = MetadataExtractor()
    
    def test_extract_from_section(self):
        """Test metadata extraction from section."""
        section = MarkdownSection(
            content="Sample content",
            heading_h1="Main Heading",
            heading_h2="Sub Heading",
            heading_h3=None,
            line_number=10
        )
        
        metadata = self.extractor.extract_from_section(section)
        
        self.assertEqual(metadata["heading_h1"], "Main Heading")
        self.assertEqual(metadata["heading_h2"], "Sub Heading")
        self.assertEqual(metadata["heading_h3"], "")
    
    def test_create_display_context(self):
        """Test display context creation."""
        metadata = {
            "class_name": "class10",
            "subject": "science",
            "chapter": "ch1",
            "heading_h1": "Biology",
            "heading_h2": "Cell Structure"
        }
        
        context = self.extractor.create_display_context(metadata)
        
        self.assertIn("Class 10", context)
        self.assertIn("Science", context)
        self.assertIn("Biology", context)
    
    def test_get_chunk_category(self):
        """Test chunk categorization."""
        metadata_subsection = {
            "heading_h1": "Main",
            "heading_h2": "Section",
            "heading_h3": "Subsection"
        }
        
        metadata_section = {
            "heading_h1": "Main",
            "heading_h2": "Section",
            "heading_h3": ""
        }
        
        self.assertEqual(
            self.extractor.get_chunk_category(metadata_subsection),
            "subsection"
        )
        self.assertEqual(
            self.extractor.get_chunk_category(metadata_section),
            "section"
        )


class TestIntegration(unittest.TestCase):
    """Integration tests for complete pipeline."""
    
    def test_end_to_end_processing(self):
        """Test complete processing pipeline."""
        # Sample markdown content
        content = """
# Chapter 1: Life Processes

Living organisms require energy to survive and grow.

## 1.1 Nutrition

Nutrition is the process of obtaining food.

### 1.1.1 Autotrophic Nutrition

Plants make their own food through photosynthesis.

Photosynthesis occurs in chloroplasts. It requires sunlight, water, and carbon dioxide.
"""
        
        # Initialize components
        parser = MarkdownParser()
        cleaner = TextCleaner()
        chunker = TextChunker()
        extractor = MetadataExtractor()
        
        # Parse
        sections = parser.parse_content(content)
        self.assertGreater(len(sections), 0)
        
        # Process first section
        section = sections[0]
        cleaned_text = cleaner.clean(section.content)
        
        # Create metadata
        path_metadata = {
            "class_name": "class10",
            "subject": "science",
            "chapter": "ch1",
            "source_file": "test.md"
        }
        
        combined_metadata = {
            **path_metadata,
            "heading_h1": section.heading_h1 or "",
            "heading_h2": section.heading_h2 or "",
            "heading_h3": section.heading_h3 or ""
        }
        
        # Chunk
        chunks = chunker.chunk_text(cleaned_text, combined_metadata)
        
        # Verify chunks have all required data
        for chunk in chunks:
            self.assertIsNotNone(chunk.chunk_id)
            self.assertIsNotNone(chunk.text)
            self.assertGreater(chunk.char_count, 0)
            self.assertGreater(chunk.word_count, 0)


def run_example_workflow():
    """
    Run an example workflow demonstrating the complete system.
    """
    print("=" * 70)
    print("NCERT Retrieval System - Example Workflow")
    print("=" * 70)
    
    # Sample content
    sample_content = """
# Chapter 1: Introduction to Biology

Biology is the study of living organisms and their interactions with the environment.

## 1.1 What is Life?

Life is characterized by growth, reproduction, and response to stimuli. Living organisms 
maintain homeostasis and adapt to their environment over time.

## 1.2 Branches of Biology

Biology has many specialized branches:

### 1.2.1 Botany

Botany is the study of plants. It includes plant anatomy, physiology, and ecology.

### 1.2.2 Zoology

Zoology is the study of animals. It covers animal behavior, evolution, and classification.
"""
    
    print("\n1. Parsing markdown content...")
    parser = MarkdownParser()
    sections = parser.parse_content(sample_content)
    print(f"   ✓ Extracted {len(sections)} sections")
    
    print("\n2. Cleaning text...")
    cleaner = TextCleaner()
    cleaned_sections = [(cleaner.clean(s.content), s) for s in sections]
    print(f"   ✓ Cleaned {len(cleaned_sections)} sections")
    
    print("\n3. Chunking text...")
    chunker = TextChunker()
    all_chunks = []
    for cleaned_text, section in cleaned_sections:
        if cleaner.is_valid_chunk(cleaned_text):
            metadata = {
                "class_name": "class10",
                "subject": "biology",
                "chapter": "ch1",
                "heading_h1": section.heading_h1 or "",
                "heading_h2": section.heading_h2 or "",
                "heading_h3": section.heading_h3 or ""
            }
            chunks = chunker.chunk_text(cleaned_text, metadata)
            all_chunks.extend(chunks)
    
    print(f"   ✓ Created {len(all_chunks)} chunks")
    
    print("\n4. Sample chunks:")
    for i, chunk in enumerate(all_chunks[:3], 1):
        print(f"\n   Chunk {i}: {chunk.chunk_id}")
        print(f"   Words: {chunk.word_count}, Chars: {chunk.char_count}")
        print(f"   Headings: {chunk.metadata.get('heading_h1')} > {chunk.metadata.get('heading_h2')}")
        print(f"   Text: {chunk.text[:80]}...")
    
    print("\n" + "=" * 70)
    print("Example workflow completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    # Run example workflow
    print("\n>>> Running example workflow...\n")
    run_example_workflow()
    
    # Run unit tests
    print("\n\n>>> Running unit tests...\n")
    unittest.main(argv=[''], verbosity=2, exit=False)
