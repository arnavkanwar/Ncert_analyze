"""Markdown to paragraph-level JSON chunk converter for NCERT retrieval."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class HeadingHierarchy:
    """Track current heading hierarchy while walking markdown lines."""

    h1: str = ""
    h2: str = ""
    h3: str = ""

    def update(self, level: int, text: str) -> None:
        if level == 1:
            self.h1 = text
            self.h2 = ""
            self.h3 = ""
        elif level == 2:
            self.h2 = text
            self.h3 = ""
        elif level == 3:
            self.h3 = text

    def as_single_heading(self) -> str:
        parts = [part for part in [self.h1, self.h2, self.h3] if part]
        return " > ".join(parts)


class MarkdownChunkConverter:
    """Convert chapter markdown files into paragraph-level JSON chunks."""

    HEADING_PATTERN = re.compile(r"^(#{1,3})\s+(.*?)\s*$")
    LINK_PATTERN = re.compile(r"\[([^\]]+)\]\([^\)]+\)")
    INLINE_CODE_PATTERN = re.compile(r"`([^`]+)`")
    STRONG_PATTERN = re.compile(r"\*\*(.*?)\*\*|__(.*?)__")
    EMPHASIS_PATTERN = re.compile(r"\*(.*?)\*|_(.*?)_")
    WHITESPACE_PATTERN = re.compile(r"\s+")

    def read_markdown(self, markdown_file: Path) -> List[str]:
        """Read markdown file lines with robust error handling."""
        if not markdown_file.exists():
            raise FileNotFoundError(f"Input markdown file not found: {markdown_file}")

        try:
            return markdown_file.read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            raise OSError(f"Failed to read markdown file '{markdown_file}': {exc}") from exc

    def clean_line(self, line: str) -> str:
        """Clean one markdown line while preserving educational text content."""
        text = line.replace("\t", " ").strip()

        # Remove leading unordered-list marker when used as paragraph text.
        text = re.sub(r"^[-*+]\s+", "", text)

        # Keep link labels and inline code text, remove markdown wrappers.
        text = self.LINK_PATTERN.sub(r"\1", text)
        text = self.INLINE_CODE_PATTERN.sub(r"\1", text)

        # Remove bold/italic wrappers while keeping the underlying words.
        text = self.STRONG_PATTERN.sub(lambda match: match.group(1) or match.group(2) or "", text)
        text = self.EMPHASIS_PATTERN.sub(lambda match: match.group(1) or match.group(2) or "", text)

        # Normalize repeated spaces after markdown cleanup.
        text = self.WHITESPACE_PATTERN.sub(" ", text)
        return text.strip()

    def build_base_metadata(
        self,
        markdown_file: Path,
        board: str = "NCERT",
        class_name: Optional[str] = None,
        subject: Optional[str] = None,
        book: Optional[str] = None,
        chapter: Optional[str] = None,
    ) -> Dict[str, str]:
        """Build metadata from folder structure, with optional overrides."""
        parts = markdown_file.parts

        extracted_class = class_name or ""
        extracted_subject = subject or ""
        extracted_book = book or ""
        extracted_chapter = chapter or ""

        for idx, part in enumerate(parts):
            if part.startswith("Class_") and not extracted_class:
                class_match = re.search(r"(\d+)", part)
                if class_match:
                    extracted_class = class_match.group(1)
                if idx + 1 < len(parts) and not extracted_subject:
                    extracted_subject = parts[idx + 1]
                if idx + 2 < len(parts) and not extracted_book:
                    extracted_book = parts[idx + 2]
                if idx + 3 < len(parts) and not extracted_chapter:
                    extracted_chapter = parts[idx + 3]
                break

        return {
            "board": board,
            "class": extracted_class,
            "subject": extracted_subject,
            "book": extracted_book,
            "chapter": extracted_chapter,
            "source_file": str(markdown_file),
        }

    @staticmethod
    def make_chunk_id(class_name: str, subject: str, chapter: str, paragraph_number: int) -> str:
        """Create stable chunk IDs in class10_science_ch1_p003 format."""
        class_part = f"class{class_name}" if class_name else "class0"
        subject_part = re.sub(r"[^a-z0-9]", "", subject.lower()) if subject else "unknown"

        chapter_digits = re.search(r"(\d+)", chapter or "")
        chapter_number = int(chapter_digits.group(1)) if chapter_digits else 0
        chapter_part = f"ch{chapter_number}"

        return f"{class_part}_{subject_part}_{chapter_part}_p{paragraph_number:03d}"

    def convert(self, markdown_file: Path, base_metadata: Dict[str, str]) -> List[Dict[str, object]]:
        """Convert markdown to paragraph chunk records with heading context."""
        lines = self.read_markdown(markdown_file)

        heading_state = HeadingHierarchy()
        paragraph_buffer: List[str] = []
        chunks: List[Dict[str, object]] = []
        paragraph_number = 1

        def flush_buffer() -> None:
            nonlocal paragraph_number
            if not paragraph_buffer:
                return

            text = " ".join(paragraph_buffer).strip()
            text = self.WHITESPACE_PATTERN.sub(" ", text)

            if not text:
                paragraph_buffer.clear()
                return

            chunk_id = self.make_chunk_id(
                class_name=base_metadata.get("class", ""),
                subject=base_metadata.get("subject", ""),
                chapter=base_metadata.get("chapter", ""),
                paragraph_number=paragraph_number,
            )

            chunk = {
                "board": base_metadata.get("board", "NCERT"),
                "class": base_metadata.get("class", ""),
                "subject": base_metadata.get("subject", ""),
                "book": base_metadata.get("book", ""),
                "chapter": base_metadata.get("chapter", ""),
                "heading": heading_state.as_single_heading(),
                "paragraph_number": paragraph_number,
                "chunk_id": chunk_id,
                "source_file": base_metadata.get("source_file", ""),
                "text": text,
            }
            chunks.append(chunk)
            paragraph_number += 1
            paragraph_buffer.clear()

        for raw_line in lines:
            stripped_line = raw_line.strip()

            # Blank line marks paragraph boundary.
            if not stripped_line:
                flush_buffer()
                continue

            heading_match = self.HEADING_PATTERN.match(stripped_line)
            if heading_match:
                flush_buffer()
                level = len(heading_match.group(1))
                heading_text = self.clean_line(heading_match.group(2))
                heading_state.update(level=level, text=heading_text)
                continue

            cleaned_line = self.clean_line(stripped_line)
            if cleaned_line:
                paragraph_buffer.append(cleaned_line)

        flush_buffer()
        logger.info("Converted %d paragraphs from %s", len(chunks), markdown_file)
        return chunks

    @staticmethod
    def write_output(chunks: List[Dict[str, object]], output_file: Path) -> Path:
        """Write chunk records to JSON file."""
        output_file.parent.mkdir(parents=True, exist_ok=True)
        try:
            with output_file.open("w", encoding="utf-8") as outfile:
                json.dump(chunks, outfile, indent=2, ensure_ascii=False)
        except OSError as exc:
            raise OSError(f"Failed to write output JSON '{output_file}': {exc}") from exc
        return output_file
