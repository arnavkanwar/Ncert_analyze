"""Heuristic filtering for retaining only real PYQ question chunks."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional


# Patterns that identify non-question content.
NON_QUESTION_PATTERNS = [
    re.compile(r"^\s*(?:section|part)\s+[a-z]\b", re.IGNORECASE),
    re.compile(r"^\s*general\s+instructions?\s*:?", re.IGNORECASE),
    re.compile(r"^\s*(?:maximum|total)\s+marks?\s*:", re.IGNORECASE),
    re.compile(r"^\s*time\s+(?:allowed|limit)\s*:", re.IGNORECASE),
    re.compile(r"^\s*(?:note|important)\s*[:\-]", re.IGNORECASE),
    re.compile(r"^\s*(?:all\s+questions?\s+(?:are\s+)?(?:compulsory|mandatory))", re.IGNORECASE),
    re.compile(r"^\s*(?:draw|answer\s+(?:in\s+)?(?:brief|detail))", re.IGNORECASE),
    re.compile(r"^\s*page\s+\d+\s*(?:of\s+\d+)?$", re.IGNORECASE),
    re.compile(r"^\s*\d+\s*$"),
    re.compile(r"^\s*(?:cbse|ncert|board)\s+(?:examination|exam|paper)", re.IGNORECASE),
    re.compile(r"^\s*(?:set|code)\s*[:-]?\s*[a-z0-9]+$", re.IGNORECASE),
]

# Positive signals for real questions.
QUESTION_SIGNALS = [
    re.compile(r"^\s*(?:Q\.?\s*)?\d+[\s.):-]", re.IGNORECASE),
    re.compile(r"\?"),
    re.compile(r"(?:explain|define|state|describe|calculate|find|show|prove|write)", re.IGNORECASE),
    re.compile(r"\(\s*[a-dA-D1-4]\s*\)"),
    re.compile(r"^\s*(?:i+\.?\s+|ii+\.?\s+|iii+\.?\s+)", re.IGNORECASE),
]

QUESTION_NUMBER_PATTERN = re.compile(r"^\s*(?:Q\.?\s*)?(\d+)\s*[.)\-:]", re.IGNORECASE)

MIN_QUESTION_WORDS = 6


def is_valid_question(text: str) -> bool:
    """Return True if text appears to be a genuine exam question."""
    stripped = (text or "").strip()
    if not stripped:
        return False

    if len(stripped.split()) < MIN_QUESTION_WORDS:
        return False

    for pattern in NON_QUESTION_PATTERNS:
        if pattern.search(stripped):
            return False

    return any(pattern.search(stripped) for pattern in QUESTION_SIGNALS)


def extract_question_number(text: str) -> Optional[int]:
    """Extract leading question number from text when present."""
    match = QUESTION_NUMBER_PATTERN.match((text or "").strip())
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def filter_pyq_chunks(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Keep only valid PYQ question-like chunks from a list of chunk dicts."""
    return [chunk for chunk in chunks if is_valid_question(str(chunk.get("text", "")))]
