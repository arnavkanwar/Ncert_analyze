# Prompt 02 — PYQ Text Filtering

## Purpose
Use this prompt when improving how PYQ PDF content is cleaned before indexing, or when the sidebar shows garbage entries (headers, instructions, blank lines, page numbers) as if they were questions.

---

## The core problem

When a PYQ PDF is parsed, the raw text contains many non-question fragments:

```
── What gets extracted from a typical CBSE PYQ PDF ──────────────────────────
Section A: Multiple Choice Questions (1 mark each)       ← section header
Q1. Which of the following is a redox reaction?          ← real question ✓
    (a) NaCl + AgNO₃ → AgCl + NaNO₃
    (b) CuO + H₂ → Cu + H₂O
    (c) Both (a) and (b)
    (d) None of the above
Maximum marks: 80      Time allowed: 3 hours             ← paper metadata
Q2. State Henry's Law.                                   ← real question ✓
Note: Attempt all questions.                             ← instruction noise
General Instructions:                                    ← instruction block
1. All questions are compulsory.
2. Draw neat diagrams wherever necessary.
────────────────────────────────────────────────────────────────────────────
```

All of the non-question lines get indexed as PYQ chunks and appear in the sidebar, polluting the question list.

---

## Current preprocessing

`QueryPreprocessor` in `src/retrieval/intelligent_query.py` cleans the query **at retrieval time** (after indexing). It removes:
- Question number prefixes (`Q23.`, `1.`, `(i)`)
- Instruction phrases ("which of the following", "choose the correct")
- MCQ option labels `(a)`, `(b)`, `(c)`, `(d)`
- Retrieval stop-words

**This does not prevent garbage from being indexed.** The fix belongs at ingestion time, before chunks enter ChromaDB.

---

## Where to add filtering

### Option A — Filter during PDF parsing (preferred)

Add a classifier/filter in the PDF text extraction step before chunks are created.

**File to modify**: `src/pipeline/pdf_text.py` (PDF extraction) or whichever script calls it before calling `chunking.py`.

**Implementation approach**:

```python
# src/pipeline/pyq_filter.py  (new file)

import re
from typing import List

# Patterns that identify non-question content
NON_QUESTION_PATTERNS = [
    re.compile(r"^\s*(?:section|part)\s+[a-z]\b", re.IGNORECASE),        # "Section A"
    re.compile(r"^\s*general\s+instructions?\s*:?", re.IGNORECASE),       # "General Instructions:"
    re.compile(r"^\s*(?:maximum|total)\s+marks?\s*:", re.IGNORECASE),     # "Maximum Marks: 80"
    re.compile(r"^\s*time\s+(?:allowed|limit)\s*:", re.IGNORECASE),       # "Time Allowed:"
    re.compile(r"^\s*(?:note|important)\s*[:—]", re.IGNORECASE),          # "Note: Attempt all"
    re.compile(r"^\s*(?:all\s+questions?\s+(?:are\s+)?(?:compulsory|mandatory))", re.IGNORECASE),
    re.compile(r"^\s*(?:draw|answer\s+(?:in\s+)?(?:brief|detail))", re.IGNORECASE),
    re.compile(r"^\s*page\s+\d+\s*(?:of\s+\d+)?$", re.IGNORECASE),       # "Page 1 of 4"
    re.compile(r"^\s*\d+\s*$"),                                            # Lone page numbers
    re.compile(r"^\s*(?:cbse|ncert|board)\s+(?:examination|exam|paper)", re.IGNORECASE),
    re.compile(r"^\s*(?:set|code)\s*[:-]?\s*[a-z0-9]+$", re.IGNORECASE), # "Set: A" or "Code: 65/1"
]

# A real question must match at least one of these positive signals
QUESTION_SIGNALS = [
    re.compile(r"^\s*(?:Q\.?\s*)?\d+[\s.):-]", re.IGNORECASE),   # Q1. or 1. or 1)
    re.compile(r"\?"),                                              # has a question mark
    re.compile(r"(?:explain|define|state|describe|calculate|find|show|prove|write)", re.IGNORECASE),
    re.compile(r"\(\s*[a-dA-D1-4]\s*\)"),                         # MCQ options (a) (b)
    re.compile(r"^\s*(?:i+\.?\s+|ii+\.?\s+|iii+\.?\s+)"),        # (i) (ii) sub-questions
]

MIN_QUESTION_WORDS = 6   # anything shorter is almost certainly noise

def is_valid_question(text: str) -> bool:
    """Return True if text looks like an actual exam question."""
    stripped = text.strip()
    if not stripped:
        return False
    word_count = len(stripped.split())
    if word_count < MIN_QUESTION_WORDS:
        return False
    for pattern in NON_QUESTION_PATTERNS:
        if pattern.search(stripped):
            return False
    return any(p.search(stripped) for p in QUESTION_SIGNALS)


def filter_pyq_chunks(chunks: List[dict]) -> List[dict]:
    """
    Filter a list of raw PYQ text chunks, keeping only genuine questions.
    
    Each chunk dict is expected to have at minimum: {"text": str, ...}
    """
    return [c for c in chunks if is_valid_question(c.get("text", ""))]
```

**Where to call it**:
In the ingestion pipeline, after PDF-to-text extraction and before generating embeddings:
```python
from src.pipeline.pyq_filter import filter_pyq_chunks

raw_chunks = extract_chunks_from_pdf(pdf_path)
clean_chunks = filter_pyq_chunks(raw_chunks)   # ← add this line
embed_and_index(clean_chunks)
```

### Option B — Filter at ChromaDB query time (quick fix)

If re-ingesting is not feasible, filter the sidebar list before returning it to the frontend.

**File to modify**: `backend/retrieval.py`, method `_get_pyqs_from_chroma`

```python
# Inside _get_pyqs_from_chroma, after building `items`:
from src.pipeline.pyq_filter import is_valid_question
items = [i for i in items if is_valid_question(i["text"])]
```

This hides garbage from the UI without touching ChromaDB, but garbage embeddings still affect retrieval quality. Option A is the correct long-term fix.

---

## Heuristics tuning guide

| Heuristic | Where | Adjust when |
|---|---|---|
| `MIN_QUESTION_WORDS` | `pyq_filter.py` | Too many short questions filtered → lower to 4; too much noise passes → raise to 8 |
| `NON_QUESTION_PATTERNS` | `pyq_filter.py` | New PDF source has different boilerplate → add its pattern |
| `QUESTION_SIGNALS` | `pyq_filter.py` | Short-answer questions without "?" get dropped → add their pattern |

---

## Testing the filter

After implementing, verify with:
```python
from src.pipeline.pyq_filter import filter_pyq_chunks, is_valid_question

# Should return False (garbage)
assert not is_valid_question("General Instructions:")
assert not is_valid_question("Maximum Marks: 80   Time: 3 Hours")
assert not is_valid_question("Page 2 of 8")
assert not is_valid_question("Section B")

# Should return True (real questions)
assert is_valid_question("Q1. Which of the following is a redox reaction?")
assert is_valid_question("State Henry's Law and give one application.")
assert is_valid_question("Explain the role of enzymes in digestion.")
```

Run against a full PYQ file and compare `len(raw_chunks)` vs `len(clean_chunks)` to measure noise ratio.

---

## Metadata tagging requirement

Every PYQ chunk **must** be stored in ChromaDB with `metadata.source = "pyq"`. The retrieval engine filters NCERT chunks with `where={"source": "ncert"}` — if PYQ chunks lack this tag they will pollute NCERT search results.

Verify at ingest with:
```python
collection.get(where={"source": "pyq"}, limit=5, include=["metadatas"])
```
