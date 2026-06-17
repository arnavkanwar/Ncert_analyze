# Prompt 04 — Upload Management for PYQs and Chapter PDFs

## Purpose
Use this prompt when adding new PYQ papers or NCERT chapter PDFs to the system, or when building a repeatable ingestion workflow.

---

## Current data flow

```
PDF / Markdown file
    │
    ▼
src/pipeline/pdf_text.py          ← extract raw text from PDF
    │
    ▼
src/processing/text_cleaner.py    ← normalize unicode, strip HTML/URLs
    │
    ▼
src/processing/chunker.py         ← paragraph-level split (50–1500 chars)
    │
    ▼
src/processing/metadata_extractor.py  ← attach class/subject/chapter/heading from file path
    │
    ▼
src/pipeline/pyq_filter.py (planned)  ← filter non-question chunks (see prompt 02)
    │
    ▼
src/embedding/embedder.py         ← encode each chunk to 768-dim vector
    │
    ▼
src/pipeline/chroma_db.py         ← upsert into ChromaDB with metadata
    │
    ▼
output/chunks/all_chunks.json     ← append to JSON snapshot (BM25 source at startup)
```

---

## Metadata tagging rules

The `source` metadata field controls which chunks go where:

| `source` value | What it means | Queried by |
|---|---|---|
| `"ncert"` | NCERT chapter paragraph | Vector search filter in retrieval engine |
| `"pyq"` | PYQ question | Sidebar (`GET /pyqs`) |

**Every chunk must have `source` set correctly before calling ChromaDB upsert.** If missing, PYQ chunks will appear in NCERT search results and vice versa.

Additional required metadata fields per chunk:

```python
{
    "chunk_id":          str,   # unique, e.g. "pyq_2023_chemistry_q14"
    "source":            str,   # "pyq" or "ncert"
    "file_name":         str,   # original PDF/md filename, e.g. "2023_chemistry_pyq.pdf"
    "class_name":        str,   # "class10", "class12", etc.
    "subject":           str,   # "chemistry", "biology", "physics"
    "question_number":   int,   # for PYQs: question number extracted from PDF
    "year":              str,   # for PYQs: "2023", "2022", etc. (if available)
    "chapter":           str,   # for NCERT: "ch1", "ch3", etc.
    "heading_h1":        str,   # for NCERT: chapter heading
    "paragraph_number":  int,   # sequential paragraph within document
    "char_count":        int,
    "word_count":        int,
}
```

---

## Adding a new PYQ paper

### Step 1 — Place the file

```
data/pyqs/
  class12_chemistry_2023.pdf
  class12_chemistry_2024.pdf   ← new file here
```

### Step 2 — Run ingestion (single file)

```bash
python scripts/ingest_pipeline.py \
  --data-dir data/pyqs/class12_chemistry_2024.pdf \
  --source pyq \
  --class class12 \
  --subject chemistry \
  --year 2024
```

If the script does not yet accept `--source` and `--year` flags, add them or set them manually in the ingestion script before calling the embedder.

### Step 3 — Verify in ChromaDB

```python
from src.pipeline.chroma_db import ChromaIndexer
indexer = ChromaIndexer(persist_dir="chroma_db", collection_name="ncert_chemistry")
result = indexer.collection.get(
    where={"source": "pyq", "file_name": "class12_chemistry_2024.pdf"},
    limit=5,
    include=["documents", "metadatas"]
)
print(result["ids"])         # should list chunk IDs
print(result["documents"])   # should show question text, not headers
```

### Step 4 — Rebuild BM25 index

The BM25 index in `IntelligentQueryEngine` is built at server startup from ChromaDB. Restart the FastAPI server to pick up new NCERT chunks:

```bash
# Restart backend
cd backend && python -m uvicorn main:app --reload --port 8000
```

---

## Adding a new NCERT chapter

### Step 1 — Place the markdown file

Follow the path convention (used by `MetadataExtractor` to infer class/subject/chapter):
```
data/NCERT/Class_12/Chemistry/Book_1/Chapter_05/chapter.md
```

### Step 2 — Run ingestion

```bash
python scripts/ingest_pipeline.py \
  --data-dir data/NCERT/Class_12/Chemistry/Book_1/Chapter_05 \
  --source ncert
```

Or run the full pipeline to re-process everything:
```bash
python run_pipeline.py
```

### Step 3 — Verify

```python
result = indexer.collection.get(
    where={"source": "ncert", "chapter": "ch5", "subject": "chemistry"},
    limit=5,
    include=["documents", "metadatas"]
)
```

---

## Idempotency — re-ingesting without duplicates

ChromaDB `upsert` (not `add`) is idempotent: re-running ingestion for the same file will overwrite existing chunks with the same `chunk_id`. Always use `collection.upsert()` not `collection.add()` in `src/pipeline/chroma_db.py`.

Verify the current behavior:
```bash
grep -n "\.add\|\.upsert" src/pipeline/chroma_db.py
```
If you see `.add(`, change it to `.upsert(` to make ingestion safe to re-run.

---

## Batch ingestion for multiple files

For processing an entire folder of new PYQ PDFs:

```bash
# Process all PDFs in a directory
for pdf in data/pyqs/*.pdf; do
  python scripts/ingest_pipeline.py --data-dir "$pdf" --source pyq
done
```

Or add a batch script:
```python
# scripts/batch_ingest_pyqs.py
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.ingest_pipeline import run_ingestion

pyq_dir = Path("data/pyqs")
for pdf_path in sorted(pyq_dir.glob("*.pdf")):
    print(f"Ingesting: {pdf_path.name}")
    run_ingestion(data_path=pdf_path, source="pyq")
```

---

## Chunk ID naming convention

Ensure chunk IDs are unique and stable (same file + paragraph → same ID):

```python
# PYQ chunk
chunk_id = f"pyq_{year}_{subject}_{file_stem}_q{question_number:03d}"
# e.g. "pyq_2024_chemistry_class12_q014"

# NCERT chunk
chunk_id = f"ncert_{class_name}_{subject}_{chapter}_p{paragraph_number:04d}"
# e.g. "ncert_class12_chemistry_ch5_p0023"
```

Using zero-padded integers ensures lexicographic sort matches numeric sort in the sidebar.

---

## Checking overall collection stats

```python
from src.pipeline.chroma_db import ChromaIndexer

indexer = ChromaIndexer(persist_dir="chroma_db", collection_name="ncert_chemistry")
total = indexer.collection.count()
ncert = indexer.collection.get(where={"source": "ncert"}, limit=1)
pyq   = indexer.collection.get(where={"source": "pyq"},   limit=1)

print(f"Total chunks: {total}")
# Count NCERT vs PYQ by checking metadata - use a loop or count query
```

Expected ratio for a healthy collection: many more NCERT chunks than PYQ chunks (NCERT provides the answer corpus; PYQs are the queries).

---

## Future: upload API endpoint

To let users upload files from the UI without SSH access, add a FastAPI endpoint:

```python
# backend/main.py
from fastapi import UploadFile, File, Form

@app.post("/upload/pyq", tags=["Upload"])
async def upload_pyq(
    file: UploadFile = File(...),
    year: str = Form(...),
    subject: str = Form(...),
    class_name: str = Form(...)
):
    """Accept a PYQ PDF, save it, run ingestion, return chunk count."""
    save_path = BASE_DIR / "data" / "pyqs" / file.filename
    save_path.write_bytes(await file.read())
    # trigger ingestion (run in background thread to avoid blocking)
    import asyncio
    loop = asyncio.get_event_loop()
    count = await loop.run_in_executor(None, run_ingestion, save_path, "pyq", year, subject, class_name)
    return {"status": "indexed", "chunks": count, "file": file.filename}
```

The frontend would call this with a `<input type="file">` form. Keep file size limits and validate MIME type (`application/pdf`) before saving.
