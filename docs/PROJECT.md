# NCERT PYQ Retrieval System — Project Documentation

**Authoritative reference. Derived from codebase as of April 2026.**

---

## What this project does

Given a CBSE Previous Year Question (PYQ), the system finds the 1–2 most relevant NCERT textbook paragraphs that explain the answer. The pipeline uses hybrid semantic search (dense vectors + BM25 keyword matching) followed by cross-encoder reranking to produce ranked results with explainability scores.

---

## Running the project

### Prerequisites

- Python 3.8+
- Node.js 18+ and npm
- ~2 GB disk space (embedding model cache)

### 1. Install Python dependencies

```bash
# From project root
pip install -r requirements.txt
```

Do NOT install from `backend/requirements.txt` — it pins outdated package versions and exists only for legacy reference.

### 2. Download embedding model (first run only, ~420 MB)

```bash
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-mpnet-base-v2')"
```

### 3. Populate ChromaDB

Run the ingestion pipeline to parse, chunk, embed, and index content:

```bash
# Ingest NCERT markdown chapters
python scripts/ingest_pipeline.py

# Or clear existing data and re-ingest
python scripts/ingest_pipeline.py --clear-existing

# Ingest from a specific directory
python scripts/ingest_pipeline.py --data-dir data/NCERT/Class_12/Chemistry

# Process textbook PDFs
python scripts/process_books.py
```

### 4. Run the backend (FastAPI)

```bash
cd backend
../.venv/Scripts/python.exe -m uvicorn main:app --reload --port 8000
```

The server binds to `0.0.0.0:8000`. Required environment variables are read from `backend/.env` (already present in the repo).

### 5. Run the frontend (React + Vite)

```bash
cd AIkaproject-main/frontend
npm install        # first time only
npm run dev
```

Frontend runs at **http://localhost:5173**

The Vite dev server proxies `/api/*` to `http://127.0.0.1:8000`. Direct API calls go to `VITE_API_BASE_URL` (default `http://127.0.0.1:8000`) configured in `AIkaproject-main/frontend/.env.example`.

### 6. Optional: Streamlit UI (alternative interface)

```bash
# From project root
streamlit run streamlit_app.py
```

---

## API reference

All endpoints are served by `backend/main.py`. Swagger UI at **http://localhost:8000/docs**.

### GET /health

```bash
curl http://localhost:8000/health
```

```json
{ "status": "healthy", "retriever_ready": true }
```

### GET /pyqs

Returns the list of indexed PYQ questions for the sidebar.

```bash
curl "http://localhost:8000/pyqs?limit=300"
```

```json
{
  "items": [
    { "pyq_id": "pyq_001", "text": "Which of the following...", "file_name": "2023_chemistry.pdf", "paragraph_number": "1" }
  ],
  "count": 42
}
```

### POST /query/by-pyq

Primary endpoint. Given a `pyq_id` from `/pyqs`, returns ranked NCERT matches with score evidence.

```bash
curl -X POST http://localhost:8000/query/by-pyq \
  -H "Content-Type: application/json" \
  -d '{"pyq_id": "pyq_001"}'
```

```json
{
  "selected_pyq": { "pyq_id": "pyq_001", "text": "...", "file_name": "2023_chemistry.pdf", "paragraph_number": "1" },
  "matches": [
    {
      "rank": 1,
      "chunk_id": "ncert_class12_chemistry_ch3_p004",
      "text": "Ionic compounds have high melting points because...",
      "score": 0.7341,
      "vector_score": 0.6812,
      "file_name": "chapter03.md",
      "paragraph_number": 4,
      "reason": "Rank #1 by hybrid relevance (score 0.734); matches query terms: ionic, melting, point."
    }
  ],
  "chart": [
    { "label": "C1", "score": 0.7341, "chunk_id": "...", "file_name": "chapter03.md", "text": "..." }
  ],
  "count": 2
}
```

### POST /query

Accepts raw question text directly (used for programmatic access).

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is electronegativity?"}'
```

```json
{ "paragraphs": ["Paragraph 1 text...", "Paragraph 2 text..."], "count": 2 }
```

---

## Retrieval pipeline

The full pipeline lives in `src/retrieval/intelligent_query.py` (`IntelligentQueryEngine`), called from `backend/retrieval.py` (`NCERTRetriever`).

```
PYQ text
    │
    ▼
[1] QueryPreprocessor.preprocess()
    Strips question numbers, MCQ option labels (a)(b)(c)(d),
    instruction phrases ("which of the following"), stop-words.
    Produces a clean concept-focused string.
    │
    ├──────────────────────────┐
    ▼                          ▼
[2a] Vector search            [2b] BM25 keyword search
     ChromaDB cosine           Lightweight in-memory index
     filter: source=ncert      built at startup from all NCERT chunks
     top_k = 30                top_k = 20
    │                          │
    └────────────┬─────────────┘
                 ▼
[3] Merge candidates (~40 unique chunks)
                 │
                 ▼
[4] CrossEncoder.predict()   ← ms-marco-MiniLM-L-12-v2
    Pairwise [query, paragraph] scoring
                 │
                 ▼
[5] Hybrid fusion (all scores normalized to [0,1])
    score = 0.55×rerank + 0.20×vector + 0.15×bm25 + 0.10×keyword_overlap
                 │
                 ▼
[6] Relevance gate: best.score < 0.15 → return empty
                 │
                 ▼
[7] Second paragraph selection
    Must score ≥ 60% of best, Jaccard overlap < 70%,
    cosine similarity < 0.85, covers new query terms
                 │
                 ▼
    Return top 1–2 paragraphs
```

### Tuning constants (`IntelligentQueryEngine`)

| Constant | Default | Effect |
|---|---|---|
| `WEIGHT_RERANK` | 0.55 | Cross-encoder influence |
| `WEIGHT_VECTOR` | 0.20 | Semantic similarity influence |
| `WEIGHT_BM25` | 0.15 | Keyword match influence |
| `WEIGHT_KEYWORD` | 0.10 | Token overlap bonus |
| `MIN_HYBRID_SCORE` | 0.15 | Below this → no result returned |
| `SECOND_PARA_RATIO` | 0.60 | Second para must be ≥ 60% of first score |
| `REDUNDANCY_THRESHOLD` | 0.70 | Jaccard overlap cap for second para |
| `VECTOR_CANDIDATES` | 30 | Vector search pool size |
| `BM25_CANDIDATES` | 20 | BM25 pool size |

---

## Ingestion pipeline

**Data flow:**

```
PDF / .md file
    │
    ▼
src/pipeline/pdf_text.py         ← extract raw text from PDF
src/parser/markdown_parser.py    ← parse heading hierarchy (H1/H2/H3)
src/processing/text_cleaner.py   ← unicode normalization, URL/HTML removal
src/processing/chunker.py        ← paragraph split, 50–1500 chars
src/processing/metadata_extractor.py  ← class/subject/chapter from file path
src/embedding/embedder.py        ← all-mpnet-base-v2 → 768-dim vectors
src/pipeline/chroma_db.py        ← upsert into ChromaDB
output/chunks/all_chunks.json    ← JSON snapshot (source for BM25 at startup)
```

**ChromaDB metadata schema per chunk:**

```json
{
  "chunk_id": "ncert_class12_chemistry_ch3_p004",
  "source": "ncert",
  "file_name": "chapter03.md",
  "class_name": "class12",
  "subject": "chemistry",
  "book": "book1",
  "chapter": "ch3",
  "heading_h1": "Chemical Bonding",
  "heading_h2": "Ionic Bond",
  "heading_h3": "",
  "paragraph_number": 4,
  "char_count": 312,
  "word_count": 55
}
```

The `source` field is critical: `"ncert"` for textbook paragraphs, `"pyq"` for PYQ questions. The retrieval engine filters with `where={"source": "ncert"}`.

---

## PYQ filtering

Raw PYQ PDFs contain non-question content (section headers, general instructions, page numbers, time limits) that gets indexed alongside real questions, polluting the sidebar.

The filter logic lives in `src/pipeline/pyq_filter.py`. It runs at ingestion time, before chunks are embedded:

```python
from src.pipeline.pyq_filter import filter_pyq_chunks
clean_chunks = filter_pyq_chunks(raw_chunks)
```

A chunk passes if it:
- Has ≥ 6 words
- Does not match noise patterns (section headers, "General Instructions:", "Maximum Marks:", page numbers)
- Matches at least one question signal (question number prefix, `?`, explain/define/calculate verbs, MCQ option labels)

See `src/pipeline/pyq_filter.py` for the full pattern list and `tests/test_pyq_filter.py` to run tests against it.

---

## Frontend

Single-page React app. No routing. Entry: `AIkaproject-main/frontend/src/App.jsx` → `QueryPage.jsx`.

**Layout:**

```
┌──────────────────┬──────────────────────┬───────────────────┐
│ SIDEBAR          │ CENTER PANEL         │ RIGHT PANEL       │
│                  │                      │                   │
│ PYQ list         │ Selected PYQ card    │ Bar chart         │
│ (collapsible)    │ Match card rank 1    │ (hybrid scores)   │
│                  │ Match card rank 2    │ Pie chart         │
│ [Q1 text...]     │                      │ (distribution)    │
│ [Q2 text...]     │                      │ Candidate pills   │
│ [Q3 text...] ←  │                      │ (collapsible)     │
└──────────────────┴──────────────────────┴───────────────────┘
```

**Key files:**

| File | Purpose |
|---|---|
| `src/pages/QueryPage.jsx` | All UI state and rendering |
| `src/services/api.js` | `fetchPyqList()`, `queryByPyqId()`, `queryNCERT()` |
| `src/styles/QueryPage.css` | Page-specific styles |
| `src/components/Card.jsx` | Generic card component |
| `vite.config.js` | Dev server port 5173, proxy `/api` → 8000 |

**Frontend → Backend calls:**

```
page load  →  GET /pyqs?limit=500           →  populate sidebar
click PYQ  →  POST /query/by-pyq            →  show matches + charts
```

**Environment:**

Copy `AIkaproject-main/frontend/.env.example` to `.env.local` and set:
```
VITE_API_BASE_URL=http://127.0.0.1:8000
```

---

## Configuration

Central config: `config/config.py`

Backend environment: `backend/.env`

| Variable | Default | Description |
|---|---|---|
| `API_HOST` | `0.0.0.0` | FastAPI bind address |
| `API_PORT` | `8000` | FastAPI port |
| `DEVICE` | `cpu` | `cpu` or `cuda` for embedding model |
| `CHROMA_DB_DIR` | `chroma_db` | ChromaDB persistence directory |
| `COLLECTION_NAME` | `ncert_chemistry` | ChromaDB collection name |
| `CORS_ORIGINS` | `http://localhost:5173,...` | Allowed origins |

---

## Tests

```bash
# Run all tests
python -m pytest tests/

# PYQ filter unit tests
python -m pytest tests/test_pyq_filter.py -v

# Legacy integration test
python tests/test_examples.py
```

---

## Key module index

| Layer | File | Class / Entry point |
|---|---|---|
| Retrieval engine | `src/retrieval/intelligent_query.py` | `IntelligentQueryEngine` |
| Query preprocessor | `src/retrieval/intelligent_query.py` | `QueryPreprocessor` |
| BM25 index | `src/retrieval/intelligent_query.py` | `BM25` |
| ChromaDB wrapper | `src/pipeline/chroma_db.py` | `ChromaIndexer` |
| Embedder | `src/embedding/embedder.py` | `TextEmbedder` |
| PYQ filter | `src/pipeline/pyq_filter.py` | `filter_pyq_chunks`, `is_valid_question` |
| FastAPI app | `backend/main.py` | `app` |
| Retrieval wrapper | `backend/retrieval.py` | `NCERTRetriever` |
| Ingestion | `scripts/ingest_pipeline.py` | CLI entrypoint |
| Book processing | `scripts/process_books.py` | CLI entrypoint |

---

## Troubleshooting

| Problem | Cause | Fix |
|---|---|---|
| `503 Retriever service not initialized` | ChromaDB is empty or missing | Run `python scripts/process_books.py` then restart backend |
| Sidebar shows garbage (headers, instructions) | PYQ filter not applied at ingest time | Re-ingest with `filter_pyq_chunks()` applied; see `src/pipeline/pyq_filter.py` |
| No match returned for a question | Score below `MIN_HYBRID_SCORE=0.15` | Check `debug.top_cross_encoder_scores` in `query_from_pyq()` output; also check `debug.preprocessed_query` to see what the engine actually searched |
| Wrong paragraphs returned | Embedding noise from MCQ boilerplate | Verify `QueryPreprocessor.preprocess()` is cleaning the question; increase `WEIGHT_BM25` if exact science terms are missed |
| `Address already in use` on port 8000 | Another process holds the port | `lsof -i :8000` then `kill <PID>`, or use `--port 8001` |
| Model download fails | No internet / firewall | Manually run: `python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-mpnet-base-v2')"` |
| ChromaDB collection empty after ingest | Wrong `COLLECTION_NAME` | Ensure `backend/.env` and `config/config.py` use the same collection name (`ncert_chemistry`) |
