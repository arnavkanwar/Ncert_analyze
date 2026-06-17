# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

NCERT Textbook Retrieval System — a local Python pipeline for semantic search over NCERT textbook markdown content. It ingests markdown files, chunks them at the paragraph level, generates dense embeddings, stores them in ChromaDB, and serves results via FastAPI and a Streamlit UI.

**Current status**: Phases 1–3 complete (Parse → Chunk → Embed → Index). Next: advanced reranking & API optimization.

## Commands

### Setup
```bash
pip install -r requirements.txt
# First-run model download (~420MB, auto-triggered):
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-mpnet-base-v2')"
```

### Run the Full Pipeline
```bash
python run_pipeline.py
```

### Ingestion (Phases 1–3: Parse, Clean, Chunk, Embed, Index)
```bash
python scripts/ingest_pipeline.py
python scripts/ingest_pipeline.py --clear-existing
python scripts/ingest_pipeline.py --data-dir path/to/data
```

### Query / Search
```bash
python scripts/query_system.py --interactive
python scripts/query_system.py --query "What is photosynthesis?"
python scripts/query_system.py --query "cell division" --class-filter class10 --subject-filter science --top-k 3
```

### Backend API (FastAPI)
```bash
cd backend && python -m uvicorn main:app --reload --port 8000
# Swagger UI: http://localhost:8000/docs
```

### Streamlit UI
```bash
streamlit run streamlit_app.py
```

### Tests
```bash
python -m pytest tests/
python tests/test_examples.py
```

### Embedding Generation (standalone)
```bash
python scripts/generate_embeddings.py \
  --input-json data/sample_phase3_input.json \
  --output-json output/embeddings/phase3_embeddings.json
```

## Architecture

### Data Flow

**Ingestion:**
```
.md files → Parse (heading hierarchy) → Clean (unicode/HTML/URLs) → Chunk (50–1500 chars)
         → Extract Metadata (class/subject/chapter) → Embed (768-dim) → Index (ChromaDB)
```

**Query:**
```
Query string → Embed → Vector Search (top 30) + BM25 (top 20) → Cross-encoder Rerank → Top 2 results
```

### Key Modules

| Layer | Location | Purpose |
|---|---|---|
| Parser | `src/parser/markdown_parser.py` | Extract H1/H2/H3 heading hierarchy from markdown |
| Cleaner | `src/processing/text_cleaner.py` | Unicode normalization, whitespace, URL/HTML removal |
| Chunker | `src/processing/chunker.py` | Paragraph-level splits with 50–1500 char constraints |
| Metadata | `src/processing/metadata_extractor.py` | Path-based class/subject/book/chapter extraction |
| Embedder | `src/embedding/embedder.py` | `all-mpnet-base-v2` → 768-dim vectors, batch + GPU support |
| VectorStore | `src/embedding/vector_store.py` | ChromaDB CRUD, cosine similarity search, metadata filters |
| Retriever | `src/retrieval/retriever.py` | High-level search interface with class/subject/chapter filters |
| Intelligent Query | `src/retrieval/intelligent_query.py` | Hybrid retrieval with reranking diagnostics & PYQ support |
| Pipeline | `src/pipeline/` | ChromaDB init, chunking orchestration, embedding pipeline |
| API | `backend/main.py` | FastAPI: `POST /query`, `GET /health`, CORS enabled |
| UI | `streamlit_app.py` | Interactive Streamlit search interface |

### Configuration

All tunable parameters live in [config/config.py](config/config.py). Key sections:

- **Embedding**: model (`all-mpnet-base-v2`), dimension (768), device (`cpu`/`cuda`)
- **Chunking**: strategy (`heading_aware`), min/max sizes (60/1500), overlap, heading prepend
- **Retrieval**: hybrid weights (rerank 0.55, vector 0.20, BM25 0.15, keyword 0.10), cross-encoder model
- **ChromaDB**: collection name, distance metric (`cosine`), DB directory

Backend environment config: [backend/.env](backend/.env) — sets `API_HOST`, `API_PORT`, `DEVICE`, `CHROMA_DB_DIR`, `COLLECTION_NAME`, `CORS_ORIGINS`.

### Chunk Schema

Each stored chunk carries:
```json
{
  "chunk_id": "class10_science_ch1_p003",
  "text": "...",
  "metadata": {
    "class_name": "class10", "subject": "science", "book": "book1", "chapter": "ch1",
    "heading_h1": "...", "heading_h2": "...", "heading_h3": "...",
    "paragraph_number": 3, "char_count": 223, "word_count": 42
  }
}
```

### Design Decisions

- **Heading-aware chunking**: each chunk prepends its heading path for embedding context, preserving educational hierarchy
- **Hybrid retrieval**: vector + BM25 + cross-encoder reranking trades latency (~350ms total) for significantly better precision
- **ChromaDB local**: no server needed, persistent on disk, supports metadata filtering natively
- **Phase independence**: each pipeline stage (ingest, embed, query) can run standalone; JSON outputs between phases enable debugging
