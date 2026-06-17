# NCERT PYQ Retrieval System

Semantic search over NCERT textbook paragraphs, driven by CBSE Previous Year Questions. Hybrid BM25 + vector retrieval with cross-encoder reranking.

## Run

**Backend** (FastAPI, port 8000):
```bash
pip install -r requirements.txt
cd backend && python -m uvicorn main:app --reload --port 8000
```

**Frontend** (React + Vite, port 5173):
```bash
cd AIkaproject-main/frontend
npm install && npm run dev
```

Open **http://localhost:5173** — select a PYQ from the sidebar to retrieve ranked NCERT paragraphs.

## Ingest content

```bash
# NCERT markdown chapters
python scripts/ingest_pipeline.py

# Textbook PDFs
python scripts/process_books.py
```

## Documentation

See [docs/PROJECT.md](docs/PROJECT.md) for full architecture, API reference, configuration, and troubleshooting.
