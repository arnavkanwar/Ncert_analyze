# Prompt 01 — Semantic Search Pipeline

## Purpose
Use this prompt when modifying the retrieval pipeline, tuning scores, or debugging why a PYQ returns wrong NCERT paragraphs.

---

## System overview

Given a PYQ question, the system finds the 1–2 most relevant NCERT paragraphs. The pipeline runs in `src/retrieval/intelligent_query.py` (`IntelligentQueryEngine`) and is called from `backend/retrieval.py` (`NCERTRetriever`).

### Step-by-step pipeline

```
PYQ text (raw)
    │
    ▼
[1] QueryPreprocessor.preprocess()
    • Strips "Q23." prefixes, "which of the following", option labels (a)(b)(c)(d)
    • Produces a clean concept-focused string
    • e.g. "ionic compound properties: high melting point, conductivity, solubility"
    │
    ▼
[2] SentenceTransformer.encode()          ← model: all-mpnet-base-v2 (768-dim)
    • Encodes cleaned query to embedding
    │
    ├──────────────────────────────────────────────────────────────┐
    ▼                                                              ▼
[3a] Vector search (ChromaDB)                          [3b] BM25 sparse search
     • Filter: source == "ncert"                            • Built at startup from all NCERT chunks
     • top_k = 30 candidates                                • top_k = 20 candidates
     • Returns cosine distances                              • Returns term-frequency scores
    │                                                              │
    └──────────────────────────┬───────────────────────────────────┘
                               ▼
[4] Merge & deduplicate candidates
    • BM25-only results (missed by vector) appended (up to 10)
    • Combined pool: up to ~40 unique candidates
                               │
                               ▼
[5] CrossEncoder.predict()                ← model: ms-marco-MiniLM-L-12-v2
    • Pairwise scoring: [query, paragraph] → float
    • Run over all ~40 candidates
                               │
                               ▼
[6] Hybrid fusion score (all normalized to [0,1])
    hybrid = 0.55 × rerank + 0.20 × vector + 0.15 × bm25 + 0.10 × keyword_overlap
                               │
                               ▼
[7] Relevance threshold gate
    • If best.hybrid_score < 0.15  →  return empty (no garbage results)
                               │
                               ▼
[8] Second paragraph selection
    • Candidate must score ≥ 60% of best score
    • Must NOT be redundant (Jaccard overlap < 70%)
    • Embedding cosine similarity < 0.85
    • Must cover at least one additional query term not in paragraph #1
                               │
                               ▼
    Return: best paragraph + (optional) second paragraph
```

---

## Key files and their roles

| File | Role |
|---|---|
| `src/retrieval/intelligent_query.py` | Full pipeline: BM25, QueryPreprocessor, IntelligentQueryEngine |
| `backend/retrieval.py` | NCERTRetriever — thin wrapper called by FastAPI |
| `backend/main.py` | FastAPI endpoints: `GET /pyqs`, `POST /query`, `POST /query/by-pyq` |
| `src/pipeline/chroma_db.py` | ChromaIndexer — wraps ChromaDB collection |
| `config/config.py` | All numeric thresholds and model names |

---

## Tuning guidance for Haiku

### Weights (IntelligentQueryEngine class constants)
```python
WEIGHT_RERANK   = 0.55   # cross-encoder — most accurate signal
WEIGHT_VECTOR   = 0.20   # semantic similarity
WEIGHT_BM25     = 0.15   # exact keyword match (critical for science terms)
WEIGHT_KEYWORD  = 0.10   # simple token overlap bonus
```
Increase `WEIGHT_BM25` if science-term exact matches are being missed.
Increase `WEIGHT_RERANK` if semantically wrong results are appearing.

### Thresholds
```python
MIN_HYBRID_SCORE   = 0.15   # raise to 0.20 to be more selective
SECOND_PARA_RATIO  = 0.60   # raise to 0.75 to require closer second match
REDUNDANCY_THRESHOLD = 0.70  # lower to 0.50 to avoid near-duplicate second paragraphs
```

### Candidate pool sizes
```python
VECTOR_CANDIDATES = 30
BM25_CANDIDATES   = 20
```
Increase if top result quality is low. Decreasing speeds up reranking.

---

## Common failure modes and fixes

| Symptom | Likely cause | Fix |
|---|---|---|
| Returns paragraphs unrelated to topic | `QueryPreprocessor` strips too aggressively; raw query noise dominates embedding | Check preprocessed query in `debug.preprocessed_query` in API response; add domain terms to STOP_WORDS if needed |
| Missing exact scientific term match | BM25 not built or weight too low | Verify BM25 index built at startup (log: "BM25 index built with N documents"); increase `WEIGHT_BM25` |
| Second paragraph is near-duplicate | Redundancy threshold too loose | Lower `REDUNDANCY_THRESHOLD` from 0.70 to 0.50 |
| No results returned | Score below `MIN_HYBRID_SCORE` | Check `debug.top_cross_encoder_scores` in API response for raw scores; lower threshold or fix preprocessing |
| Wrong collection queried | `filter_metadata={"source": "ncert"}` not applied | Verify ChromaIndexer.search passes the where-filter; PYQ chunks must be tagged `source=pyq` at ingest time |

---

## API response structure (for reference)

`POST /query/by-pyq` returns:
```json
{
  "selected_pyq": { "pyq_id": "...", "text": "...", "file_name": "...", "paragraph_number": "..." },
  "matches": [
    { "rank": 1, "chunk_id": "...", "text": "...", "score": 0.73, "vector_score": 0.61,
      "file_name": "ch3_chemistry.md", "paragraph_number": 4, "reason": "..." }
  ],
  "chart": [ { "label": "C1", "score": 0.73, "chunk_id": "...", "file_name": "...", "text": "..." } ],
  "count": 2
}
```

The `debug` object (available in `query_from_pyq` output, not exposed to frontend) contains `preprocessed_query` and `top_cross_encoder_scores` — useful for diagnosing retrieval failures.
