# Prompt 03 — UI Improvements

## Purpose
Use this prompt when modifying the React frontend. It describes the current UI structure, known usability gaps, and prioritized improvements.

---

## Current UI structure

Single-page app at `AIkaproject-main/frontend/src/pages/QueryPage.jsx`.

```
┌─────────────────────────────────────────────────────────────────────┐
│ SIDEBAR (collapsible)        │ CENTER PANEL        │ RIGHT PANEL    │
│ ─────────────────────────── │ ─────────────────── │ ───────────── │
│ "PYQ Questions"              │ Selected PYQ card   │ Bar chart      │
│ {N} questions                │ Rank-1 match card   │ Pie chart      │
│                              │ Rank-2 match card   │ Candidate pills│
│ [button] Q1 text...          │                     │                │
│ [button] Q2 text...          │                     │                │
│ [button] Q3 text... (active) │                     │                │
│  ...                         │                     │                │
└─────────────────────────────────────────────────────────────────────┘
```

**Key state variables** (`QueryPage.jsx`):
- `pyqs` — full list from `GET /pyqs`
- `selectedPyq` — currently active PYQ item
- `result` — `QueryByPyqResponse` from `POST /query/by-pyq`
- `loadingPyqs / loadingResult` — loading flags
- `sidebarOpen` — collapse toggle
- `expandedCandidates` — which candidate pills are expanded

**API calls** (`src/services/api.js`):
- `fetchPyqList(limit)` → `GET /pyqs`
- `queryByPyqId(pyqId)` → `POST /query/by-pyq`

---

## Prioritized improvements

### P1 — Sidebar search / filter

**Problem**: With 300+ questions, finding a specific one requires scrolling through the full list.

**Implementation**:
```jsx
// Add to QueryPage state:
const [searchQuery, setSearchQuery] = useState('');

// Filtered list (memo):
const filteredPyqs = useMemo(() =>
  pyqs.filter(q =>
    !searchQuery ||
    q.text.toLowerCase().includes(searchQuery.toLowerCase()) ||
    q.file_name.toLowerCase().includes(searchQuery.toLowerCase())
  ), [pyqs, searchQuery]);

// Add above the sidebar list:
<input
  className="sidebar-search"
  placeholder="Search questions…"
  value={searchQuery}
  onChange={e => setSearchQuery(e.target.value)}
/>

// Replace pyqs.map with filteredPyqs.map in the sidebar list render.
```

Add CSS:
```css
.sidebar-search {
  width: 100%;
  padding: 6px 10px;
  margin: 8px 0;
  background: #0d1117;
  border: 1px solid #30363d;
  border-radius: 6px;
  color: #e6edf3;
  font-size: 0.8125rem;
}
.sidebar-search:focus { outline: none; border-color: #818cf8; }
```

---

### P2 — Group sidebar by source file

**Problem**: Questions from different years/papers are mixed together without visual grouping.

**Implementation**:
```jsx
// Group by file_name using useMemo:
const groupedPyqs = useMemo(() => {
  const groups = {};
  filteredPyqs.forEach(q => {
    const key = q.file_name || 'Unknown';
    if (!groups[key]) groups[key] = [];
    groups[key].push(q);
  });
  return groups;
}, [filteredPyqs]);

// Render grouped:
{Object.entries(groupedPyqs).map(([fileName, items]) => (
  <div key={fileName} className="pyq-group">
    <p className="pyq-group-label">{fileName}</p>
    {items.map(item => (
      <button key={item.pyq_id} className={`pyq-item ${selectedPyq?.pyq_id === item.pyq_id ? 'active' : ''}`}
        onClick={() => handleSelectPyq(item)}>
        <p className="pyq-item-text">{item.text}</p>
      </button>
    ))}
  </div>
))}
```

Add CSS:
```css
.pyq-group-label {
  font-size: 0.7rem;
  font-weight: 700;
  color: #818cf8;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  padding: 10px 14px 4px;
}
```

---

### P3 — Expose score breakdown in match cards

**Problem**: Match cards show a single `score` but hide how it was computed (vector vs BM25 vs cross-encoder). Educators reviewing results need to understand why a paragraph was selected.

**Implementation**:

The `matches` array already contains `vector_score` and `score` (hybrid). Request the backend to also return `rerank_score` and `bm25_score` by extending `ParagraphMatch` in `backend/main.py`:

```python
class ParagraphMatch(BaseModel):
    rank: int
    chunk_id: str
    text: str
    score: float          # hybrid
    vector_score: float
    rerank_score: float   # add this
    bm25_score: float     # add this
    file_name: str
    paragraph_number: str | int
    reason: str
```

And populate from `backend/retrieval.py` `query_by_pyq_id_with_diagnostics`:
```python
"rerank_score": float(row.rerank_score),
"bm25_score": float(row.bm25_score),
```

Frontend — add to match card:
```jsx
<div className="score-breakdown">
  <span title="Cross-encoder">CE: {match.rerank_score?.toFixed(3)}</span>
  <span title="Vector similarity">Vec: {match.vector_score?.toFixed(3)}</span>
  <span title="BM25 keyword">BM25: {match.bm25_score?.toFixed(3)}</span>
</div>
```

---

### P4 — Empty state when no PYQs are indexed

**Problem**: If ChromaDB has no PYQs, the sidebar shows "0 questions" with no explanation.

**Implementation** — add to sidebar list:
```jsx
{!loadingPyqs && pyqs.length === 0 && (
  <div className="sidebar-empty">
    <p>No PYQs indexed yet.</p>
    <p>Run the ingestion pipeline to add questions.</p>
  </div>
)}
```

---

### P5 — Keyboard navigation

**Problem**: Power users can't navigate the question list without a mouse.

**Implementation**:
```jsx
// Track focused index
const [focusedIdx, setFocusedIdx] = useState(-1);

// Keyboard handler on the sidebar list container:
const handleSidebarKeyDown = (e) => {
  if (e.key === 'ArrowDown') setFocusedIdx(i => Math.min(i + 1, filteredPyqs.length - 1));
  if (e.key === 'ArrowUp') setFocusedIdx(i => Math.max(i - 1, 0));
  if (e.key === 'Enter' && focusedIdx >= 0) handleSelectPyq(filteredPyqs[focusedIdx]);
};
```

---

### P6 — Loading skeleton for match cards

**Problem**: The center panel is blank while `loadingResult` is true, creating layout shift.

**Implementation** — replace the `loadingResult` block:
```jsx
{loadingResult && (
  <div className="skeleton-list">
    {[1, 2].map(n => (
      <div key={n} className="skeleton-card">
        <div className="skeleton-line short" />
        <div className="skeleton-line" />
        <div className="skeleton-line" />
        <div className="skeleton-line medium" />
      </div>
    ))}
  </div>
)}
```

CSS:
```css
@keyframes shimmer { from { background-position: -400px 0; } to { background-position: 400px 0; } }
.skeleton-card { border-radius: 10px; border: 1px solid #21262d; padding: 16px; margin-bottom: 12px; }
.skeleton-line {
  height: 12px; border-radius: 6px; margin-bottom: 8px;
  background: linear-gradient(90deg, #161b22 25%, #21262d 50%, #161b22 75%);
  background-size: 400px 100%;
  animation: shimmer 1.4s infinite;
}
.skeleton-line.short { width: 40%; }
.skeleton-line.medium { width: 65%; }
```

---

## Files to modify

| Change | File |
|---|---|
| P1 search, P2 grouping, P4 empty state, P5 keyboard, P6 skeleton | `AIkaproject-main/frontend/src/pages/QueryPage.jsx` |
| P1/P2/P6 styles | `AIkaproject-main/frontend/src/styles/QueryPage.css` |
| P3 score breakdown (backend) | `backend/main.py`, `backend/retrieval.py` |
| P3 score breakdown (frontend) | `QueryPage.jsx` match card render |

## Dev server

```bash
cd AIkaproject-main/frontend
npm run dev          # Vite dev server on http://localhost:5173
```

Backend must be running separately on port 8000 (`VITE_API_BASE_URL` defaults to `http://127.0.0.1:8000`).
