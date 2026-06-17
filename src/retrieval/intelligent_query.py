"""PYQ-to-NCERT retrieval engine with hybrid BM25+vector search, 
advanced query preprocessing, and cross-encoder reranking.

IMPROVED VERSION — Key changes:
1. BM25 sparse retrieval fused with vector search (hybrid)
2. Much better PYQ query preprocessing (concept extraction, noise removal)
3. Larger candidate pool (30-50) for cross-encoder reranking  
4. Stronger cross-encoder model (L-12 instead of L-6)
5. Relevance threshold to avoid returning garbage results
6. Better second-paragraph selection logic
"""

from __future__ import annotations

import json
import logging
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import numpy as np
from sentence_transformers import CrossEncoder, SentenceTransformer

from src.pipeline.chroma_db import ChromaIndexer

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────
# BM25 implementation (lightweight, no external dependency)
# ──────────────────────────────────────────────────────────────────

class BM25:
    """Lightweight BM25 index for sparse keyword retrieval.
    
    WHY BM25 IS CRITICAL:
    Vector search excels at semantic similarity but MISSES exact scientific 
    terms. For example, a PYQ asking about "electronegativity" should match 
    chunks containing that exact word — BM25 handles this perfectly.
    
    Vector search would also match "electron affinity" or "polarity" which 
    are related but NOT what was asked.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_count = 0
        self.avg_dl = 0.0
        self.doc_lengths: List[int] = []
        self.tf: List[Dict[str, int]] = []  # term freq per doc
        self.df: Dict[str, int] = {}        # document freq per term
        self.doc_ids: List[str] = []

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Simple whitespace + alphanumeric tokenizer."""
        return [t.lower() for t in re.findall(r"[a-zA-Z0-9]+", text) if len(t) > 1]

    def index(self, documents: List[Dict[str, str]]) -> None:
        """Build BM25 index from list of {id, text} dicts."""
        self.doc_count = len(documents)
        total_length = 0

        for doc in documents:
            doc_id = doc["id"]
            tokens = self._tokenize(doc["text"])
            doc_len = len(tokens)
            total_length += doc_len
            self.doc_lengths.append(doc_len)
            self.doc_ids.append(doc_id)

            tf = Counter(tokens)
            self.tf.append(dict(tf))

            for term in set(tokens):
                self.df[term] = self.df.get(term, 0) + 1

        self.avg_dl = total_length / max(1, self.doc_count)

    def score(self, query: str, top_k: int = 50) -> List[Dict[str, float]]:
        """Score all documents against query, return top-k."""
        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        scores: List[float] = []
        for idx in range(self.doc_count):
            s = 0.0
            dl = self.doc_lengths[idx]
            for term in query_tokens:
                if term not in self.tf[idx]:
                    continue
                tf_val = self.tf[idx][term]
                df_val = self.df.get(term, 0)
                idf = math.log((self.doc_count - df_val + 0.5) / (df_val + 0.5) + 1.0)
                numerator = tf_val * (self.k1 + 1)
                denominator = tf_val + self.k1 * (1 - self.b + self.b * dl / self.avg_dl)
                s += idf * numerator / denominator
            scores.append(s)

        # Get top-k indices
        indexed_scores = [(i, s) for i, s in enumerate(scores) if s > 0]
        indexed_scores.sort(key=lambda x: x[1], reverse=True)

        results = []
        for i, s in indexed_scores[:top_k]:
            results.append({"id": self.doc_ids[i], "score": s})
        return results


# ──────────────────────────────────────────────────────────────────
# Data classes
# ──────────────────────────────────────────────────────────────────

@dataclass
class RankedChunk:
    """Internal ranked NCERT candidate for selected PYQ."""
    chunk_id: str
    text: str
    text_raw: str           # Original text without heading prefix (for display)
    metadata: Dict[str, Any]
    vector: List[float]
    vector_score: float
    rerank_score: float
    bm25_score: float = 0.0
    keyword_score: float = 0.0
    hybrid_score: float = 0.0


# ──────────────────────────────────────────────────────────────────
# Query preprocessing
# ──────────────────────────────────────────────────────────────────

class QueryPreprocessor:
    """Transform PYQ text into an optimal retrieval query.
    
    WHY THIS MATTERS:
    Raw PYQ: "Q23. Which of the following is NOT a property of ionic compounds?
              (a) High melting point (b) Conductivity in solution 
              (c) Low boiling point (d) Solubility in water"
    
    This has TONS of noise: question number, "which of the following", 
    "NOT", option labels. The embedding model wastes capacity on these.
    
    Cleaned: "properties of ionic compounds: high melting point, 
              conductivity in solution, low boiling point, solubility in water"
    
    This focuses the embedding on the actual NCERT concepts.
    """

    # Question noise patterns to remove
    QUESTION_PREFIX = re.compile(
        r"^\s*(?:Q(?:uestion)?\.?\s*)?\d+\s*[.):—-]\s*",
        re.IGNORECASE
    )

    # Instruction phrases that add no retrieval value
    INSTRUCTION_NOISE = [
        r"which\s+(?:of\s+the\s+following|one\s+(?:of\s+the\s+following)?)",
        r"choose\s+the\s+(?:correct|incorrect|right|wrong)\s+(?:option|answer|statement)",
        r"select\s+the\s+(?:correct|most\s+appropriate)",
        r"(?:mark|identify|pick)\s+the\s+(?:correct|right|wrong)",
        r"the\s+correct\s+(?:option|answer|statement)\s+is",
        r"(?:is|are)\s+(?:correct|incorrect|true|false)\s*[?:.]",
        r"(?:given|consider)\s+the\s+following\s+statements?",
        r"(?:read|consider)\s+the\s+(?:passage|statements?)\s+(?:below|given)",
        r"assertion.*reason",
        r"match\s+the\s+following",
        r"from\s+the\s+options?\s+given\s+below",
    ]
    INSTRUCTION_PATTERN = re.compile(
        "|".join(INSTRUCTION_NOISE), re.IGNORECASE
    )

    # Option label patterns: (a), (A), A), a., 1., (i), (ii)
    OPTION_SPLIT = re.compile(
        r"\s*(?:\([a-dA-D1-4ivx]+\)|[a-dA-D][).]\s|[1-4][).]\s)",
    )

    # Words with zero retrieval value
    STOP_WORDS = {
        "which", "what", "when", "where", "how", "why", "none", "both", "all",
        "following", "correct", "incorrect", "statement", "statements", "option",
        "options", "choose", "mark", "true", "false", "most", "least", "among",
        "from", "with", "that", "this", "these", "those", "there", "their",
        "each", "every", "answer", "question", "given", "below", "above",
        "select", "identify", "pick", "not", "are", "is", "was", "were",
        "the", "and", "for", "into", "has", "have", "had", "been",
        "respectively", "only", "write", "explain", "define", "state",
        "describe", "mention", "list", "give", "name",
    }

    @classmethod
    def preprocess(cls, raw_pyq: str) -> str:
        """Transform raw PYQ text into an optimal retrieval query.
        
        Steps:
        1. Remove question number prefix
        2. Strip instruction phrases
        3. Extract stem and option concepts
        4. Build a clean concept-focused query
        """
        if not raw_pyq or not raw_pyq.strip():
            return ""

        text = " ".join(raw_pyq.replace("\r", " ").replace("\n", " ").split())

        # Step 1: Remove question number prefix
        text = cls.QUESTION_PREFIX.sub("", text).strip()

        # Step 2: Split into stem and options
        option_parts = cls.OPTION_SPLIT.split(text)
        
        if len(option_parts) > 1:
            stem = option_parts[0].strip()
            options = [p.strip() for p in option_parts[1:] if p.strip()]
        else:
            stem = text
            options = []

        # Step 3: Clean the stem of instruction noise
        stem = cls.INSTRUCTION_PATTERN.sub("", stem).strip()
        stem = re.sub(r"\s+", " ", stem).strip()
        # Remove trailing punctuation noise
        stem = re.sub(r"[?:]+\s*$", "", stem).strip()

        # Step 4: Extract key concept terms from options
        option_concepts = []
        for opt in options:
            # Remove "Both A and B", "None of the above" type options
            if re.match(r"(?:both|none|all)\s+(?:of\s+)?(?:the\s+)?(?:above|these|a|b|c|d)", opt, re.IGNORECASE):
                continue
            # Extract meaningful terms
            terms = [t for t in re.findall(r"[a-zA-Z0-9]+", opt.lower()) if t not in cls.STOP_WORDS and len(t) > 2]
            option_concepts.extend(terms)

        # Step 5: Build the final query
        # Deduplicate option concepts while preserving order
        seen = set()
        unique_concepts = []
        for c in option_concepts:
            if c not in seen:
                seen.add(c)
                unique_concepts.append(c)

        if unique_concepts:
            concept_hint = ", ".join(unique_concepts[:15])
            query = f"{stem}. Key concepts: {concept_hint}"
        else:
            query = stem

        return query.strip()

    @classmethod
    def extract_key_terms(cls, text: str) -> Set[str]:
        """Extract important scientific/concept terms from text."""
        tokens = set(re.findall(r"[a-zA-Z0-9]+", text.lower()))
        return {t for t in tokens if t not in cls.STOP_WORDS and len(t) > 2}


# ──────────────────────────────────────────────────────────────────
# Main query engine
# ──────────────────────────────────────────────────────────────────

class IntelligentQueryEngine:
    """Retrieve NCERT paragraphs that best explain a selected CBSE PYQ.
    
    Key improvements:
    1. Hybrid retrieval: BM25 (keyword) + Vector (semantic) search
    2. Advanced PYQ preprocessing with concept extraction
    3. Larger candidate pool (30) for cross-encoder reranking
    4. Better cross-encoder model
    5. Relevance threshold gating (won't return garbage)
    6. Better second-paragraph selection
    """

    # Cross-encoder: L-12 is significantly better than L-6
    # Still fast enough for real-time use on CPU
    DEFAULT_RERANKER = "cross-encoder/ms-marco-MiniLM-L-12-v2"

    # Hybrid fusion weights
    WEIGHT_RERANK = 0.55      # Cross-encoder dominates final ranking
    WEIGHT_VECTOR = 0.20      # Semantic similarity
    WEIGHT_BM25 = 0.15        # Keyword matching (critical for science terms)
    WEIGHT_KEYWORD = 0.10     # Simple overlap bonus

    # Relevance thresholds
    MIN_HYBRID_SCORE = 0.15           # Below this = garbage, don't return
    SECOND_PARA_RATIO = 0.60          # 2nd para must be >= 60% of 1st para score
    REDUNDANCY_THRESHOLD = 0.70       # Jaccard overlap > 70% = redundant

    # Candidate pool size for reranking
    VECTOR_CANDIDATES = 30            # Get 30 from vector search
    BM25_CANDIDATES = 20              # Get 20 from BM25
    RERANK_POOL = 40                  # Rerank top 40 unique candidates

    def __init__(
        self,
        chroma_dir: Path,
        collection_name: str = "ncert_chemistry",
        embedding_model: str = "sentence-transformers/all-mpnet-base-v2",
        reranker_model: str = None,
        chunks_path: Path = Path("output/chunks/all_chunks.json"),
        device: str = "cpu",
    ) -> None:
        self.embedder = SentenceTransformer(embedding_model, device=device)
        self.reranker = CrossEncoder(reranker_model or self.DEFAULT_RERANKER, device=device)
        self.indexer = ChromaIndexer(persist_dir=chroma_dir, collection_name=collection_name)
        self.chunks_path = Path(chunks_path)
        self.preprocessor = QueryPreprocessor()

        # Build BM25 index from all NCERT chunks in ChromaDB
        self._bm25: Optional[BM25] = None
        self._bm25_doc_map: Dict[str, Dict] = {}
        self._build_bm25_index()

    def _build_bm25_index(self) -> None:
        """Build BM25 sparse index from NCERT documents in ChromaDB.
        
        WHY: BM25 is built at startup from all NCERT chunks. This enables
        exact keyword matching alongside vector similarity search.
        """
        try:
            # Get all NCERT documents from ChromaDB
            result = self.indexer.collection.get(
                where={"source": "ncert"},
                include=["documents", "metadatas"],
                limit=10000,  # Adjust if you have more chunks
            )

            ids = result.get("ids", [])
            docs = result.get("documents", [])
            metas = result.get("metadatas", [])

            if not ids:
                logger.warning("No NCERT documents found in ChromaDB for BM25 index")
                return

            bm25_docs = []
            for i, doc_id in enumerate(ids):
                text = docs[i] if i < len(docs) else ""
                meta = metas[i] if i < len(metas) else {}
                bm25_docs.append({"id": doc_id, "text": text})
                self._bm25_doc_map[doc_id] = {
                    "text": text,
                    "metadata": meta,
                }

            self._bm25 = BM25()
            self._bm25.index(bm25_docs)
            logger.info("BM25 index built with %d NCERT documents", len(bm25_docs))

        except Exception as e:
            logger.warning("Failed to build BM25 index: %s", e)
            self._bm25 = None

    # ------------------------------------------------------------------
    # Utility methods
    # ------------------------------------------------------------------

    @staticmethod
    def _cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
        arr1 = np.array(vec1, dtype=float)
        arr2 = np.array(vec2, dtype=float)
        denom = np.linalg.norm(arr1) * np.linalg.norm(arr2)
        if math.isclose(float(denom), 0.0):
            return 0.0
        return float(np.dot(arr1, arr2) / denom)

    @staticmethod
    def _token_set(text: str) -> Set[str]:
        return {tok for tok in re.findall(r"[a-zA-Z0-9]+", text.lower()) if len(tok) > 2}

    @staticmethod
    def _minmax_normalize(values: List[float]) -> List[float]:
        if not values:
            return []
        minimum = min(values)
        maximum = max(values)
        if math.isclose(minimum, maximum):
            return [1.0 for _ in values]
        return [(v - minimum) / (maximum - minimum) for v in values]

    @staticmethod
    def _sigmoid(score: float) -> float:
        return 1.0 / (1.0 + math.exp(-score))

    # ------------------------------------------------------------------
    # PYQ list management
    # ------------------------------------------------------------------

    def get_pyq_list(self, limit: int = 250) -> List[Dict[str, str]]:
        if not self.chunks_path.exists():
            return []

        rows = json.loads(self.chunks_path.read_text(encoding="utf-8"))
        pyqs = [row for row in rows if str(row.get("source", "")).lower() == "pyq"]

        result: List[Dict[str, str]] = []
        for row in pyqs[:limit]:
            result.append(
                {
                    "pyq_id": str(row.get("chunk_id", "")),
                    "text": str(row.get("text", "")),
                    "file_name": str(row.get("file_name", "")),
                    "paragraph_number": str(row.get("paragraph_number", "")),
                }
            )
        return result

    def _get_pyq_by_id(self, pyq_id: str) -> Dict[str, str]:
        for item in self.get_pyq_list(limit=10000):
            if item["pyq_id"] == pyq_id:
                return item
        raise ValueError(f"PYQ id not found: {pyq_id}")

    # ------------------------------------------------------------------
    # Hybrid retrieval: Vector + BM25
    # ------------------------------------------------------------------

    def _get_bm25_results(self, query_text: str, top_k: int = 20) -> Dict[str, float]:
        """Get BM25 scores for query.
        
        Returns dict of {chunk_id: bm25_score}.
        """
        if self._bm25 is None:
            return {}

        results = self._bm25.score(query_text, top_k=top_k)
        return {r["id"]: r["score"] for r in results}

    def _get_vector_results(
        self, query_embedding: List[float], top_k: int = 30
    ) -> Dict[str, Any]:
        """Get vector search results from ChromaDB."""
        return self.indexer.search(
            query_embedding=query_embedding,
            top_k=top_k,
            filter_metadata={"source": "ncert"},
        )

    # ------------------------------------------------------------------
    # Ranking and fusion
    # ------------------------------------------------------------------

    def _to_ranked_chunks(
        self,
        raw_results: Dict[str, Any],
        query_text: str,
        bm25_scores: Optional[Dict[str, float]] = None,
    ) -> List[RankedChunk]:
        """Fuse vector search + BM25 + cross-encoder into a unified ranking.
        
        WHY HYBRID FUSION:
        - Vector search: finds semantically related paragraphs
        - BM25: finds exact keyword matches (critical for science terms)
        - Cross-encoder: precise pairwise relevance scoring
        
        Each signal captures different types of relevance.
        Combining them is consistently better than any single signal.
        """
        # Chroma can return list/tuple/numpy-array containers depending on versions.
        # Normalize each field into a mutable Python list so we can append BM25-only rows.
        def _to_mutable_list(value: Any) -> List[Any]:
            """
            Chroma query results are 2D: value[0] contains the actual 1D result list.
            Convert to mutable Python list, handling numpy arrays and tuples.
            """
            if not value:
                return []
            try:
                # Chroma wraps results: extract the first (and only) batch
                first_batch = value[0] if isinstance(value, (list, tuple)) else value
                if first_batch is None:
                    return []
                # Convert numpy array or tuple to Python list
                return list(first_batch)
            except (IndexError, TypeError):
                return []

        ids = _to_mutable_list(raw_results.get("ids", []))
        docs = _to_mutable_list(raw_results.get("documents", []))
        metas = _to_mutable_list(raw_results.get("metadatas", []))
        dists = _to_mutable_list(raw_results.get("distances", []))
        embs = _to_mutable_list(raw_results.get("embeddings", []))

        if not ids:
            return []

        bm25_scores = bm25_scores or {}

        # Merge BM25-only results that vector search missed
        bm25_only_ids = set(bm25_scores.keys()) - set(ids)
        for bm25_id in list(bm25_only_ids)[:10]:  # Add top-10 BM25-only results
            if bm25_id in self._bm25_doc_map:
                doc_info = self._bm25_doc_map[bm25_id]
                ids.append(bm25_id)
                docs.append(doc_info["text"])
                metas.append(doc_info["metadata"])
                dists.append(1.0)  # Max cosine distance (will be normalized)
                embs.append([])

        # --- Cross-encoder reranking (pairwise scoring) ---
        # Use raw text for reranking (without heading prefix) when possible
        rerank_texts = []
        for doc in docs:
            # Strip heading prefix for reranking (the cross-encoder
            # should see natural text, not our internal format)
            clean = re.sub(r"^\[.*?\]\s*", "", doc)
            rerank_texts.append(clean)

        rerank_pairs = [[query_text, rt] for rt in rerank_texts]
        rerank_scores = self.reranker.predict(rerank_pairs)

        # --- Compute all score components ---
        query_terms = QueryPreprocessor.extract_key_terms(query_text)

        vector_scores: List[float] = []
        rerank_scores_list: List[float] = []
        bm25_scores_list: List[float] = []
        keyword_scores: List[float] = []

        for idx in range(len(ids)):
            # Vector similarity (convert cosine distance to similarity)
            vs = 1 - float(dists[idx]) if idx < len(dists) else 0.0
            vector_scores.append(max(vs, 0.0))

            # Cross-encoder score
            rerank_scores_list.append(float(rerank_scores[idx]))

            # BM25 score
            bm25_s = bm25_scores.get(ids[idx], 0.0)
            bm25_scores_list.append(float(bm25_s))

            # Simple keyword overlap
            doc_terms = self._token_set(docs[idx])
            if query_terms and doc_terms:
                overlap = len(query_terms.intersection(doc_terms)) / max(1, len(query_terms))
            else:
                overlap = 0.0
            keyword_scores.append(float(overlap))

        # --- Normalize score components where needed ---
        # Cross-encoder raw scores are unbounded; convert them to [0, 1] probabilities.
        rerank_prob = [self._sigmoid(score) for score in rerank_scores_list]
        # Vector scores are already cosine similarities in [0, 1] after clipping.
        # BM25 uses query-relative min-max normalization because raw magnitudes depend on query length.
        bm25_norm = self._minmax_normalize(bm25_scores_list) if any(bm25_scores_list) else [0.0] * len(ids)

        # --- Compute hybrid fusion score ---
        ranked: List[RankedChunk] = []
        for idx, chunk_id in enumerate(ids):
            hybrid_score = (
                self.WEIGHT_RERANK * rerank_prob[idx]
                + self.WEIGHT_VECTOR * vector_scores[idx]
                + self.WEIGHT_BM25 * bm25_norm[idx]
                + self.WEIGHT_KEYWORD * keyword_scores[idx]
            )

            vector_payload: List[float] = []
            if idx < len(embs) and embs[idx] is not None:
                try:
                    vector_payload = list(embs[idx])
                except Exception:
                    vector_payload = []

            # Get raw text (without heading prefix) for display
            raw_doc = docs[idx]
            display_text = re.sub(r"^\[.*?\]\s*", "", raw_doc)

            ranked.append(
                RankedChunk(
                    chunk_id=chunk_id,
                    text=raw_doc,
                    text_raw=display_text,
                    metadata=metas[idx] if idx < len(metas) else {},
                    vector=vector_payload,
                    vector_score=vector_scores[idx],
                    rerank_score=rerank_scores_list[idx],
                    bm25_score=bm25_scores_list[idx],
                    keyword_score=keyword_scores[idx],
                    hybrid_score=float(hybrid_score),
                )
            )

        ranked.sort(key=lambda row: row.hybrid_score, reverse=True)
        return ranked

    # ------------------------------------------------------------------
    # Second paragraph selection
    # ------------------------------------------------------------------

    def _adds_new_information(self, first_text: str, second_text: str) -> bool:
        """Check if second paragraph adds genuinely new information."""
        tokens_a = self._token_set(first_text)
        tokens_b = self._token_set(second_text)
        if not tokens_a or not tokens_b:
            return True
        jaccard = len(tokens_a.intersection(tokens_b)) / max(1, len(tokens_a.union(tokens_b)))
        return jaccard < self.REDUNDANCY_THRESHOLD

    def _covers_additional_query_concept(
        self, query: str, first_text: str, second_text: str
    ) -> bool:
        """Check if second paragraph covers query concepts not in first."""
        query_terms = QueryPreprocessor.extract_key_terms(query)
        if not query_terms:
            return True

        first_terms = self._token_set(first_text)
        second_terms = self._token_set(second_text)

        covered_by_first = query_terms.intersection(first_terms)
        covered_by_second = query_terms.intersection(second_terms)
        additional = covered_by_second - covered_by_first
        return len(additional) > 0

    def _select_second_paragraph(
        self, query: str, best: RankedChunk, candidates: List[RankedChunk]
    ) -> tuple:
        """Select the best second supporting paragraph.
        
        Returns (status, paragraph_or_None).
        
        A second paragraph is selected ONLY if it:
        1. Has a decent score relative to the first
        2. Is NOT redundant with the first
        3. Covers additional query concepts
        """
        if len(candidates) < 2:
            return "not_available", None

        for candidate in candidates[1:6]:  # Check top 5 alternatives
            # Score ratio check
            if candidate.hybrid_score < self.SECOND_PARA_RATIO * best.hybrid_score:
                continue

            # Redundancy check (text overlap)
            if not self._adds_new_information(best.text_raw, candidate.text_raw):
                continue

            # Embedding redundancy check
            if (len(best.vector) > 0 and len(candidate.vector) > 0 
                and self._cosine_similarity(best.vector, candidate.vector) > 0.85):
                continue

            # Must cover additional query concepts
            if not self._covers_additional_query_concept(query, best.text_raw, candidate.text_raw):
                continue

            return "selected", candidate

        return "discarded_no_suitable_candidate", None

    # ------------------------------------------------------------------
    # Main query method
    # ------------------------------------------------------------------

    def query_from_pyq(
        self,
        pyq_id: Optional[str] = None,
        pyq_text: Optional[str] = None,
        top_k: int = 10,
    ) -> Dict[str, Any]:
        """Main entry point: given a PYQ, find the best NCERT paragraph(s).
        
        Pipeline:
        1. Preprocess PYQ text (remove noise, extract concepts)
        2. Get candidates from vector search (top 30)
        3. Get candidates from BM25 search (top 20)
        4. Merge and deduplicate candidates
        5. Cross-encoder rerank all candidates
        6. Hybrid fusion scoring
        7. Apply relevance threshold
        8. Select best + optional second paragraph
        """
        # --- Resolve PYQ text ---
        if pyq_id:
            selected_pyq = self._get_pyq_by_id(pyq_id)
            raw_query = selected_pyq["text"].strip()
        else:
            if not pyq_text or not pyq_text.strip():
                raise ValueError("Either pyq_id or pyq_text must be provided")
            raw_query = pyq_text.strip()
            selected_pyq = {
                "pyq_id": "ad_hoc_pyq",
                "text": raw_query,
                "file_name": "manual_input",
                "paragraph_number": "",
            }

        # --- Step 1: Preprocess query ---
        query_text = self.preprocessor.preprocess(raw_query)
        logger.info("Preprocessed query: '%s' -> '%s'", raw_query[:80], query_text[:80])

        # --- Step 2: Vector search ---
        query_embedding = self.embedder.encode(
            query_text,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).tolist()

        raw_results = self._get_vector_results(
            query_embedding, top_k=self.VECTOR_CANDIDATES
        )

        # --- Step 3: BM25 search ---
        bm25_scores = self._get_bm25_results(query_text, top_k=self.BM25_CANDIDATES)

        # --- Steps 4-6: Merge, rerank, hybrid fusion ---
        ranked = self._to_ranked_chunks(
            raw_results=raw_results,
            query_text=query_text,
            bm25_scores=bm25_scores,
        )

        # --- Step 7: Relevance threshold ---
        if not ranked:
            return self._empty_response(selected_pyq, preprocessed_query=query_text)

        best = ranked[0]

        # If even the best result is below threshold, return nothing
        if best.hybrid_score < self.MIN_HYBRID_SCORE:
            logger.warning(
                "Best result score %.3f below threshold %.3f — no relevant match",
                best.hybrid_score, self.MIN_HYBRID_SCORE,
            )
            return self._empty_response(selected_pyq, ranked[:5], preprocessed_query=query_text)

        # --- Step 8: Select second paragraph ---
        second_status, second_chunk = self._select_second_paragraph(
            query_text, best, ranked
        )

        second_payload = None
        if second_chunk:
            second_payload = {
                "chunk_id": second_chunk.chunk_id,
                "text": second_chunk.text_raw,
                "score": second_chunk.hybrid_score,
                "metadata": second_chunk.metadata,
            }

        response = {
            "selected_pyq": selected_pyq,
            "best_matching_ncert_paragraph": {
                "chunk_id": best.chunk_id,
                "text": best.text_raw,    # Display raw text, not heading-prepended
                "score": best.hybrid_score,
                "metadata": best.metadata,
            },
            "second_supporting_paragraph": second_payload,
            "debug": {
                "preprocessed_query": query_text,
                "top_cross_encoder_scores": [
                    {
                        "chunk_id": row.chunk_id,
                        "score": row.hybrid_score,
                        "cross_score": row.rerank_score,
                        "vector_score": row.vector_score,
                        "bm25_score": row.bm25_score,
                        "keyword_score": row.keyword_score,
                        "file_name": str(row.metadata.get("file_name", "")),
                        "paragraph_number": row.metadata.get("paragraph_number", ""),
                        "heading": row.metadata.get("heading_context", ""),
                    }
                    for row in ranked[:8]
                ],
                "second_paragraph_status": second_status,
                "retrieved_files": sorted(
                    {str(row.metadata.get("file_name", "")) for row in ranked[:8]}
                ),
                "bm25_candidates": len(bm25_scores),
                "total_candidates_reranked": len(ranked),
            },
        }
        return response

    def _empty_response(
        self,
        selected_pyq: Dict,
        ranked: Optional[List[RankedChunk]] = None,
        preprocessed_query: str = "",
    ) -> Dict[str, Any]:
        """Build response when no relevant results found."""
        ranked = ranked or []
        debug_scores = []
        if ranked:
            debug_scores = [
                {
                    "chunk_id": row.chunk_id,
                    "score": row.hybrid_score,
                    "cross_score": row.rerank_score,
                    "vector_score": row.vector_score,
                    "bm25_score": row.bm25_score,
                    "keyword_score": row.keyword_score,
                    "file_name": str(row.metadata.get("file_name", "")),
                    "paragraph_number": row.metadata.get("paragraph_number", ""),
                    "heading": row.metadata.get("heading_context", ""),
                }
                for row in ranked
            ]

        retrieved_files = sorted(
            {str(row.metadata.get("file_name", "")) for row in ranked}
        )

        return {
            "selected_pyq": selected_pyq,
            "best_matching_ncert_paragraph": None,
            "second_supporting_paragraph": None,
            "debug": {
                "preprocessed_query": preprocessed_query,
                "top_cross_encoder_scores": debug_scores,
                "second_paragraph_status": "discarded_no_candidates",
                "retrieved_files": retrieved_files,
                "bm25_candidates": 0,
                "total_candidates_reranked": len(ranked),
            },
        }

    # Keep the old method name as an alias for _prepare_query_text
    def _prepare_query_text(self, text: str) -> str:
        """Legacy wrapper — delegates to QueryPreprocessor."""
        return self.preprocessor.preprocess(text)

    # Keep _mcq_keyword_set for backward compatibility
    def _mcq_keyword_set(self, text: str) -> set:
        return QueryPreprocessor.extract_key_terms(text)
