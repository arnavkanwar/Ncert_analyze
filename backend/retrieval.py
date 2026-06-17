"""
Core retrieval function for NCERT paragraphs.

This module uses the improved IntelligentQueryEngine from the main codebase
to retrieve top 2 NCERT paragraphs for a given PYQ.

IMPROVED: Works with the new hybrid retrieval engine (BM25 + vector + cross-encoder).
"""

import sys
from pathlib import Path
import logging
import re
from typing import Dict, List
import math

# Add main codebase to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.retrieval.intelligent_query import IntelligentQueryEngine
from src.pipeline.pyq_filter import is_valid_question

logger = logging.getLogger(__name__)


class NCERTRetriever:
    """Retrieves NCERT paragraphs for PYQ questions."""
    
    def __init__(
        self,
        chroma_dir: str = "chroma_db",
        collection_name: str = "ncert_chemistry",
        chunks_path: str = "output/chunks/all_chunks.json",
        device: str = "cpu"
    ):
        """
        Initialize the NCERT retriever.
        
        Args:
            chroma_dir: Directory with ChromaDB persistence
            collection_name: ChromaDB collection name
            chunks_path: Path to chunks JSON file
            device: 'cpu' or 'cuda'
        """
        try:
            self.engine = IntelligentQueryEngine(
                chroma_dir=Path(chroma_dir),
                collection_name=collection_name,
                chunks_path=Path(chunks_path),
                device=device
            )
            logger.info("NCERTRetriever initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize NCERTRetriever: {e}")
            raise
    
    def get_top_paragraphs(self, pyq_text: str, top_k: int = 2) -> list:
        """
        Retrieve top 2 most relevant NCERT paragraphs for a PYQ.
        
        Args:
            pyq_text: The PYQ question text
            top_k: Number of paragraphs to retrieve (default: 2)
            
        Returns:
            List of top 2 paragraph texts
            
        Raises:
            ValueError: If pyq_text is empty
            Exception: If ChromaDB is not initialized
        """
        if not pyq_text or not pyq_text.strip():
            raise ValueError("PYQ text cannot be empty")
        
        try:
            candidate_pool = max(20, top_k * 10)
            # Query the engine with the PYQ
            result = self.engine.query_from_pyq(
                pyq_text=pyq_text,
                top_k=candidate_pool
            )
            
            # Extract paragraphs from result
            paragraphs = []
            
            # Add best matching paragraph
            if result.get("best_matching_ncert_paragraph"):
                best = result["best_matching_ncert_paragraph"]
                paragraphs.append(best["text"])
                logger.info(f"Retrieved best match: {best['chunk_id']}")
            
            # Add second supporting paragraph if available
            if result.get("second_supporting_paragraph"):
                second = result["second_supporting_paragraph"]
                paragraphs.append(second["text"])
                logger.info(f"Retrieved second match: {second['chunk_id']}")
            
            if not paragraphs:
                logger.warning(f"No paragraphs found for query: {pyq_text[:50]}...")
            
            return paragraphs[:top_k]
            
        except Exception as e:
            logger.error(f"Error retrieving paragraphs: {e}")
            raise

    @staticmethod
    def _tokenize(text: str) -> set:
        return {tok for tok in re.findall(r"[a-zA-Z0-9]+", text.lower()) if len(tok) > 2}

    @staticmethod
    def _sigmoid(score: float) -> float:
        return 1.0 / (1.0 + math.exp(-score))

    @staticmethod
    def _normalize(scores: List[float]) -> List[float]:
        if not scores:
            return []
        min_score = min(scores)
        max_score = max(scores)
        if math.isclose(max_score, min_score):
            return [1.0 for _ in scores]
        return [(s - min_score) / (max_score - min_score) for s in scores]

    def _build_reason(self, query_text: str, paragraph_text: str, score: float, rank: int) -> str:
        query_tokens = self._tokenize(query_text)
        para_tokens = self._tokenize(paragraph_text)
        overlap = sorted(list(query_tokens.intersection(para_tokens)))
        top_terms = ", ".join(overlap[:5]) if overlap else "key semantic concepts"
        return (
            f"Rank #{rank} by hybrid relevance (score {score:.3f}); "
            f"matches query terms: {top_terms}."
        )

    def _get_pyqs_from_chroma(self, limit: int = 300) -> List[Dict]:
        """Read indexed PYQs directly from Chroma to avoid stale JSON snapshots."""
        payload = self.engine.indexer.collection.get(
            where={"source": "pyq"},
            limit=limit,
            include=["documents", "metadatas"],
        )

        ids = payload.get("ids", [])
        docs = payload.get("documents", [])
        metas = payload.get("metadatas", [])

        items: List[Dict] = []
        for idx, pyq_id in enumerate(ids):
            meta = metas[idx] if idx < len(metas) and metas[idx] else {}
            question_number = meta.get("question_number", "")
            try:
                question_number = int(question_number)
            except Exception:
                question_number = 0

            text = str(docs[idx]) if idx < len(docs) else ""

            # Always validate — question_number is set on every chunk during
            # ingestion (idx+1), so it cannot be used as a real-question signal.
            if not is_valid_question(text):
                continue

            items.append(
                {
                    "pyq_id": str(pyq_id),
                    "text": text,
                    "file_name": str(meta.get("file_name", "")),
                    "paragraph_number": str(question_number if question_number else ""),
                    "_sort_qn": question_number,
                }
            )

        # Stable ordering for a predictable sidebar experience.
        items.sort(key=lambda row: (row.get("file_name", ""), row.get("_sort_qn", 0), row.get("pyq_id", "")))
        for row in items:
            row.pop("_sort_qn", None)
        return items

    def _get_pyq_by_id_from_chroma(self, pyq_id: str) -> Dict:
        payload = self.engine.indexer.collection.get(
            ids=[pyq_id],
            include=["documents", "metadatas"],
        )

        ids = payload.get("ids", [])
        if not ids:
            raise ValueError(f"PYQ id not found: {pyq_id}")

        meta = (payload.get("metadatas", [{}]) or [{}])[0] or {}
        doc = (payload.get("documents", [""]) or [""])[0]
        question_number = meta.get("question_number", "")

        return {
            "pyq_id": str(ids[0]),
            "text": str(doc),
            "file_name": str(meta.get("file_name", "")),
            "paragraph_number": str(question_number),
        }

    def get_pyq_list(self, limit: int = 300) -> List[Dict]:
        """Return available PYQs for sidebar selection."""
        items = self._get_pyqs_from_chroma(limit=limit)
        if items:
            return items
        return self.engine.get_pyq_list(limit=limit)

    def query_by_pyq_id_with_diagnostics(
        self,
        pyq_id: str,
        top_k_return: int = 2,
        top_k_candidates: int = 20,
    ) -> Dict:
        """Return top matches with scores and chart-friendly diagnostics.
        
        IMPROVED: Uses the new hybrid retrieval pipeline.
        """
        if not pyq_id or not pyq_id.strip():
            raise ValueError("pyq_id cannot be empty")

        selected_pyq = self._get_pyq_by_id_from_chroma(pyq_id.strip())
        query_text = selected_pyq["text"].strip()
        
        # Use new preprocessor
        query_text_for_retrieval = self.engine._prepare_query_text(query_text)

        query_embedding = self.engine.embedder.encode(
            query_text_for_retrieval,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).tolist()

        # --- IMPROVED: Hybrid retrieval (vector + BM25) ---
        raw_results = self.engine._get_vector_results(
            query_embedding, top_k=max(top_k_candidates, 30)
        )
        bm25_scores = self.engine._get_bm25_results(
            query_text_for_retrieval, top_k=20
        )
        ranked = self.engine._to_ranked_chunks(
            raw_results=raw_results,
            query_text=query_text_for_retrieval,
            bm25_scores=bm25_scores,
        )

        matches = []
        for idx, row in enumerate(ranked[:top_k_return], start=1):
            confidence = float(getattr(row, "hybrid_score", 0.0))
            matches.append(
                {
                    "rank": idx,
                    "chunk_id": row.chunk_id,
                    "text": row.text_raw,  # Use raw text for display
                    "score": confidence,
                    "vector_score": float(row.vector_score),
                    "rerank_score": float(row.rerank_score),
                    "bm25_score": float(row.bm25_score),
                    "file_name": str(row.metadata.get("file_name", "")),
                    "paragraph_number": row.metadata.get("paragraph_number", ""),
                    "reason": self._build_reason(query_text_for_retrieval, row.text_raw, confidence, idx),
                }
            )

        chart = []
        for idx, row in enumerate(ranked[:6], start=1):
            confidence = float(getattr(row, "hybrid_score", 0.0))
            chart.append(
                {
                    "label": f"C{idx}",
                    "score": confidence,
                    "chunk_id": row.chunk_id,
                    "file_name": str(row.metadata.get("file_name", "")),
                    "text": row.text_raw,  # Use raw text for display
                }
            )

        return {
            "selected_pyq": selected_pyq,
            "matches": matches,
            "chart": chart,
            "count": len(matches),
        }


def get_retriever():
    """Factory function to get retriever instance (for dependency injection)."""
    return NCERTRetriever()
