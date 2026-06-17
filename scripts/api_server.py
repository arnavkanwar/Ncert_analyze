"""Optional FastAPI endpoint for semantic query."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel

from src.retrieval.intelligent_query import IntelligentQueryEngine

app = FastAPI(title="NCERT Chemistry Query API")
engine = IntelligentQueryEngine(chroma_dir=Path("chroma_db"), collection_name="ncert_chemistry", device="cpu")


class QueryRequest(BaseModel):
    question: str


@app.post("/query")
def query(req: QueryRequest):
    return engine.query(question=req.question, top_k=10)
