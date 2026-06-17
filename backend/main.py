"""
FastAPI backend for NCERT paragraph retrieval.

Endpoint:
  POST /query - Retrieve top 2 NCERT paragraphs for a PYQ
"""

import logging
from typing import List
from pathlib import Path
from contextlib import asynccontextmanager
import asyncio
import sys
import subprocess

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

try:
    from .retrieval import NCERTRetriever
except ImportError:
    from retrieval import NCERTRetriever

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

# ============================================================================
# Global Retriever Instance
# ============================================================================

retriever = None

INGESTION_MARKER = BASE_DIR / "output" / ".last_ingestion_pdf_mtime"


def _pdf_fingerprint() -> tuple:
    """Return (max_mtime, file_count) across data/books and data/pyqs PDFs.

    Tracking both values detects two distinct change types:
    - file_count change: a new file added (even with an old mtime)
    - max_mtime change: an existing file was modified/replaced
    """
    roots = [BASE_DIR / "data" / "books", BASE_DIR / "data" / "pyqs"]
    latest = 0.0
    count = 0
    for root in roots:
        if not root.exists():
            continue
        for pdf_file in root.rglob("*.pdf"):
            try:
                latest = max(latest, pdf_file.stat().st_mtime)
                count += 1
            except OSError:
                continue
    return latest, count


def _read_last_ingestion_marker() -> tuple:
    """Return (max_mtime, file_count) from the marker file."""
    if not INGESTION_MARKER.exists():
        return 0.0, 0
    try:
        raw = INGESTION_MARKER.read_text(encoding="utf-8").strip()
        parts = raw.split(",")
        mtime = float(parts[0]) if parts else 0.0
        count = int(parts[1]) if len(parts) > 1 else 0
        return mtime, count
    except Exception:
        return 0.0, 0


def _write_last_ingestion_marker(mtime: float, count: int) -> None:
    INGESTION_MARKER.parent.mkdir(parents=True, exist_ok=True)
    INGESTION_MARKER.write_text(f"{mtime},{count}", encoding="utf-8")


def _run_full_ingestion() -> None:
    """Run full book+pyq ingestion script using current Python interpreter."""
    command = [sys.executable, "scripts/process_books.py"]
    logger.info("Running ingestion sync: %s", " ".join(command))
    result = subprocess.run(command, cwd=str(BASE_DIR), check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Ingestion failed with exit code {result.returncode}")


def _sync_ingestion_if_needed(force: bool = False) -> bool:
    """Ingest new PDFs only when source files changed since last successful run."""
    latest_mtime, current_count = _pdf_fingerprint()
    if current_count == 0:
        logger.info("No PDFs found in data/books or data/pyqs; skipping ingestion sync")
        return False

    last_mtime, last_count = _read_last_ingestion_marker()
    needs_sync = force or current_count != last_count or latest_mtime > last_mtime
    if not needs_sync:
        logger.info("Ingestion sync not required; no new PDFs detected")
        return False

    _run_full_ingestion()
    _write_last_ingestion_marker(latest_mtime, current_count)
    logger.info("Ingestion sync completed successfully")
    return True


def _create_retriever() -> NCERTRetriever:
    return NCERTRetriever(
        chroma_dir=str(BASE_DIR / "chroma_db"),
        collection_name="ncert_chemistry",
        chunks_path=str(BASE_DIR / "output/chunks/all_chunks.json"),
        device="cpu"
    )


def _reload_retriever() -> None:
    global retriever
    retriever = _create_retriever()


# How often the background task polls for new PDFs (seconds).
_WATCH_INTERVAL = 30


async def _background_watcher() -> None:
    """Poll data/books and data/pyqs every _WATCH_INTERVAL seconds.

    When new or changed PDFs are detected (via mtime comparison), the ingestion
    pipeline runs automatically and the retriever is reloaded so the sidebar
    reflects the new content without a server restart.
    """
    logger.info("Background file watcher started (interval=%ds)", _WATCH_INTERVAL)
    while True:
        await asyncio.sleep(_WATCH_INTERVAL)
        try:
            loop = asyncio.get_event_loop()
            synced = await loop.run_in_executor(None, _sync_ingestion_if_needed)
            if synced:
                logger.info("New PDFs detected — reloading retriever after auto-ingest")
                _reload_retriever()
                logger.info("Retriever reloaded; sidebar will reflect new content on next /pyqs call")
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.error("Background watcher error (will retry next cycle): %s", exc)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage app lifecycle: initialization, background watcher, and cleanup."""
    global retriever

    # Startup — run ingestion if PDFs changed since last run, then load retriever
    logger.info("Starting FastAPI server...")
    try:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, _sync_ingestion_if_needed)
        _reload_retriever()
        logger.info("NCERT Retriever initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize retriever: {e}")
        raise

    # Start background watcher that auto-ingests newly dropped PDFs
    watcher_task = asyncio.create_task(_background_watcher())

    yield

    # Shutdown
    watcher_task.cancel()
    try:
        await watcher_task
    except asyncio.CancelledError:
        pass
    logger.info("Shutting down FastAPI server...")
    retriever = None


# ============================================================================
# Pydantic Models
# ============================================================================

class QueryRequest(BaseModel):
    """Request model for /query endpoint."""
    question: str = Field(
        ...,
        min_length=3,
        max_length=2000,
        description="The PYQ question text"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "question": "What is photosynthesis? Explain in detail."
            }
        }


class QueryResponse(BaseModel):
    """Response model for /query endpoint."""
    paragraphs: List[str] = Field(
        default_factory=list,
        description="List of top 2 most relevant NCERT paragraphs"
    )
    count: int = Field(
        description="Number of paragraphs retrieved"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "paragraphs": [
                    "Photosynthesis is the process by which plants use sunlight...",
                    "The light-dependent reactions occur in the thylakoid membranes..."
                ],
                "count": 2
            }
        }


class PyqItem(BaseModel):
    """PYQ item for sidebar display."""
    pyq_id: str
    text: str
    file_name: str = ""
    paragraph_number: str = ""


class PyqListResponse(BaseModel):
    """Response model for PYQ list endpoint."""
    items: List[PyqItem]
    count: int


class QueryByPyqRequest(BaseModel):
    """Request model for querying by selected pyq_id."""
    pyq_id: str = Field(..., min_length=1, description="Selected PYQ id from /pyqs")


class ParagraphMatch(BaseModel):
    """Detailed ranked paragraph match."""
    rank: int
    chunk_id: str
    text: str
    score: float
    vector_score: float
    rerank_score: float
    bm25_score: float
    file_name: str
    paragraph_number: str | int
    reason: str


class ScorePoint(BaseModel):
    """Chart-ready score data point."""
    label: str
    score: float
    chunk_id: str
    file_name: str
    text: str


class QueryByPyqResponse(BaseModel):
    """Response model for sidebar-based retrieval with explainability."""
    selected_pyq: PyqItem
    matches: List[ParagraphMatch]
    chart: List[ScorePoint]
    count: int


class ErrorResponse(BaseModel):
    """Error response model."""
    detail: str = Field(description="Error message")
    code: str = Field(description="Error code")


class UploadResponse(BaseModel):
    """Response model for upload ingestion endpoint."""
    status: str
    chunks: int
    file: str


class SyncResponse(BaseModel):
    """Response model for ingestion sync endpoint."""
    status: str
    synced: bool


# ============================================================================
# FastAPI App
# ============================================================================

app = FastAPI(
    title="NCERT Retrieval API",
    description="API to retrieve top 2 most relevant NCERT paragraphs for PYQ questions",
    version="1.0.0",
    lifespan=lifespan
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (change in production)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Endpoints
# ============================================================================

@app.get("/", tags=["Health"])
async def root():
    """Root endpoint - API info."""
    return {
        "message": "NCERT Paragraph Retrieval API",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint."""
    if retriever is None:
        raise HTTPException(
            status_code=503,
            detail="Retriever not initialized"
        )
    
    return {
        "status": "healthy",
        "retriever_ready": True
    }


@app.get("/pyqs", response_model=PyqListResponse, tags=["Query"])
async def list_pyqs(limit: int = 300):
    """Return available PYQs for sidebar selection."""
    if retriever is None:
        raise HTTPException(
            status_code=503,
            detail="Retriever service not initialized",
            headers={"X-Error-Code": "RETRIEVER_NOT_INITIALIZED"}
        )

    items = retriever.get_pyq_list(limit=limit)
    return PyqListResponse(items=items, count=len(items))


@app.post("/query/by-pyq", response_model=QueryByPyqResponse, tags=["Query"])
async def query_by_pyq(request: QueryByPyqRequest):
    """Retrieve top NCERT paragraphs and diagnostics for selected pyq_id."""
    if retriever is None:
        raise HTTPException(
            status_code=503,
            detail="Retriever service not initialized",
            headers={"X-Error-Code": "RETRIEVER_NOT_INITIALIZED"}
        )

    try:
        result = retriever.query_by_pyq_id_with_diagnostics(
            pyq_id=request.pyq_id,
            top_k_return=2,
            top_k_candidates=20,
        )
        return QueryByPyqResponse(**result)
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=str(e),
            headers={"X-Error-Code": "INVALID_PYQ"}
        )
    except Exception as e:
        logger.error(f"Error processing query by pyq_id: {e}", exc_info=True)
        raise HTTPException(
            status_code=503,
            detail=f"Error retrieving paragraphs: {str(e)}",
            headers={"X-Error-Code": "RETRIEVAL_ERROR"}
        )


@app.post(
    "/query",
    response_model=QueryResponse,
    responses={
        200: {"description": "Successfully retrieved paragraphs"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        503: {"model": ErrorResponse, "description": "Service unavailable"},
    },
    tags=["Query"]
)
async def query_ncert(request: QueryRequest):
    """
    Retrieve top 2 most relevant NCERT paragraphs for a PYQ.
    
    **Input:**
    - `question`: The PYQ question text (3-2000 characters)
    
    **Output:**
    - `paragraphs`: List of top 2 NCERT paragraph texts
    - `count`: Number of paragraphs retrieved
    
    **Example:**
    ```json
    {
      "question": "What is photosynthesis?"
    }
    ```
    
    **Response:**
    ```json
    {
      "paragraphs": [
        "Paragraph 1 text...",
        "Paragraph 2 text..."
      ],
      "count": 2
    }
    ```
    """
    if retriever is None:
        logger.error("Retriever not initialized")
        raise HTTPException(
            status_code=503,
            detail="Retriever service not initialized. Please ensure ChromaDB is populated.",
            headers={"X-Error-Code": "RETRIEVER_NOT_INITIALIZED"}
        )
    
    try:
        # Log incoming query
        logger.info(f"Query received: {request.question[:100]}...")
        
        # Retrieve paragraphs
        paragraphs = retriever.get_top_paragraphs(request.question, top_k=2)
        
        # Log results
        logger.info(f"Retrieved {len(paragraphs)} paragraphs")
        
        # Return response
        return QueryResponse(
            paragraphs=paragraphs,
            count=len(paragraphs)
        )
        
    except ValueError as e:
        logger.warning(f"Invalid query: {e}")
        raise HTTPException(
            status_code=400,
            detail=str(e),
            headers={"X-Error-Code": "INVALID_QUERY"}
        )
    
    except Exception as e:
        logger.error(f"Error processing query: {e}", exc_info=True)
        raise HTTPException(
            status_code=503,
            detail=f"Error retrieving paragraphs: {str(e)}",
            headers={"X-Error-Code": "RETRIEVAL_ERROR"}
        )


@app.post("/upload/pyq", response_model=UploadResponse, tags=["Upload"])
async def upload_pyq(
    file: UploadFile = File(...),
    year: str = Form(...),
    subject: str = Form(...),
    class_name: str = Form(...),
):
    """Accept a PYQ PDF, ingest it, and return indexing status."""
    if file.content_type not in {"application/pdf", "application/x-pdf", "application/octet-stream"}:
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported",
            headers={"X-Error-Code": "INVALID_FILE_TYPE"},
        )

    save_dir = BASE_DIR / "data" / "pyqs"
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / file.filename
    save_path.write_bytes(await file.read())

    try:
        from scripts.ingest_pipeline import run_ingestion

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: run_ingestion(
                data_path=save_path,
                source="pyq",
                year=year,
                subject=subject,
                class_name=class_name,
            ),
        )
    except Exception as exc:
        logger.error("Upload ingestion failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to ingest uploaded PYQ: {exc}",
            headers={"X-Error-Code": "UPLOAD_INGEST_FAILED"},
        )

    return UploadResponse(status="indexed", chunks=int(result.get("count", 0)), file=file.filename)


@app.post("/admin/sync", response_model=SyncResponse, tags=["Admin"])
async def sync_ingestion(force: bool = False):
    """Sync PDFs from data folders into index and reload retriever."""
    try:
        loop = asyncio.get_event_loop()
        synced = await loop.run_in_executor(None, lambda: _sync_ingestion_if_needed(force=force))
        _reload_retriever()
        return SyncResponse(status="ok", synced=bool(synced))
    except Exception as exc:
        logger.error("Sync failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Sync failed: {exc}",
            headers={"X-Error-Code": "SYNC_FAILED"},
        )


# ============================================================================
# Exception Handlers
# ============================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions."""
    logger.error(f"HTTP Exception: {exc.status_code} - {exc.detail}")
    payload = {
        "detail": exc.detail,
        "code": exc.headers.get("X-Error-Code", "UNKNOWN_ERROR") if exc.headers else "UNKNOWN_ERROR",
    }
    return JSONResponse(
        status_code=exc.status_code,
        content=payload,
        headers=exc.headers or None,
    )


# ============================================================================
# Startup/Shutdown Events (Alternative to lifespan)
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Log startup."""
    logger.info("FastAPI application started")


@app.on_event("shutdown")
async def shutdown_event():
    """Log shutdown."""
    logger.info("FastAPI application shutdown")


if __name__ == "__main__":
    import uvicorn
    
    # Run with: python -m uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,  # Disable reload to ensure src/ changes are picked up
        log_level="info"
    )
