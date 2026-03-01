import os
import hashlib
import logging
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from .rag_engine import RAGService

load_dotenv(Path(__file__).parent.parent / ".env")
API_KEY_PREVIEW = os.environ.get("NVIDIA_API_KEY", "NOT_FOUND")[:12]
print(f"KEY: {API_KEY_PREVIEW}...")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="RAG Service", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

rag = RAGService()

# ── Schemas ──────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=2, description="Вопрос пользователя")
    top_k: int    = Field(7, ge=1, le=20, description="Количество извлекаемых фрагментов")

class IndexRequest(BaseModel):
    file_ids: list[str] = Field(..., min_length=1)

# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "chunks_indexed": rag.total_chunks()}

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Принимает файл, извлекает текст, возвращает file_id. Индексация не запускается."""
    allowed_exts = {".txt", ".md", ".pdf", ".docx"}
    ext = os.path.splitext(file.filename)[1].lower() if file.filename else ""

    if ext not in allowed_exts:
        raise HTTPException(400, f"Unsupported format: {ext}. Allowed: {', '.join(allowed_exts)}")

    content = await file.read()
    if len(content) > 50 * 1024 * 1024:
        raise HTTPException(413, "File too large (max 50MB)")

    file_id = await run_in_threadpool(lambda: hashlib.sha256(content).hexdigest()[:16])

    try:
        text = await run_in_threadpool(rag.extract_text, content, ext, file.filename)
    except Exception as e:
        logger.error(f"Failed to extract text from {file.filename}: {e}", exc_info=True)
        raise HTTPException(422, f"Failed to extract text: {e}")

    if len(text.strip()) < 10:
        raise HTTPException(422, "File appears empty or has no extractable text")

    rag.store_pending(file_id, file.filename, ext, text)

    return {
        "file_id":   file_id,
        "file_name": file.filename,
        "chars":     len(text),
        "status":    "uploaded",
    }

@app.post("/index")
async def index_files(req: IndexRequest):
    """Чанкинг + эмбеддинги + сохранение в ChromaDB. BM25 пересобирается один раз в конце."""
    if not req.file_ids:
        raise HTTPException(400, "No file IDs provided")

    results, errors = [], []
    for fid in req.file_ids:
        try:
            res = await rag.index_file(fid)
            results.append(res)
        except Exception as e:
            logger.error(f"Error indexing {fid}: {e}", exc_info=True)
            errors.append({"file_id": fid, "error": str(e)})

    if results:
        await rag.rebuild_bm25()

    return {"indexed": results, "errors": errors}

@app.post("/query")
async def query(req: QueryRequest):
    """Запускает LangGraph-пайплайн: retrieve → grade → generate."""
    if rag.total_chunks() == 0:
        raise HTTPException(404, "No documents indexed yet")

    try:
        result = await rag.query(req.question, top_k=req.top_k)
    except Exception as e:
        logger.error(f"Query error: {e}", exc_info=True)
        raise HTTPException(500, "Internal Server Error during query execution.")

    return result

@app.get("/documents")
def list_documents():
    return {"documents": rag.list_documents()}

@app.delete("/documents/{file_id}")
async def delete_document(file_id: str):
    await rag.delete_document(file_id)
    return {"status": "deleted", "file_id": file_id}

# ── Serve frontend ────────────────────────────────────────────────────────────

frontend_dir = Path(__file__).parent.parent / "frontend"
if frontend_dir.is_dir():
    app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")

    @app.get("/")
    def serve_frontend():
        return FileResponse(str(frontend_dir / "index.html"))
