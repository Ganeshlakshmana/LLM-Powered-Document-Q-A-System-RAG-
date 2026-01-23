from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

from src.config import settings
from src.ingest import load_docs, chunk_docs
from src.vectorstore import build_faiss, save_faiss, load_faiss
from src.rag import answer as rag_answer


load_dotenv()

app = FastAPI(title="LLM Document Q&A (RAG)")
vectorstore = None

class IngestRequest(BaseModel):
    filename: str  # e.g. "sample.pdf"

class AskRequest(BaseModel):
    question: str
    top_k: int | None = None

@app.on_event("startup")
def startup():
    global vectorstore
    vectorstore = load_faiss(settings.index_dir)


@app.get("/health")
def health():
    return {"status": "ok", "indexed": vectorstore is not None}

@app.post("/ingest")
def ingest(req: IngestRequest):
    global vectorstore
    try:
        file_path = (settings.data_dir / req.filename).resolve()
        if settings.data_dir.resolve() not in file_path.parents:
            raise HTTPException(400, "Invalid filename/path")
        if not file_path.exists():
            raise HTTPException(404, f"File not found: {req.filename}")

        # 1) Load docs
        docs = load_docs(file_path)

        # 2) Chunk docs
        chunks = chunk_docs(docs, settings.chunk_size, settings.chunk_overlap)

        # 3) Build vectorstore (embeddings + FAISS)
        vectorstore = build_faiss(chunks)

        # 4) Save
        save_faiss(vectorstore, settings.index_dir)

        return {
            "message": "Indexed successfully",
            "docs": len(docs),
            "chunks": len(chunks),
            "file": req.filename
        }

    except Exception as e:
        return {
            "error_type": type(e).__name__,
            "error": str(e)
        }


@app.post("/ask")
def ask(req: AskRequest):
    if vectorstore is None:
        raise HTTPException(400, "No index found. Call /ingest first.")

    top_k = req.top_k or settings.top_k
    ans, sources = rag_answer(vectorstore, req.question, top_k)
    return {"answer": ans, "sources": sources}
