from pathlib import Path
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from src.config import settings

def build_faiss(chunks):
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=settings.openai_api_key
    )
    return FAISS.from_documents(chunks, embeddings)

def save_faiss(vs, index_dir: Path):
    index_dir.mkdir(parents=True, exist_ok=True)
    vs.save_local(str(index_dir))

def load_faiss(index_dir: Path):
    faiss_file = index_dir / "index.faiss"
    pkl_file = index_dir / "index.pkl"

    if not (faiss_file.exists() and pkl_file.exists()):
        return None

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=settings.openai_api_key
    )
    return FAISS.load_local(
        str(index_dir),
        embeddings,
        allow_dangerous_deserialization=True
    )
