from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_docs(file_path: Path):
    if file_path.suffix.lower() == ".pdf":
        return PyPDFLoader(str(file_path)).load()
    if file_path.suffix.lower() in [".txt", ".md"]:
        return TextLoader(str(file_path), encoding="utf-8").load()
    raise ValueError(f"Unsupported file type: {file_path.suffix}")

def chunk_docs(docs, chunk_size: int, chunk_overlap: int):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(docs)
