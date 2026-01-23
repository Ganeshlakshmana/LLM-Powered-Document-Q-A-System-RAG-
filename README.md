# LLM-Powered Document Q&A System (RAG)

A production-style Retrieval-Augmented Generation (RAG) system that enables accurate question answering over private documents by combining vector-based retrieval with large language models.

This project demonstrates how to ground LLM responses in source documents using embeddings and similarity search, reducing hallucinations and improving factual reliability.

---

## 🚀 Overview

Large Language Models are powerful but lack access to private or domain-specific documents.  
This project solves that limitation by implementing a **Retrieval-Augmented Generation (RAG)** pipeline that:

- Ingests documents (PDF/TXT)
- Splits them into semantic chunks
- Generates vector embeddings
- Retrieves relevant context using similarity search
- Produces grounded answers using an LLM

The system is exposed via a **FastAPI** service and persists its vector index for efficient reuse.

---

## 🧠 Key Features

- 📄 Document ingestion (PDF / TXT)
- ✂️ Recursive text chunking
- 🔢 Embedding generation using OpenAI models
- 📦 FAISS-based vector similarity search
- 🤖 Context-aware LLM responses (RAG)
- 📌 Source citations with page numbers and snippets
- 💾 Persistent vector index across restarts
- 🌐 REST API built with FastAPI

---

## 🏗️ Architecture

User Question
↓
Vector Similarity Search (FAISS)
↓
Top-K Relevant Chunks
↓
LLM with Retrieved Context
↓
Grounded Answer + Source Citations


---

## 🛠️ Tech Stack

- **Language**: Python
- **API Framework**: FastAPI
- **LLM & Embeddings**: OpenAI (via LangChain)
- **Vector Store**: FAISS
- **Document Processing**: PyPDF
- **Configuration**: Pydantic Settings
- **Environment Management**: python-dotenv

---

## 📂 Project Structure

rag_doc_qa/
├── src/
│ ├── app.py # FastAPI entry point
│ ├── config.py # Centralized configuration
│ ├── ingest.py # Document loading & chunking
│ ├── rag.py # Retrieval + LLM logic
│ └── vectorstore.py # FAISS index management
├── data/ # Input documents (not committed)
├── indexes/ # Persisted vector index (not committed)
├── requirements.txt
├── .env.example
└── README.md


---

## ⚙️ Setup & Usage

### 1️⃣ Clone the repository
```bash
git clone https://github.com/<your-username>/rag-doc-qa.git
cd rag-doc-qa
2️⃣ Create and activate virtual environment
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
3️⃣ Install dependencies
pip install -r requirements.txt
4️⃣ Configure environment variables
Create a .env file:

OPENAI_API_KEY=your_openai_api_key
5️⃣ Run the API
uvicorn src.app:app --reload
Open Swagger UI:

http://127.0.0.1:8000/docs
📌 API Endpoints
Health Check
GET /health

Returns service status and index availability.

Ingest Document
POST /ingest

{
  "filename": "sample.pdf"
}
Builds and persists the vector index.

Ask a Question
POST /ask

{
  "question": "What is this document about?",
  "top_k": 3
}
Returns:

Grounded answer

Source snippets with page references

🔒 Security & Best Practices
API keys are never committed to the repository

Vector indexes are generated locally and ignored by Git

File ingestion is restricted to a controlled directory

Configuration is centralized and environment-driven

📈 Future Improvements
Hybrid retrieval (BM25 + vector search)

Support for multiple documents and folders

Streaming responses

Evaluation metrics for retrieval quality

UI frontend for interactive usage