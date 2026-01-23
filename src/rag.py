from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from src.config import settings

PROMPT = ChatPromptTemplate.from_template(
"""Answer the question using ONLY the context below.
If the answer is not in the context, say "I don't know".

Context:
{context}

Question:
{question}
"""
)

def answer(vectorstore, question: str, top_k: int = 3):
    retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})
    docs = retriever.get_relevant_documents(question)

    context = "\n\n".join(d.page_content for d in docs)
    messages = PROMPT.format_messages(context=context, question=question)

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        openai_api_key=settings.openai_api_key
    )
    resp = llm.invoke(messages)

    sources = []
    for d in docs:
        meta = d.metadata or {}
        sources.append({
            "source": meta.get("source"),
            "page": meta.get("page"),
            "snippet": d.page_content[:350]
        })

    return resp.content, sources
