from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import os
import shutil

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()

# ── App setup ──────────────────────────────────────
app = FastAPI(
    title="RAG Chatbot API",
    description="Chat with your PDF documents using AI",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Global state ───────────────────────────────────
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY")
)

vectorstore = None
chat_history = []
uploaded_files = []

os.makedirs("data/uploads", exist_ok=True)

# ── Prompt ─────────────────────────────────────────
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful assistant. Answer questions using ONLY the context below.
Rules:
- Answer directly and precisely from the context.
- If the answer is NOT in the context, say: "I don't know based on the provided documents."
- Never make up information.
- Keep answers concise and factual.
- If asked for a list, use bullet points.

Context:
{context}"""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}")
])

# ── Helper functions ───────────────────────────────
def format_docs(docs):
    formatted = []
    for doc in docs:
        source = os.path.basename(doc.metadata.get('source', 'unknown'))
        page = doc.metadata.get('page', '?')
        formatted.append(f"[Source: {source} | Page: {page}]\n{doc.page_content}")
    return "\n\n".join(formatted)

def format_citations(docs):
    seen = set()
    citations = []
    for doc in docs:
        source = os.path.basename(doc.metadata.get('source', 'unknown'))
        page = int(doc.metadata.get('page', 0)) + 1
        key = f"{source}-{page}"
        if key not in seen:
            seen.add(key)
            citations.append({"file": source, "page": page})
    return citations

def index_pdf(filepath):
    global vectorstore
    loader = PyPDFLoader(filepath)
    pages = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    chunks = splitter.split_documents(pages)
    if vectorstore is None:
        vectorstore = FAISS.from_documents(chunks, embeddings)
    else:
        new_vs = FAISS.from_documents(chunks, embeddings)
        vectorstore.merge_from(new_vs)
    vectorstore.save_local("faiss_index")
    return len(chunks)

# ── Request models ─────────────────────────────────
class ChatRequest(BaseModel):
    question: str
    session_id: str = "default"

class ClearRequest(BaseModel):
    session_id: str = "default"

# ── Endpoints ──────────────────────────────────────

@app.get("/")
def root():
    return {"message": "RAG Chatbot API is running!"}

@app.get("/health")
def health():
    return {
        "status": "ok",
        "vectorstore_ready": vectorstore is not None,
        "uploaded_files": uploaded_files
    }

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files accepted")

    filepath = f"data/uploads/{file.filename}"
    with open(filepath, "wb") as f:
        shutil.copyfileobj(file.file, f)

    chunks_created = index_pdf(filepath)
    uploaded_files.append(file.filename)

    return {
        "message": f"{file.filename} uploaded and indexed successfully",
        "chunks_created": chunks_created,
        "total_files": len(uploaded_files)
    }

@app.post("/chat")
async def chat(req: ChatRequest):
    global chat_history

    if vectorstore is None:
        raise HTTPException(
            status_code=400,
            detail="No documents uploaded yet. Please upload a PDF first."
        )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    docs = retriever.invoke(req.question)
    context = format_docs(docs)

    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({
        "context": context,
        "chat_history": chat_history,
        "question": req.question
    })

    citations = format_citations(docs)

    chat_history.append(HumanMessage(content=req.question))
    chat_history.append(AIMessage(content=answer))
    if len(chat_history) > 6:
        chat_history = chat_history[-6:]

    return {
        "answer": answer,
        "sources": citations,
        "question": req.question
    }

@app.post("/clear")
def clear_memory(req: ClearRequest):
    global chat_history
    chat_history = []
    return {"message": "Memory cleared successfully"}

@app.get("/files")
def list_files():
    return {
        "uploaded_files": uploaded_files,
        "total": len(uploaded_files)
    }

# ── Run ────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)