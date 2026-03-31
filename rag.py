from dotenv import load_dotenv
import os
import numpy as np
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# 1. Load PDF
print("Loading PDF...")
loader = PyPDFLoader("data/jeevan.pdf")
pages = loader.load()
print(f"Loaded {len(pages)} pages")

# 2. Chunk
print("Chunking...")
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100
)
chunks = splitter.split_documents(pages)
print(f"Created {len(chunks)} chunks")

# 3. Embed + store
print("Embedding...")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
vectorstore = FAISS.from_documents(chunks, embeddings)
vectorstore.save_local("faiss_index")
print("Vector store ready!")

# 4. LLM
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY")
)

# 5. Custom prompt — forces grounded answers
prompt = PromptTemplate.from_template("""
You are a helpful assistant. Answer the question using ONLY the context provided below.

Rules:
- If the answer is clearly in the context, answer it directly and precisely.
- If the answer is NOT in the context, say exactly: "I don't know based on the provided document."
- Never make up information.
- Keep answers concise and factual.
- If asked for a list, use bullet points.

Context:
{context}

Question: {question}

Answer:""")

# 6. Retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# 7. Chain
chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# 8. Chat loop
print("\n=== RAG Chatbot Ready! Type 'quit' to exit ===\n")
while True:
    question = input("You: ")
    if question.lower() == "quit":
        break

    print("\nBot:", end=" ", flush=True)
    answer = chain.invoke(question)
    print(answer)

    docs = retriever.invoke(question)
    seen = set()
    print("\nSources:")
    for doc in docs:
        page = doc.metadata.get('page', '?')
        source = doc.metadata.get('source', '?')
        key = f"{source}-{page}"
        if key not in seen:
            seen.add(key)
            print(f"  - Page {page} of {source}")
    print()
