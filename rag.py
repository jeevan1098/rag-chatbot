from dotenv import load_dotenv
import os
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# 1. Load ALL PDFs
print("Loading all PDFs from data/ folder...")
loader = DirectoryLoader(
    "data/",
    glob="**/*.pdf",
    loader_cls=PyPDFLoader
)
documents = loader.load()
pdf_names = list(set(os.path.basename(doc.metadata['source']) for doc in documents))
print(f"Loaded {len(documents)} pages from {len(pdf_names)} PDFs")

# 2. Chunk
print("Chunking...")
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100
)
chunks = splitter.split_documents(documents)
print(f"Created {len(chunks)} chunks total")

# 3. Embed + store
print("Embedding all chunks...")
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

# 5. Prompt
prompt = PromptTemplate.from_template("""
You are a helpful assistant. Answer the question using ONLY the context below.

Rules:
- Answer directly and precisely from the context.
- If the answer is NOT in the context, say: "I don't know based on the provided documents."
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
    formatted = []
    for doc in docs:
        source = os.path.basename(doc.metadata.get('source', 'unknown'))
        page = doc.metadata.get('page', '?')
        formatted.append(f"[Source: {source} | Page: {page}]\n{doc.page_content}")
    return "\n\n".join(formatted)

# 7. Chain that returns BOTH answer and source documents
rag_chain_with_sources = RunnableParallel(
    answer=( 
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    ),
    sources=retriever
)

# 8. Citation formatter
def format_citations(source_docs):
    seen = set()
    citations = []
    for doc in source_docs:
        source = os.path.basename(doc.metadata.get('source', 'unknown'))
        page = int(doc.metadata.get('page', 0)) + 1  # convert to human page number
        key = f"{source}-{page}"
        if key not in seen:
            seen.add(key)
            citations.append({
                "file": source,
                "page": page,
                "preview": doc.page_content[:120].replace('\n', ' ') + "..."
            })
    return citations

# 9. Chat loop
print(f"\n=== RAG Chatbot Ready! Loaded: {pdf_names} ===\n")

while True:
    question = input("You: ")
    if question.lower() == "quit":
        break

    print("\nBot:", end=" ", flush=True)
    
    result = rag_chain_with_sources.invoke(question)
    answer = result["answer"]
    citations = format_citations(result["sources"])
    
    print(answer)
    
    print("\n--- Sources & Citations ---")
    for i, c in enumerate(citations, 1):
        print(f"[{i}] {c['file']} — Page {c['page']}")
        print(f"    Preview: {c['preview']}")
    print()