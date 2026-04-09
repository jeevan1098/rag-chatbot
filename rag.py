from dotenv import load_dotenv
import os
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage

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

# 5. Prompt with chat history
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful assistant. Answer questions using ONLY the context below.

Rules:
- Answer directly and precisely from the context.
- If the answer is NOT in the context, say: "I don't know based on the provided documents."
- Never make up information.
- Keep answers concise and factual.
- If asked for a list, use bullet points.
- You remember the conversation history — use it for follow-up questions.

Context:
{context}"""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}")
])

# 6. Retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

def format_docs(docs):
    formatted = []
    for doc in docs:
        source = os.path.basename(doc.metadata.get('source', 'unknown'))
        page = doc.metadata.get('page', '?')
        formatted.append(f"[Source: {source} | Page: {page}]\n{doc.page_content}")
    return "\n\n".join(formatted)

# 7. Chain with memory
def format_citations(source_docs):
    seen = set()
    citations = []
    for doc in source_docs:
        source = os.path.basename(doc.metadata.get('source', 'unknown'))
        page = int(doc.metadata.get('page', 0)) + 1
        key = f"{source}-{page}"
        if key not in seen:
            seen.add(key)
            citations.append(f"  [{len(citations)+1}] {source} — Page {page}")
    return citations

# 8. Chat loop with memory
print(f"\n=== RAG Chatbot with Memory Ready! ===")
print(f"Loaded: {pdf_names}\n")
print("Try follow-up questions like 'tell me more about that' or 'which of those is best for AI?'\n")

chat_history = []  # stores conversation turns

while True:
    question = input("You: ")
    if question.lower() == "quit":
        break
    if question.lower() == "clear":
        chat_history = []
        print("Memory cleared!\n")
        continue

    # Get relevant docs
    docs = retriever.invoke(question)
    context = format_docs(docs)

    # Build chain
    chain = prompt | llm | StrOutputParser()

    # Run with history
    answer = chain.invoke({
        "context": context,
        "chat_history": chat_history,
        "question": question
    })

    print(f"\nBot: {answer}")

    # Show citations
    citations = format_citations(docs)
    if citations:
        print("\nSources:")
        for c in citations:
            print(c)

    print()

    # Save to memory
    chat_history.append(HumanMessage(content=question))
    chat_history.append(AIMessage(content=answer))

    # Keep last 6 turns (3 exchanges) to avoid context overflow
    if len(chat_history) > 6:
        chat_history = chat_history[-6:]