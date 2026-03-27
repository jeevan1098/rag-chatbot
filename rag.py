from dotenv import load_dotenv
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# 1. Load your PDF
print("Loading PDF...")
loader = PyPDFLoader("data/jeevan.pdf")
pages = loader.load()
print(f"Loaded {len(pages)} pages")

# 2. Chunk it
print("Chunking...")
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
chunks = splitter.split_documents(pages)
print(f"Created {len(chunks)} chunks")

# 3. Embed + store in FAISS
print("Embedding... (first time takes 1-2 mins)")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
vectorstore = FAISS.from_documents(chunks, embeddings)
vectorstore.save_local("faiss_index")
print("Vector store saved!")

# 4. Build RAG chain
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY")
)

prompt = PromptTemplate.from_template("""You are a helpful assistant. Use ONLY the context below to answer.
If the answer is not in the context, say "I don't know based on the provided document."

Context:
{context}

Question: {question}

Answer:""")

retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# 5. Chat loop
print("\n=== RAG Chatbot Ready! Type 'quit' to exit ===\n")
while True:
    question = input("You: ")
    if question.lower() == "quit":
        break

    print("\nBot: ", end="", flush=True)
    answer = chain.invoke(question)
    print(answer)

    docs = retriever.invoke(question)
    print("\nSources:")
    for doc in docs:
        print(f"  - Page {doc.metadata.get('page', '?')} of {doc.metadata.get('source', '?')}")
    print()
