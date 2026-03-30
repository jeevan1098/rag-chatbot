from dotenv import load_dotenv
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import numpy as np

load_dotenv()

# 1. Load your PDF
print("Loading PDF...")
loader = PyPDFLoader("data/jeevan.pdf")
pages = loader.load()
print(f"Loaded {len(pages)} pages")

# 2. Chunk it
print("Chunking...")
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100
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
# DAY 3 EXPERIMENTS — understanding embeddings

# Experiment 1: See what a vector actually looks like
print("\n--- EXPERIMENT 1: What does an embedding look like? ---")
sample_text = "Jeevan has experience in machine learning and deep learning"
vector = embeddings.embed_query(sample_text)
print(f"Text: '{sample_text}'")
print(f"Vector length: {len(vector)} numbers")
print(f"First 10 numbers: {[round(x, 4) for x in vector[:10]]}")

# Experiment 2: Compare similar vs different texts
print("\n--- EXPERIMENT 2: Similar vs different meaning ---")
import numpy as np

text1 = "machine learning and artificial intelligence"
text2 = "deep learning and neural networks"
text3 = "cooking recipes and food ingredients"

vec1 = embeddings.embed_query(text1)
vec2 = embeddings.embed_query(text2)
vec3 = embeddings.embed_query(text3)

def cosine_similarity(a, b):
    a, b = np.array(a), np.array(b)
    return round(float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))), 4)

sim_12 = cosine_similarity(vec1, vec2)
sim_13 = cosine_similarity(vec1, vec3)

print(f"'{text1}'")
print(f"'{text2}'")
print(f"Similarity (ML vs DL): {sim_12}  <-- should be HIGH")
print(f"")
print(f"'{text1}'")
print(f"'{text3}'")
print(f"Similarity (ML vs cooking): {sim_13}  <-- should be LOW")

# Experiment 3: See what chunks get retrieved for a question
print("\n--- EXPERIMENT 3: What chunks does your question retrieve? ---")
question = "What are Jeevan's technical skills?"

temp_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
retrieved_docs = temp_retriever.invoke(question)

print(f"Question: '{question}'")
print(f"Top {len(retrieved_docs)} chunks retrieved:\n")
for i, doc in enumerate(retrieved_docs):
    print(f"Chunk {i+1} (page {doc.metadata.get('page', '?')}):")
    print(f"{doc.page_content[:200]}...")
    print()

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
