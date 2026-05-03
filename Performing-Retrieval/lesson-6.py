# ============================================================
# Lesson 6: RAG with Web Pages using Ollama (Gemma 4)
# ============================================================

# This script demonstrates a complete RAG pipeline using a web page:
# 1. Load web content
# 2. Split into chunks
# 3. Generate embeddings
# 4. Store in vector database (Chroma)
# 5. Retrieve relevant context
# 6. Pass to LLM (Gemma 4) for answering


# ============================================================
# Step 1: Install Dependencies (run in terminal)
# ============================================================
# pip install langchain langchain-community langchain-ollama chromadb


# ============================================================
# Step 2: Import Required Modules
# ============================================================

# Web loader to fetch content from URL
from langchain_community.document_loaders import WebBaseLoader

# Text splitter for chunking
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Ollama embeddings and LLM
from langchain_ollama import OllamaEmbeddings, ChatOllama

# Vector database
from langchain_community.vectorstores import Chroma

# Prompt and chaining utilities
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


# ============================================================
# Step 3: Load Web Page
# ============================================================

# URL of the article to process
URL = "https://www.theverge.com/2024/4/18/24133808/meta-ai-assistant-llama-3-chatgpt-openai-rival"

# Initialize loader
loader = WebBaseLoader(URL)

# Load content from the web page
docs = loader.load()


# ============================================================
# Step 4: Split into Chunks
# ============================================================

# Split content into smaller chunks for better retrieval
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,      # size of each chunk
    chunk_overlap=50     # overlap to preserve context
)

chunks = text_splitter.split_documents(docs)


# ============================================================
# Step 5: Generate Embeddings and Store in Chroma
# ============================================================

# Initialize embedding model (must be embedding model, not Gemma)
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# Create vector store from chunks
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings
)

# Create retriever interface
retriever = vectorstore.as_retriever()


# ============================================================
# Step 6: Format Retrieved Documents
# ============================================================

# Combine retrieved chunks into a single context string
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# ============================================================
# Step 7: Define LLM and Prompt
# ============================================================

# Initialize LLM (Gemma 4 via Ollama)
llm = ChatOllama(model="gemma4")

# Prompt template to guide the model
template = """
SYSTEM: You are a question-answering assistant.
Use only the provided context to answer.
If you don't know, say "I don't know."

Question: {question}

Context:
{context}
"""

prompt = PromptTemplate.from_template(template)


# ============================================================
# Step 8: Build RAG Chain
# ============================================================

# Chain:
# question -> retriever -> format_docs -> prompt -> LLM -> output
chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)


# ============================================================
# Step 9: Query the System
# ============================================================

# Example queries based on the web article
questions = [
    "What is Meta AI assistant?",
    "What models are being compared in the article?",
    "What is Llama 3?",
    "How does Meta's AI compare to competitors?",
    "What is the size of the largest Llama 3 model?"
]

# Run queries
for q in questions:
    print("\nQuestion:", q)
    answer = chain.invoke(q)
    print("Answer:", answer)