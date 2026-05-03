# ============================================================
# Lesson 5: RAG with PDF using Ollama (Gemma 4)
# ============================================================

# This script demonstrates a full Retrieval-Augmented Generation (RAG) pipeline:
# 1. Load a PDF
# 2. Split it into chunks
# 3. Generate embeddings
# 4. Store them in a vector database (Chroma)
# 5. Retrieve relevant context
# 6. Pass context to an LLM (Gemma 4) to generate answers


# ============================================================
# Step 1: Import Required Modules
# ============================================================

# Load PDF documents
from langchain_community.document_loaders import PyPDFLoader

# Split text into chunks
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Ollama models (embedding + LLM)
from langchain_ollama import OllamaEmbeddings, ChatOllama

# Vector database
from langchain_community.vectorstores import Chroma

# Prompt and chaining utilities
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


# ============================================================
# Step 2: Load the PDF
# ============================================================

# Provide path to your PDF file
loader = PyPDFLoader("RAG_Demo.pdf")

# Load all pages from the PDF
pages = loader.load()


# ============================================================
# Step 3: Split into Chunks
# ============================================================

# Split the document into smaller chunks
# chunk_size: max characters per chunk
# chunk_overlap: overlap between chunks to preserve context
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

# Create chunks from pages
chunks = text_splitter.split_documents(pages)


# ============================================================
# Step 4: Generate Embeddings and Create Vector Store
# ============================================================

# Initialize embedding model (must be an embedding model, not Gemma)
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# Store chunks in Chroma vector database
# This automatically:
# - Converts text to embeddings
# - Stores vectors
# - Creates index for similarity search
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings
)


# ============================================================
# Step 5: Create Retriever
# ============================================================

# Convert vector store into a retriever
retriever = vectorstore.as_retriever()


# Helper function to combine retrieved documents into a single string
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# ============================================================
# Step 6: Define LLM and Prompt
# ============================================================

# Initialize LLM (Gemma 4 via Ollama)
llm = ChatOllama(model="gemma4")

# Prompt template to control LLM behavior
template = """
SYSTEM: You are a question-answering assistant.
Use only the provided context to answer.
If you don't know, say "I don't know."

QUESTION: {question}

CONTEXT:
{context}
"""

prompt = PromptTemplate.from_template(template)


# ============================================================
# Step 7: Build RAG Chain
# ============================================================

# Pipeline:
# question -> retriever -> format_docs -> prompt -> LLM -> output parser
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
# Step 8: Query the System
# ============================================================

# Sample questions based on your PDF

questions = [
    "Who is Samuel and what is his daily routine?",
    "What role does Elena play in Samuel's life?",
]

# Execute queries
for q in questions:
    print("\nQuestion:", q)
    answer = chain.invoke(q)
    print("Answer:", answer)