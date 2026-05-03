# ============================================================
# Lesson: Using Retrieval Chain (RAG) with Chroma + Ollama (Gemma)
# ============================================================

# This is a COMPLETE working RAG pipeline optimized for:
# - Mac (M1/M2/M3)
# - Python 3.14
# - Ollama (local LLM)
#
# No FAISS required. Uses Chroma instead.


# ============================================================
# Step 1: Install Dependencies (run in terminal)
# ============================================================
# pip install langchain langchain-community langchain-ollama chromadb


# ============================================================
# Step 2: Import Required Modules
# ============================================================

from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama, OllamaEmbeddings

from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain

from langchain_community.vectorstores import Chroma

import os


# ============================================================
# Step 3: Fix USER_AGENT Warning
# ============================================================

os.environ["USER_AGENT"] = "rag-learning"


# ============================================================
# Step 4: Load Web Documents
# ============================================================

URL1 = "https://techcrunch.com/2024/03/04/anthropic-claims-its-new-models-beat-gpt-4/"
URL2 = "https://techcrunch.com/2024/03/28/ai21-labs-new-text-generating-ai-model-is-more-efficient-than-most/"

loader = WebBaseLoader([URL1, URL2])
documents = loader.load()

print("Documents loaded:", len(documents))


# ============================================================
# Step 5: Split Documents into Chunks
# ============================================================

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,     # slightly larger for better context
    chunk_overlap=50
)

chunks = text_splitter.split_documents(documents)

print("Total chunks created:", len(chunks))


# ============================================================
# Step 6: Create Embeddings (Ollama)
# ============================================================

# IMPORTANT:
# Gemma is NOT an embedding model
# Use a dedicated embedding model
embeddings = OllamaEmbeddings(model="nomic-embed-text")


# ============================================================
# Step 7: Create Chroma Vector Store (Persistent)
# ============================================================

# This will store vectors locally in ./chroma_db
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

# Create retriever
retriever = vectorstore.as_retriever()


# ============================================================
# Step 8: Create Prompt Template
# ============================================================

prompt_template = """
Answer the question {input} based ONLY on the context below:

<context>
{context}
</context>

If the answer is not found, say "I don't know."
"""

prompt = PromptTemplate.from_template(prompt_template)


# ============================================================
# Step 9: Initialize LLM (Gemma via Ollama)
# ============================================================

llm = ChatOllama(
    model="gemma4",
    temperature=0.0   # deterministic answers
)


# ============================================================
# Step 10: Build Chains
# ============================================================

# Combine retrieved documents into prompt
combine_docs_chain = create_stuff_documents_chain(llm, prompt)

# Create Retrieval Chain (RAG)
chain = create_retrieval_chain(retriever, combine_docs_chain)


# ============================================================
# Step 11: Run Queries
# ============================================================

queries = [
    "What models did Anthropic release?",
    "What is special about AI21 Labs model?",
    "Compare Anthropic and AI21 models",
    "What is the context window mentioned?"
]

for query in queries:
    print("\n==============================")
    print("Question:", query)

    result = chain.invoke({"input": query})

    answer = result.get("output") or result.get("answer") or result.get("text")

    print("Answer:", answer)


# ============================================================
# Step 12: Notes & Best Practices
# ============================================================

# WHY CHROMA?
# - Works perfectly on Mac (M3)
# - No installation issues like FAISS
# - Supports persistence (saved to disk)

# RAG PIPELINE FLOW:
# Query -> Retriever -> Relevant Chunks -> Prompt -> LLM -> Answer

# IMPORTANT TIPS:
# - Keep temperature low (0.0) for factual tasks
# - Tune chunk_size for better retrieval quality
# - Always use SAME embedding model for indexing + querying

# PERSISTENCE:
# - Data is saved in ./chroma_db
# - Next run can reuse it instead of recomputing embeddings

# WHEN TO USE THIS:
# - Question answering over documents
# - Multi-document reasoning
# - Reducing hallucinations

# LIMITATIONS:
# - Quality depends on chunking + embeddings
# - Web scraping may fail on some sites

# ============================================================