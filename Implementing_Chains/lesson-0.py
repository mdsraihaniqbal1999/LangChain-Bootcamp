# ============================================================
# Lesson: Using Document Chain (Stuff Chain) with Ollama (Gemma)
# ============================================================

# This script demonstrates how to use LangChain’s
# create_stuff_documents_chain with Ollama (Gemma model) to:
# 1. Load multiple web documents
# 2. Merge them into a single context
# 3. Query the LLM using that combined context
#
# This is ideal when:
# - You have a small number of documents
# - Total content fits within the model’s context window


# ============================================================
# Step 1: Install Dependencies (run in terminal)
# ============================================================
# pip install langchain langchain-community langchain-ollama


# ============================================================
# Step 2: Import Required Modules
# ============================================================
import os
os.environ["USER_AGENT"] = "rag-learning"

# Ollama chat model (Gemma)
from langchain_ollama import ChatOllama

# Prompt template
from langchain_core.prompts import ChatPromptTemplate

# Web loader
from langchain_community.document_loaders import WebBaseLoader

# Stuff document chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain


# ============================================================
# Step 3: Load Documents from URLs
# ============================================================

# TechCrunch article URLs
URL1 = "https://techcrunch.com/2024/02/27/microsoft-made-a-16-million-investment-in-mistral-ai/"
URL2 = "https://techcrunch.com/2024/03/28/ai21-labs-new-text-generating-ai-model-is-more-efficient-than-most/"

# Load both articles
loader = WebBaseLoader([URL1, URL2])
data = loader.load()

# Verify documents loaded
print("Number of documents loaded:", len(data))

# Preview first document
print("\nPreview of first document:")
print(data[0].page_content[:200])


# ============================================================
# Step 4: Build Prompt Template
# ============================================================

# Create a system prompt using merged context
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "Identify the models launched by Mistral AI and AI21 Labs:\n\n{context}"
    )
])


# ============================================================
# Step 5: Initialize LLM (Gemma via Ollama) and Create Chain
# ============================================================

# Initialize Gemma model running locally via Ollama
llm = ChatOllama(model="gemma4")

# Create Stuff Documents Chain
chain = create_stuff_documents_chain(llm, prompt)


# ============================================================
# Step 6: Invoke the Chain
# ============================================================

# Pass the documents as context
result = chain.invoke({"context": data})

# Print result
print("\nFinal Answer:")
print(result)
