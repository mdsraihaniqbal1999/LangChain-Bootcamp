# ============================================================
# Lesson 4: Vector Stores & Semantic Search using Chroma
# ============================================================

# Step 1: Import Dependencies
# OllamaEmbeddings → converts text into vectors
# Chroma → vector database to store and retrieve embeddings
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

# Initialize the embedding model
# "nomic-embed-text" is an Ollama embedding model
# It converts text into high-dimensional vectors
embeddings = OllamaEmbeddings(model="nomic-embed-text")


# Step 2: Prepare Documents
# These are sample documents (sports headlines)
# In real-world use cases, data can come from:
# - PDFs
# - APIs
# - Databases
docs = [
    "Thrilling Finale Awaits: The Countdown to the Cricket World Cup Championship",
    "Global Giants Clash: Football World Cup Semi-Finals Set the Stage for Epic Showdowns",
    "Record Crowds and Unforgettable Moments: Highlights from the Cricket World Cup",
    "From Underdogs to Contenders: Football World Cup Surprises and Breakout Stars"
]


# Step 3: Create a Vector Store (Chroma)
# This does 3 things internally:
# 1. Converts text → embeddings
# 2. Stores embeddings in a vector database
# 3. Creates an index for fast similarity search
vectorstore = Chroma.from_texts(
    texts=docs,
    embedding=embeddings
)


# ============================================================
# Step 4: Perform Semantic Search
# ============================================================

# Query 1: Cricket-related search
# Even though "Rohit Sharma" is NOT in the documents,
# semantic search will return cricket-related content
results_cricket = vectorstore.similarity_search("Rohit Sharma", k=2)

print("\nCricket Query Results:")
for doc in results_cricket:
    print(doc.page_content)


# Expected Output (approximate):
# Record Crowds and Unforgettable Moments: Highlights from the Cricket World Cup
# Thrilling Finale Awaits: The Countdown to the Cricket World Cup Championship


# Query 2: Football-related search
# "Lionel Messi" is also not explicitly present,
# but semantic similarity helps retrieve relevant documents
results_football = vectorstore.similarity_search("Lionel Messi", k=2)

print("\nFootball Query Results:")
for doc in results_football:
    print(doc.page_content)


# Expected Output (approximate):
# From Underdogs to Contenders: Football World Cup Surprises and Breakout Stars
# Global Giants Clash: Football World Cup Semi-Finals Set the Stage for Epic Showdowns


# ============================================================
# Key Concepts Explained
# ============================================================

# 1. Embeddings:
#    - Converts text into numerical vectors
#    - Similar meaning → similar vectors

# 2. Vector Database (Chroma):
#    - Stores embeddings
#    - Performs fast similarity search

# 3. Semantic Search:
#    - Searches by meaning, not keywords
#    - Example:
#        Query: "Rohit Sharma"
#        Result: Cricket-related headlines (even without exact match)

# 4. Similarity Search:
#    - Compares query vector with stored vectors
#    - Returns top-k closest matches

# 5. Parameter k:
#    - k=2 → returns top 2 results
#    - k=1 → returns best match only


# ============================================================
# How It Works Internally
# ============================================================

# Documents → Embedding Model → Vectors → Stored in Chroma
# Query → Embedding Model → Query Vector
# Query Vector ≈ Document Vectors → Top-K Matches Returned