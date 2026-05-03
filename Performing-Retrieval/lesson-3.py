# ============================================
# Lesson 3: Generating Embeddings using Ollama
# ============================================

# Import Ollama Embeddings from LangChain integration
# This class is used to convert text into vector representations
from langchain_ollama import OllamaEmbeddings

# Initialize the embedding model
# "nomic-embed-text" is a lightweight and efficient embedding model
# used to convert text into numerical vectors (required for RAG)
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# Sample documents (these act as our dataset)
# In real-world scenarios, this could be:
# - PDFs
# - Web pages
# - Database records
docs = [
    "Thrilling Finale Awaits: The Countdown to the Cricket World Cup Championship",
    "Global Giants Clash: Football World Cup Semi-Finals Set the Stage for Epic Showdowns",
    "Record Crowds and Unforgettable Moments: Highlights from the Cricket World Cup",
    "From Underdogs to Contenders: Football World Cup Surprises and Breakout Stars"
]

# Convert all documents into embeddings (vectors)
# Each document → one vector
# Output: List of vectors (list of list of floats)
embed_docs = embeddings.embed_documents(docs)

# Print number of embeddings generated
# Should be equal to number of documents
print(len(embed_docs))

# Print the first embedding vector
# This will be a long list of floating point numbers
# representing the semantic meaning of the text
print(embed_docs[0])