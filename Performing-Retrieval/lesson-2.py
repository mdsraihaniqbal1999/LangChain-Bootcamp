# Lesson: Chunking Documents - Loading PDF and recursive splitting for semantic search

# CORRECT IMPORTS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. Load PDF and split into pages
loader = PyPDFLoader("RAG_Demo.pdf")
pages = loader.load_and_split()
print(len(pages))
print(pages[0].page_content)

# 2. Configure recursive character splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,      # Max characters per chunk
    chunk_overlap=50     # Overlap between chunks to preserve context
)

# 3. Split pages into smaller chunks for embedding
chunks = text_splitter.split_documents(pages)
print(len(chunks))

# 4. Inspect chunks
print(chunks[0])  # First chunk with content and metadata
print(chunks[1])  # Second chunk with overlap