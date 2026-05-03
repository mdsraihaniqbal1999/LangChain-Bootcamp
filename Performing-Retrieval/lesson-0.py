# Lesson-0: Loading PDF's for RAG (Retrieval-Augmented Generation) Pipeline
# This script demonstrates how to load and split a PDF document into pages
# using LangChain's PyPDFLoader. The loaded pages will be used for embedding
# and retrieval in a RAG system.

# Import the PyPDFLoader class from LangChain's community document loaders
# PyPDFLoader is specifically designed to load PDF files and extract text content
from langchain_community.document_loaders import PyPDFLoader

# Create a loader instance for the PDF file named "RAG_Demo.pdf"
# The loader will read the PDF file from the current working directory
# Make sure the file exists in the same folder as this script
loader = PyPDFLoader("RAG_Demo.pdf")

# Load the PDF and split it into individual page documents
# The load_and_split() method performs two operations:
# 1. Load() - Extracts text content from each page of the PDF
# 2. Split() - Separates each page into its own Document object
# Each page becomes a separate Document with page_content and metadata
pages = loader.load_and_split()

# Print the entire list of page Document objects
# This will show metadata like source file path and page numbers
# Useful for debugging to verify that pages were loaded correctly
print(pages)

# Print the total number of pages loaded from the PDF
# The len() function counts how many Document objects are in the pages list
# This helps confirm that all pages were successfully extracted
print(f"Length of pages: {len(pages)}")

# Print the first 500 characters of the first page's text content
# The page_content attribute contains the extracted text from each page
# The [:500] slice limits output to 500 characters for readability
# This allows quick verification that text extraction worked properly
print(f"First page content: {pages[0].page_content[:500]}")



