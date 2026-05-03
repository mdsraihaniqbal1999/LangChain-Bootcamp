# Lesson 1: Performing Retrieval with WebBaseLoader
# Import WebBaseLoader to fetch and extract text from web pages
from langchain_community.document_loaders import WebBaseLoader

# URL of the article about Meta's AI assistant and Llama 3
URL = "https://www.theverge.com/2024/4/18/24133808/meta-ai-assistant-llama-3-chatgpt-openai-rival"

# Create loader instance and fetch the web page
loader = WebBaseLoader(URL)
data = loader.load()  # Returns list of Document objects with page_content and metadata

# Print the full extracted content and metadata
print(data)

# Print number of Document objects (typically 1 per URL)
print(len(data))

# Print the first (and only) Document containing the web page content
print(data[0])
