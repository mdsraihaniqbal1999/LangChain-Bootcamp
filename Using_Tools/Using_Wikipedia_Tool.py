# ============================================================
# Lesson: Using Wikipedia Tool in LangChain
# ============================================================

# This lesson explains how to use the built-in Wikipedia tool
# in LangChain to fetch real-time summaries from Wikipedia.
#
# Tools allow LLM applications to:
# - Access external knowledge
# - Improve accuracy
# - Provide up-to-date information



# ============================================================
# Step 1: Install Dependencies
# ============================================================

# Run this in your terminal:
# pip install langchain langchain-community
# pip install wikipedia # Wikipedia API wrapper


# ============================================================
# Step 2: Import Required Modules
# ============================================================

from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper


# ============================================================
# Step 3: Configure Wikipedia API Wrapper
# ============================================================

# The wrapper controls how data is fetched from Wikipedia.
#
# Parameters:
# - top_k_results: Number of pages to fetch
# - doc_content_chars_max: Max characters per result

api_wrapper = WikipediaAPIWrapper(
    top_k_results=1,            # Fetch only top result
    doc_content_chars_max=1000  # Limit response length
)

# Initialize the Wikipedia tool
tool = WikipediaQueryRun(api_wrapper=api_wrapper)


# ============================================================
# Step 4: Inspect Tool Metadata
# ============================================================

# Every LangChain tool has metadata attributes

print("Tool Name:", tool.name)            # Example: "wikipedia"
print("Description:", tool.description)  # What the tool does
print("Arguments:", tool.args)           # Expected inputs


# ============================================================
# Step 5: Run the Wikipedia Tool
# ============================================================

# Query the tool with a topic
result = tool.run({"query": "Neural Network"})

print("\nWikipedia Result:")
print(result)


# ============================================================
# Step 6: How It Works
# ============================================================

# - The tool sends a request to Wikipedia
# - Fetches the most relevant page
# - Returns a summary (limited by char count)
#
# No API key is required for basic usage


# ============================================================
# Step 7: Common Use Cases
# ============================================================

# 1. LCEL Chains
#    - Wrap tool inside a chain for structured workflows
#
# 2. RAG Pipelines
#    - Use Wikipedia as an external knowledge source
#
# 3. Hybrid Q&A Systems
#    - Combine LLM responses + Wikipedia data
#
# Example:
# LLM doesn't know something → tool fetches → LLM answers


# ============================================================
# Step 8: Best Practices
# ============================================================

# - Handle empty responses (Wikipedia may not have results)
# - Avoid too many rapid requests (rate limits)
# - Limit response size to reduce noise
#
# Example:
# if not result:
#     print("No data found")


# ============================================================
# Step 9: Next Steps
# ============================================================

# - Build custom tools
# - Use multiple tools together
# - Learn Agents (automatic tool selection)
# - Integrate APIs (weather, finance, etc.)
