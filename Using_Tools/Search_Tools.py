# ============================================================
# Lesson: Using DuckDuckGo Search Tool in LangChain
# ============================================================

# This lesson explains how to use the DuckDuckGo search tool
# in LangChain to fetch real-time web results.
#
# DuckDuckGo is a privacy-focused search engine that can be
# used as a lightweight alternative to APIs like Tavily.
#
# It does NOT require an API key, making it ideal for learning.




# ============================================================
# Step 1: Install Dependencies
# ============================================================

# Run this in your terminal:
# pip install langchain langchain-community


# ============================================================
# Step 2: Import Required Modules
# ============================================================

from langchain_community.tools import DuckDuckGoSearchRun


# ============================================================
# Step 3: Initialize the DuckDuckGo Tool
# ============================================================

# This tool performs a simple web search and returns results
tool = DuckDuckGoSearchRun()


# ============================================================
# Step 4: Inspect Tool Metadata
# ============================================================

# Every LangChain tool has metadata

print("Tool Name:", tool.name)            # Example: "duckduckgo_search"
print("Description:", tool.description)  # What the tool does
print("Arguments:", tool.args)           # Expected input format


# ============================================================
# Step 5: Run a Search Query
# ============================================================

# Perform a search query
result = tool.run("When is ICC Men's T20 World Cup 2024 starting?")

print("\nSearch Results:")
print(result)


# ============================================================
# Step 6: Alternative (Structured Results)
# ============================================================

# If you want structured output (list of results),
# use DuckDuckGoSearchResults instead

from langchain_community.tools import DuckDuckGoSearchResults

structured_tool = DuckDuckGoSearchResults()

response = structured_tool.invoke({
    "query": "When is ICC Men's T20 World Cup 2024 starting?"
})

print("\nNumber of results:", len(response))
print("First result:", response[0])


# ============================================================
# Step 7: Response Structure
# ============================================================

# Each result is typically a dictionary with:
# - title: Title of the webpage
# - link: URL of the result
# - snippet/content: Short summary


# ============================================================
# Step 8: Common Use Cases
# ============================================================

# 1. Real-time Q&A systems
#    - Fetch live web data before answering
#
# 2. RAG + Search Hybrid
#    - Combine vector DB results + web search
#
# 3. Fact-checking pipelines
#    - Validate LLM responses with search results


# ============================================================
# Step 9: Best Practices
# ============================================================

# - Use structured results when building pipelines
# - Handle empty or irrelevant results
# - Avoid excessive repeated queries
#
# Example:
# if not response:
#     print("No results found")


# ============================================================
# Step 10: DuckDuckGo vs Tavily
# ============================================================

# DuckDuckGo:
# - No API key required
# - Simple and free
# - Less optimized for LLM workflows
#
# Tavily:
# - Requires API key
# - Optimized for LLM + RAG
# - Provides cleaner, structured context


# ============================================================
# Step 11: Next Steps
# ============================================================

# - Combine search results into LLM prompts
# - Use tools inside chains
# - Learn Agents (automatic tool selection)
# - Integrate multiple tools together
