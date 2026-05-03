# ============================================================
# Lesson: Using Yahoo Finance News Tool in LangChain
# ============================================================

# This lesson explains how to use the Yahoo Finance News Tool
# in LangChain to fetch real-time stock news and market insights.
#
# This tool is useful for:
# - Financial applications
# - Trading assistants
# - Market analysis systems


# ============================================================
# Step 1: Install Dependencies
# ============================================================

# Run this in your terminal:
# pip install langchain langchain-community
# pip3 install yfinance


# ============================================================
# Step 2: Import Required Module
# ============================================================

from langchain_community.tools.yahoo_finance_news import YahooFinanceNewsTool


# ============================================================
# Step 3: Initialize the Tool
# ============================================================

# Create an instance of the Yahoo Finance News tool
tool = YahooFinanceNewsTool()


# ============================================================
# Step 4: Inspect Tool Metadata
# ============================================================

# Like all LangChain tools, this tool exposes metadata

print("Tool Name:", tool.name)            # Example: "yahoo_finance_news"
print("Description:", tool.description)  # What the tool does
print("Arguments:", tool.args)           # Expected input


# ============================================================
# Step 5: Fetch News for a Stock Symbol
# ============================================================

# Provide a stock ticker symbol (e.g., NVDA, AAPL, TSLA)
result = tool.run("AAPL")

print("\nYahoo Finance News:")
print(result)


# ============================================================
# Step 6: How It Works
# ============================================================

# - The tool sends a request to Yahoo Finance
# - Scrapes the news section for the given stock
# - Returns:
#     - Latest headlines
#     - Brief summary of stock performance
#     - Key financial insights


# ============================================================
# Step 7: Example Output (Simplified)
# ============================================================

# Nvidia (NVDA) Rises But Trails Market
# NVDA closed at $877.57 with a +0.03% change...


# ============================================================
# Step 8: Common Use Cases
# ============================================================

# 1. Agent Workflows
#    - Agents fetch latest news before making decisions
#
# 2. Financial Dashboards
#    - Display live stock news in UI
#
# 3. Trading Systems
#    - Combine news with signals for strategy building


# ============================================================
# Step 9: Best Practices
# ============================================================

# - Avoid too many rapid requests (rate limiting risk)
# - Handle empty responses gracefully
# - Combine with LLMs for summarization
#
# Example:
# if not result:
#     print("No news found for this symbol")


# ============================================================
# Step 10: Notes
# ============================================================

# - No API key required
# - Uses live scraping from Yahoo Finance
# - Internet connection required


# ============================================================
# Step 11: Next Steps
# ============================================================

# - Combine with other tools (search, calculator, etc.)
# - Use inside LangChain agents
# - Build finance assistants with real-time insights
