# LCEL Demo 6: Advanced Patterns (RunnableLambda + Chain Introspection)
# ---------------------------------------------------------------------
# This example demonstrates advanced LCEL usage:
# 1. Building a basic chain
# 2. Injecting custom Python logic using RunnableLambda
# 3. Computing metrics (e.g., text length)
# 4. Inspecting the internal chain graph using grandalf


# Step 1: Imports
# ----------------
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_ollama import OllamaLLM


# Step 2: Create Basic LCEL Chain
# -------------------------------
# Prompt -> LLM -> Output Parser

prompt = ChatPromptTemplate.from_template(
    "Give me a one-line description of {topic}"
)

llm = OllamaLLM(model="gemma4")
output_parser = StrOutputParser()

chain = prompt | llm | output_parser

result = chain.invoke({"topic": "AI"})
print(result)


# Step 3: Add Custom Transformation (RunnableLambda)
# --------------------------------------------------
# Convert output text to title case

def to_titlecase(text: str) -> str:
    return text.title()

chain_with_transform = (
    prompt
    | llm
    | output_parser
    | RunnableLambda(to_titlecase)
)

titlecased = chain_with_transform.invoke({"topic": "AI"})
print(titlecased)


# Step 4: Compute Metrics (Text Length)
# -------------------------------------
# Add another custom function to calculate length

def get_len(text: str) -> int:
    print(text)  # debug output
    return len(text)

chain_with_metrics = (
    prompt
    | llm
    | output_parser
    | RunnableLambda(to_titlecase)
    | RunnableLambda(get_len)
)

length = chain_with_metrics.invoke({"topic": "AI"})
print(length)


# Step 5: Inspect Chain Graph (requires grandalf)
# -----------------------------------------------
# Install:
# pip install grandalf

graph = chain_with_metrics.get_graph()

# Print raw graph object
print(graph)

# Print ASCII visualization of the chain
graph.print_ascii()


# Expected ASCII Structure:
# -------------------------
# +---------------------------+
# |       PromptInput         |
# +---------------------------+
# |           *               |
# +---------------------------+
# |   ChatPromptTemplate      |
# +---------------------------+
# |           *               |
# +---------------------------+
# |        OllamaLLM          |
# +---------------------------+
# |           *               |
# +---------------------------+
# |     StrOutputParser       |
# +---------------------------+
# |           *               |
# +---------------------------+
# | RunnableLambda(to_titlecase) |
# +---------------------------+
# |           *               |
# +---------------------------+
# |   RunnableLambda(get_len) |
# +---------------------------+


# Notes:
# ------
# - RunnableLambda allows injecting custom Python functions into LCEL pipelines
# - Useful for transformations, validation, logging, metrics, etc.
# - You can chain multiple RunnableLambda steps
# - get_graph() helps visualize and debug pipeline structure
# - grandalf enables ASCII visualization of execution flow


# Best Practices:
# ---------------
# - Keep custom functions pure (no side effects)
# - Avoid debug prints in production
# - Use small, reusable transformation functions
# - Use graph inspection for debugging complex pipelines