# LCEL Demo 3: Batching in LangChain
# ---------------------------------
# This example demonstrates batching, where multiple prompts are processed
# in parallel to improve performance and reduce total latency.

# Step 1: Import required components
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


# Step 2: Define the Prompt
# -------------------------
# Create a simple prompt template with a variable {question}

prompt = PromptTemplate(
    input_variables=["question"],
    template="Answer the following question concisely: {question}"
)


# Step 3: Initialize the LLM
# --------------------------
# Using Ollama with a local model

llm = OllamaLLM(model="gemma4")


# Step 4: Create Output Parser
# ----------------------------
# Converts model output into a clean string

output_parser = StrOutputParser()


# Step 5: Build the LCEL Chain
# ----------------------------
# Prompt -> LLM -> Output Parser

chain = prompt | llm | output_parser


# Step 6: Prepare Batch Inputs
# ----------------------------
# List of input dictionaries

questions = [
    {"question": "Tell me about The Godfather movie."},
    {"question": "Tell me about the Avatar movie."},
]


# Step 7: Run Batch Inference
# ---------------------------
# Executes all inputs in parallel

responses = chain.batch(questions)


# Step 8: Access Results
# ----------------------
# Each response corresponds to the respective input

print(responses[0])
print(responses[1])


# Notes on Batching:
# ------------------
# - batch() allows multiple inputs to be processed in a single call
# - LangChain handles parallel execution internally
# - Useful for tasks like summarization, Q&A, or bulk processing


# Performance Comparison:
# -----------------------
# Single Inference:
# - Multiple separate requests
# - Higher total latency
#
# Batched Inference:
# - Single combined request
# - Lower overall latency
# - Better throughput


# Key Benefits:
# -------------
# - Reduced total execution time
# - Efficient use of LLM resources
# - Scales well for high-volume workloads


# Next Steps:
# -----------
# - Runnable pass-through: insert custom logic between chain steps
# - Advanced LCEL: build complex and optimized pipelines