# Step 1: Import required components
# ---------------------------------
# ChatPromptTemplate: used to define structured prompts
# OllamaLLM: local LLM interface (Ollama)
# StrOutputParser: converts model output into plain string

from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


# Step 2: Create the Prompt
# -------------------------
# Define a template with a placeholder variable {question}

prompt = ChatPromptTemplate.from_template(
    """You are a helpful assistant.
    Answer the following question: {question}"""
)


# Step 3: Initialize the LLM
# --------------------------
# Using Ollama with a local model

llm = OllamaLLM(model="gemma4")


# Step 4: Create Output Parser
# ----------------------------
# Converts LLM output into a clean string

output_parser = StrOutputParser()


# Step 5: Build the LCEL Chain
# ----------------------------
# Use pipe operator (|) to connect components:
# Prompt -> LLM -> Output Parser

chain = prompt | llm | output_parser


# Step 6: Invoke the Chain
# ------------------------
# Pass input as a dictionary matching prompt variables

response = chain.invoke({
    "question": "Tell me about The Godfather movie"
})

print(response)


# Step 7: Inspect Input Schema
# ----------------------------
# Shows expected input format for the chain

print(chain.input_schema.schema())


# Step 8: Inspect Output Schema
# -----------------------------
# Shows the format of the chain output

print(chain.output_schema.schema())


# Step 9: Inspect Individual Component Schemas
# --------------------------------------------

# 9.1 LLM Input Schema
print(llm.input_schema.schema())


# 9.2 LLM Output Schema
print(llm.output_schema.schema())


# Notes:
# ------
# - Ollama runs locally, so ensure the model (gemma4) is pulled and running
# - The chain enforces structured input/output contracts
# - Each component exposes schemas for validation and debugging