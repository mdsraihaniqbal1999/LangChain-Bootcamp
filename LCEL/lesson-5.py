# LCEL Demo 5: Multi-Stage Content Generation Pipeline
# ---------------------------------------------------
# This example demonstrates how to build a multi-step content generation
# workflow using LCEL (LangChain Expression Language).
#
# The pipeline performs:
# 1. Title generation
# 2. Outline creation
# 3. Blog writing (200 words)
# 4. Summary generation (for social media)
#
# Each stage is modular and can be independently modified or replaced.


# Step 1: Imports
# ---------------
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import OllamaLLM


# Step 2: Initialize shared components
# ------------------------------------
llm = OllamaLLM(model="gemma4")
parser = StrOutputParser()


# Step 3: Define Individual Chains
# --------------------------------

# 3.1 Title Chain
# Generates a title from a given topic
title_chain = (
    ChatPromptTemplate.from_template(
        "Generate an impactful blog title for: {input}"
    )
    | llm
    | parser
    | {"title": RunnablePassthrough()}
)


# 3.2 Outline Chain
# Creates a structured outline from the generated title
outline_chain = (
    ChatPromptTemplate.from_template(
        "Generate a detailed outline for: {title}"
    )
    | llm
    | parser
    | {"outline": RunnablePassthrough()}
)


# 3.3 Blog Chain
# Writes a 200-word blog using the outline
blog_chain = (
    ChatPromptTemplate.from_template(
        "Write a 200-word blog post based on this outline:\n{outline}"
    )
    | llm
    | parser
    | {"blog": RunnablePassthrough()}
)


# 3.4 Summary Chain
# Generates a short summary from the blog content
summary_chain = (
    ChatPromptTemplate.from_template(
        "Generate a concise social media summary for this blog:\n{blog}"
    )
    | llm
    | parser
)


# Step 4: Compose Full Content Chain
# ----------------------------------
# Output of one stage becomes input to the next
content_chain = title_chain | outline_chain | blog_chain | summary_chain


# Step 5: Invoke the Chain
# ------------------------
result = content_chain.invoke({
    "input": "The impact of AI on jobs"
})

print(result)


# Notes:
# ------
# - Each stage returns a dictionary with a specific key (title, outline, blog)
# - RunnablePassthrough helps pass outputs forward while preserving structure
# - The final output is a summary string
#
# - This pipeline makes multiple LLM calls internally
# - Execution time will be higher than a single prompt
#
# - You can customize models per stage if needed:
#   title_chain -> fast model
#   blog_chain  -> high-quality model
#   summary     -> lightweight model


# Example customization:
# ----------------------
# You can replace llm in any stage with a different model instance
# for better performance or cost optimization
#
# Example:
# title_chain = prompt | fast_llm | parser | {"title": RunnablePassthrough()}
#
# Then rebuild:
# content_chain = title_chain | outline_chain | blog_chain | summary_chain