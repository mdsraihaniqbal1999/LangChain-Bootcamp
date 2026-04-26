# LCEL Demo 4: RunnablePassthrough
# --------------------------------
# This example demonstrates how to use RunnablePassthrough in LangChain pipelines
# for forwarding data unchanged and modifying inputs dynamically using assign().

# Step 1: Import required components
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


# Step 2: Define the Prompt
# -------------------------
# Prompt expects two inputs: topic and question

prompt = ChatPromptTemplate.from_template(
    "You are a helpful assistant on {topic}.\n"
    "Answer the following question: {question}"
)


# Step 3: Initialize LLM and Parser
llm = OllamaLLM(model="gemma4")
output_parser = StrOutputParser()


# Step 4: Basic Chain (Prompt -> LLM -> Parser)
# --------------------------------------------

chain = prompt | llm | output_parser

response = chain.invoke({
    "topic": "movies",
    "question": "Tell me about The Godfather movie"
})

print(response)


# Step 5: Add RunnablePassthrough (No-Op)
# ---------------------------------------
# Passthrough forwards input/output unchanged

# At the beginning
chain1 = RunnablePassthrough() | prompt | llm | output_parser
print(chain1.invoke({
    "topic": "movies",
    "question": "Tell me about The Godfather movie"
}))

# At the end
chain2 = prompt | llm | output_parser | RunnablePassthrough()
print(chain2.invoke({
    "topic": "movies",
    "question": "Tell me about The Godfather movie"
}))


# Step 6: Verify Output with Passthrough
# --------------------------------------

chain_end = prompt | llm | output_parser | RunnablePassthrough()

result = chain_end.invoke({
    "topic": "movies",
    "question": "Tell me about The Godfather movie"
})

print(result)


# Step 7: Inject / Override Inputs using assign()
# -----------------------------------------------
# assign() allows modifying or adding keys in the input dictionary

new_chain = (
    RunnablePassthrough() |
    RunnablePassthrough.assign(topic=lambda _: "movies") |
    prompt |
    llm |
    output_parser
)

# Now only 'question' is required
response = new_chain.invoke({
    "question": "Tell me about Inception"
})

print(response)


# Step 8: Test assign in isolation
# --------------------------------

test_chain = (
    RunnablePassthrough() |
    RunnablePassthrough.assign(topic=lambda _: "movies")
)

result = test_chain.invoke({
    "question": "Tell me about Inception"
})

print(result)

# Expected Output:
# {'question': 'Tell me about Inception', 'topic': 'movies'}


# Common assign patterns:
# -----------------------
# Static value injection:
# RunnablePassthrough.assign(topic=lambda _: "news")
#
# Dynamic value injection:
# RunnablePassthrough.assign(timestamp=lambda _: datetime.utcnow().isoformat())
#
# Multiple assignments:
# RunnablePassthrough.assign(a=lambda _: 1).assign(b=lambda _: 2)


# Summary:
# --------
# - RunnablePassthrough acts as a no-op (passes data unchanged)
# - Useful for debugging, logging, or reserving pipeline steps
# - assign() enables dynamic modification of inputs
# - Helps build flexible, clean, and testable pipelines