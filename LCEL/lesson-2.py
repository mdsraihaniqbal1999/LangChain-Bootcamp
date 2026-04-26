# LCEL Demo 2: Chain Invocation Methods
# ------------------------------------
# This example demonstrates three primary ways to execute a chain in LangChain:
# 1. Synchronous invocation (invoke)
# 2. Streaming output (stream)
# 3. Batch processing (batch)

# Step 1: Import required components
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


# Step 2: Construct prompt, LLM, and parser
prompt = ChatPromptTemplate.from_template(
    "You are a helpful assistant.\nAnswer the following question: {question}"
)

llm = OllamaLLM(model="gemma4")

output_parser = StrOutputParser()


# Step 3: Build the chain
chain = prompt | llm | output_parser


# Step 4: Synchronous Invocation (invoke)
# ---------------------------------------
# Executes the entire chain and waits for the full response

result = chain.invoke({
    "question": "Tell me about The Godfather Movie"
})

print(result)

# Note:
# - invoke() blocks execution until the full response is generated
# - Suitable for simple or short tasks


# Step 5: Streaming Output (stream)
# --------------------------------
# Streams output token-by-token or chunk-by-chunk

for chunk in chain.stream({
    "question": "Tell me about The Godfather Movie"
}):
    print(chunk, end="")

# Note:
# - stream() yields partial outputs as they are generated
# - Useful for real-time display or large responses


# Step 6: Batch Processing (batch)
# --------------------------------
# Processes multiple inputs in parallel

questions = [
    {"question": "What is LangChain?"},
    {"question": "Explain chain-of-thought prompting."},
    {"question": "How does streaming work?"}
]

results = chain.batch(questions)

for response in results:
    print(response)

# Note:
# - batch() improves throughput by handling multiple inputs together
# - Ideal for high-volume or parallel workloads


# Summary:
# --------
# invoke -> synchronous execution, returns full result
# stream -> incremental output, yields chunks
# batch  -> parallel execution for multiple inputs


# What’s Next:
# ------------
# - Runnable Interface: create custom reusable components
# - Pass-through components: chain steps without modifying data