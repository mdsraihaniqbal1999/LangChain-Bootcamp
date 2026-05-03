# ============================================================
# Lesson: Building a Custom Tool (Ollama + Gemma)
# ============================================================

# This lesson explains how to create a custom LangChain tool
# and use it with a local LLM via Ollama (Gemma model).
#
# Instead of OpenAI, we will use:
# - Ollama
# - Gemma model (e.g., gemma:2b or gemma:7b)



# ============================================================
# Step 1: Install Dependencies
# ============================================================

# Run this in your terminal:
# pip install langchain langchain-community langchain-ollama


# ============================================================
# Step 2: Import Required Modules
# ============================================================

from langchain.tools import tool
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import OllamaLLM


# ============================================================
# Step 3: Define a Custom Tool
# ============================================================

# Create a tool using the @tool decorator
# This simulates fetching flight status

@tool
def GetFlightStatus(flight_no: str) -> str:
    """Gets flight status and schedule"""

    # NOTE:
    # In real-world applications, replace this with an API call

    return (
        f"Flight {flight_no} departed at 5:20 PM. "
        "It is on-time and expected to arrive at 8:10 PM at Gate B12."
    )


# ============================================================
# Step 4: Inspect Tool Metadata
# ============================================================

print("Tool Name:", GetFlightStatus.name)
print("Description:", GetFlightStatus.description)
print("Arguments:", GetFlightStatus.args)


# ============================================================
# Step 5: Initialize Ollama (Gemma Model)
# ============================================================

# Make sure you have pulled the model:
# ollama pull gemma:2b

llm = OllamaLLM(model="gemma4")


# ============================================================
# Step 6: Create Prompt Template
# ============================================================

prompt = PromptTemplate.from_template(
    "Based on the context: {context}\nAnswer the query: {query}"
)


# ============================================================
# Step 7: Output Parser
# ============================================================

output_parser = StrOutputParser()


# ============================================================
# Step 8: Generate Context Using Tool
# ============================================================

flight = "EK524"
context = GetFlightStatus.run(flight)

print("\nGenerated Context:")
print(context)


# ============================================================
# Step 9: Create Chain
# ============================================================

# Chain = Prompt -> LLM -> Output Parser
chain = prompt | llm | output_parser


# ============================================================
# Step 10: Invoke the Chain
# ============================================================

print("\nQuery: status")
print(chain.invoke({"context": context, "query": "status"}))

print("\nQuery: departure time")
print(chain.invoke({"context": context, "query": "departure time"}))

print("\nQuery: arrival time")
print(chain.invoke({"context": context, "query": "arrival time"}))

print("\nQuery: gate")
print(chain.invoke({"context": context, "query": "gate"}))


# ============================================================
# Step 11: How It Works
# ============================================================

# 1. Tool generates structured context
# 2. Prompt injects context into LLM
# 3. LLM extracts specific answers
# 4. Output parser formats the result


# ============================================================
# Step 12: Example Output
# ============================================================

# status -> On-time
# departure time -> 5:20 PM
# arrival time -> 8:10 PM
# gate -> B12


# ============================================================
# Step 13: Next Steps
# ============================================================

# - Replace mock function with real API
# - Add multiple tools
# - Use Agents for automatic tool usage
# - Combine with RAG pipelines
