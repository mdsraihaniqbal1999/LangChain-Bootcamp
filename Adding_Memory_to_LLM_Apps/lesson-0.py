# Demo: Adding Short-Term Memory in LangChain
# ------------------------------------------
# This example demonstrates how to add short-term memory (conversation history)
# to a chatbot using LangChain + Ollama (gemma4 model).
#
# You will learn:
# 1. Why stateless chat fails for follow-ups
# 2. How to add history using MessagesPlaceholder
# 3. How to build a memory-enabled chatbot


# Step 1: Stateless Prompt (No Memory)
# -----------------------------------
# The model does NOT remember previous messages

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM

model = OllamaLLM(model="gemma4")

prompt = ChatPromptTemplate.from_messages([
    ("system", "You're an assistant with expertise in {ability}."),
    ("human", "{input}"),
])

base_chain = prompt | model


# First call: establish context
response1 = base_chain.invoke({
    "ability": "math",
    "input": "What's a right-angled triangle?"
})
print(response1)


# Second call: follow-up WITHOUT memory
response2 = base_chain.invoke({
    "ability": "math",
    "input": "What are the other types?"
})
print(response2)

# Problem:
# The model may not understand the second question
# because it has NO memory of the first one


# Step 2: Add Memory using MessagesPlaceholder
# --------------------------------------------
from langchain_core.prompts import MessagesPlaceholder

prompt_with_memory = ChatPromptTemplate.from_messages([
    ("system", "You're an assistant with expertise in {ability}."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}"),
])

memory_chain = prompt_with_memory | model


# Step 3: Create Conversation History
# -----------------------------------
history = [
    ("human", "What's a right-angled triangle?"),
    ("ai", "A triangle with one 90° angle.")
]


# Step 4: Invoke with History
# ---------------------------
response = memory_chain.invoke({
    "ability": "math",
    "input": "What are the other types?",
    "history": history
})

print(response)

# Now the model understands context and responds correctly


# Step 5: Complete Memory-Enabled Chatbot
# ---------------------------------------

prompt_full = ChatPromptTemplate.from_messages([
    ("system", "You're an assistant who's good at {ability}."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}"),
])

chat_chain = prompt_full | model

history = [
    ("human", "What's a right-angled triangle?"),
    ("ai", "A triangle with one 90° angle.")
]

result = chat_chain.invoke({
    "ability": "math",
    "input": "What are the other types?",
    "history": history
})

print(result)


# Step 6: Add Configurable Behavior
# --------------------------------
# Example: limit response length

prompt_limited = ChatPromptTemplate.from_messages([
    ("system", "You're an assistant who's good at {ability}. Respond in 20 words or fewer."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}"),
])

limited_chain = prompt_limited | model

response = limited_chain.invoke({
    "ability": "math",
    "input": "What are the other types?",
    "history": history
})

print(response)


# Summary:
# --------
# - Stateless chains do not remember previous inputs
# - MessagesPlaceholder allows injecting conversation history
# - History is a list of (role, content) tuples
# - Enables multi-turn, context-aware conversations

