# LCEL Demo: Dealing with Memory (Redis-Based Persistent Memory)
# --------------------------------------------------------------
# This example demonstrates how to persist chat history using Redis
# to enable long-term memory across sessions in a LangChain application.
#
# You will learn:
# 1. How to build a base chain (Prompt + LLM)
# 2. How to store chat history in Redis
# 3. How to manage multiple sessions (multi-user / multi-topic)
# 4. How to resume conversations after restart


# Step 1: Imports
# ----------------
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama import OllamaLLM
from langchain_community.chat_message_histories import RedisChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory


# Step 2: Base Chain Setup (Prompt + LLM)
# ---------------------------------------
model = OllamaLLM(model="gemma4")

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You’re an assistant skilled in {ability}. Keep responses under 20 words.",
    ),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}"),
])

base_chain = prompt | model


# Step 3: Configure Redis Memory
# ------------------------------
# Make sure Redis is running locally on port 6379

REDIS_URL = "redis://localhost:6379/0"

def get_message_history(session_id: str):
    return RedisChatMessageHistory(session_id=session_id, url=REDIS_URL)


# Step 4: Wrap Chain with Message History
# ---------------------------------------
redis_chain = RunnableWithMessageHistory(
    base_chain,
    get_message_history,
    input_messages_key="input",
    history_messages_key="history",
)


# Step 5: Run Conversation Threads
# --------------------------------

# 5.1 Math Thread (Session 1)
redis_chain.invoke(
    {"ability": "math", "input": "What does cosine mean?"},
    config={"configurable": {"session_id": "math-thread1"}},
)

redis_chain.invoke(
    {"ability": "math", "input": "Tell me more!"},
    config={"configurable": {"session_id": "math-thread1"}},
)

# The second call uses stored history from Redis


# 5.2 Physics Thread (Session 2)
redis_chain.invoke(
    {"ability": "physics", "input": "What is the theory of relativity?"},
    config={"configurable": {"session_id": "phy-thread1"}},
)

redis_chain.invoke(
    {"ability": "physics", "input": "Tell me more!"},
    config={"configurable": {"session_id": "phy-thread1"}},
)

# Separate session → separate memory


# Step 6: Inspect Stored Data in Redis (Manual)
# ---------------------------------------------
# Run these commands in terminal:

# docker ps
# docker exec -it <container_id> sh
# redis-cli

# List all keys:
# KEYS *

# View a session history:
# LRANGE message_store:math-thread1 0 -1
# LRANGE message_store:phy-thread1 0 -1


# Step 7: Resume Conversations After Restart
# ------------------------------------------
# Even after restarting your app, Redis retains history

redis_chain.invoke(
    {"ability": "math", "input": "Tell me more!"},
    config={"configurable": {"session_id": "math-thread1"}},
)

redis_chain.invoke(
    {"ability": "physics", "input": "Tell me more!"},
    config={"configurable": {"session_id": "phy-thread1"}},
)

# You can even switch session contexts dynamically:

redis_chain.invoke(
    {"ability": "math", "input": "Tell me more!"},
    config={"configurable": {"session_id": "phy-thread1"}},
)
# This will load physics history instead of math


# Summary:
# --------
# - Short-term memory = stored in Python (lost on restart)
# - Long-term memory = stored in Redis (persistent)
# - RunnableWithMessageHistory automatically:
#     * loads history before execution
#     * saves messages after execution
# - session_id separates different users or conversations
# - Enables multi-session, multi-tenant chat systems


# When to Use Redis Memory:
# -------------------------
# - Chatbots with user sessions
# - Multi-user applications
# - Systems requiring persistence across restarts
# - Production-grade LLM applications

# Step 1: Open a terminal and start Redis server
# redis-server
# → This starts the Redis server (should show "Ready to accept connections")

# Step 2: Open a new terminal window and start Redis CLI
# redis-cli
# → This opens the Redis command line interface

# Step 3: Check if any keys (stored sessions) exist
# KEYS *
# → Lists all keys (e.g., message_store:math-thread1)

# Step 4: View all messages in a specific session
# LRANGE message_store:math-thread1 0 -1
# → Shows full chat history for that session

# Step 5: View messages for another session
# LRANGE message_store:phy-thread1 0 -1
# → Shows chat history for physics session

# Step 6: Count number of messages in a session
# LLEN message_store:math-thread1

# Step 7: View only the latest message
# LRANGE message_store:math-thread1 -1 -1

# Step 8: Delete a specific session
# DEL message_store:math-thread1

# Step 9: Clear all data in Redis (use carefully)
# FLUSHALL