# Lesson: Understanding and Using Tools in LangChain

This lesson explains how **LangChain Tools** extend LLM capabilities by enabling **real-time data access and external integrations**.

---

## 📚 Documentation Index
Fetch the complete documentation index here:  
https://notes.kodekloud.com/llms.txt  

Use this file to explore all available pages before diving deeper.

---

## 🧠 What You’ll Learn

In this lesson, you will understand:

- What **Tools** are in LangChain
- How they differ from **RAG (Retrieval-Augmented Generation)**
- When to use **Tools vs RAG**
- Real-world usage patterns

---

# 🔧 What Is a Tool?

A **Tool** in LangChain is a wrapper around external functionality such as:

- APIs (weather, stock, flight data)
- Custom Python functions
- External services (databases, SaaS APIs)
- Runtime environments (Python REPL)

### ✅ Tools Enable:

- Real-time data retrieval
- Dynamic computation
- Integration with third-party services

---

# ✈️ Real-World Example: Airline Chatbot

Let’s break this down using a practical example:

## 📦 Baggage Policy (RAG)

- Stored in a PDF
- Already indexed in a vector database
- Retrieved using semantic search

👉 Use **RAG**

---

## 🛫 Flight Arrival Time (Tool)

- Not stored in documents
- Changes in real-time
- Requires live API call

👉 Use **Tool (API call)**

---

# ⚡ Why Tools Matter

Without tools:
- LLMs rely only on training data or indexed documents

With tools:
- LLMs can fetch **live, up-to-date information**
- Perform **real computations**
- Interact with **external systems**

---

# 🧰 Built-In Tools in LangChain

LangChain provides several ready-to-use tools:

- **Wikipedia** → Fetch summaries or full articles  
- **Search** → Perform live web searches  
- **YouTube** → Extract and summarize transcripts  
- **Python REPL** → Execute Python code  
- **Custom Functions** → Wrap your own APIs  

---

# 🤖 Tools + Agents

Tools become powerful when combined with **Agents**:

- Agents decide:
  - Which tool to use
  - When to use it
  - How to combine results

👉 Example:
- User asks: *"What’s the weather in Bangalore and convert it to Fahrenheit?"*
- Agent:
  1. Calls weather API
  2. Uses Python tool for conversion
  3. Returns final answer

---

# ⚖️ When to Use RAG vs Tools

| Capability            | RAG (Pre-Indexed)                     | Tools (Real-Time)                    |
|----------------------|--------------------------------------|-------------------------------------|
| Data Source          | Static docs (PDFs, articles)         | Live APIs, services                 |
| Latency              | Batch (preprocessed)                 | Real-time                           |
| Use Cases            | FAQs, policies, historical data      | Flight status, stock prices         |
| Complexity           | Vector DB + retriever + LLM          | API + tool wrapper + LLM            |

---

## 🚨 Key Rule

> ❌ Don’t use RAG for real-time data  
> ✅ Use Tools for live information

---

# 🔄 Putting It All Together

### Airline Chatbot Architecture:

- **Baggage Policy** → RAG  
- **Flight Status** → Tool (API call)

👉 This hybrid approach is how real-world AI systems are built.

---

# 🚀 Key Takeaways

- Tools extend LLMs beyond static knowledge
- RAG is best for **stored, indexed data**
- Tools are best for **live, dynamic data**
- Real applications use **both together**

---

# 🔜 What’s Next?

In upcoming lessons, you’ll learn:

- How to implement **LangChain tools**
- How to build **agents**
- How to orchestrate multiple tools together

This is where LLM apps become truly powerful.