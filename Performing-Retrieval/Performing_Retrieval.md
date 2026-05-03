# 📘 Context Window & RAG Explained

## 🔹 What is Context Window?

A **context window** refers to the total amount of information a language model can process at once.

```
Context Window = Input (Prompt + History) + Output (Generated Response)
```

### Flow:

```
User → Application → (Prompt + Previous Responses) → Language Model
```

* The model uses this **window of context** to generate accurate responses.
* It includes:

  * Current prompt
  * Previous conversation (history)
  * Retrieved external data (if used)

---

## 🔹 What is Context?

**Context** is additional information provided to the model to improve accuracy.

### Sources of Context:

* Databases
* Documents
* APIs
* Web Search

```
Context → External Data Sources → Injected into Prompt → LLM
```

### Why Context is Important?

* Reduces hallucinations
* Improves factual accuracy
* Provides up-to-date information

---

## 🔹 What is Hallucination?

**Hallucination** occurs when a language model generates:

* Incorrect
* Fabricated
* Misleading information

👉 This usually happens due to **lack of proper context**.

---

## 🔹 What is RAG?

**RAG (Retrieval-Augmented Generation)** is a framework used to **inject external context into prompts** before sending them to the LLM.

```
RAG = Retrieval + Augmented Generation
```

### Key Benefits:

* Explainability
* Transparency
* Access Control
* Data Privacy

---

## 🔹 Why RAG is Needed?

* Every LLM has **limitations in context window size**
* Cannot store or remember all data
* RAG helps by retrieving **only relevant information**

---

# 🔄 RAG Workflow

```
User → Chatbot → LLM
             ↑     ↓
        Retrieval System
             ↓
     Databases / Documents
```

### Detailed Flow:

```
User → Prompt → Chatbot
                ↓
              LLM
                ↓
        Search / Retrieval
                ↓
     Databases & Documents
                ↓
      Context Returned to LLM
                ↓
        Final Response
```

---

# ⚙️ RAG Phase 1: Indexing (Data Preparation)

This phase prepares data for efficient retrieval.

## Steps:

```
Load → Split → Embed → Store (Vector Database)
```

### 1. Load

* Load documents using tools (e.g., document loaders)

### 2. Split

* Break large text into smaller chunks
* Done using **text splitters**

### 3. Embed

* Convert text into vectors using embedding models

```
Text → Embedding Model → Vector (Numerical Representation)
```

### 4. Store

* Store vectors in a **vector database**

### 5. Indexing

* Organize vectors for fast retrieval

---

## ✅ Example (Simple)

Suppose you have a document:

```
"Python is a programming language used for AI and web development."
```

### Process:

```
Load:
  Document loaded

Split:
  "Python is a programming language"
  "Used for AI and web development"

Embed:
  Each chunk → Converted into vectors

Store:
  Saved in vector database
```

---

# 🔍 RAG Phase 2: Retrieval

This phase retrieves relevant data during a query.

## Steps:

```
Query → Convert to Vector → Retrieve → Add to Prompt → LLM → Response
```

### Detailed Flow:

```
User Query
    ↓
Convert Query → Vector
    ↓
Semantic Search (Vector Database)
    ↓
Retrieve Relevant Chunks
    ↓
Augment Prompt with Context
    ↓
Send to LLM
    ↓
Final Answer
```

---

## 🔹 What is Semantic Search?

* Searches based on **meaning**, not keywords
* Uses vector similarity

```
Query Vector ≈ Document Vectors → Best Matches Retrieved
```

---

## 🔹 Tools (Example: LangChain)

* Document Loaders → Load data
* Text Splitters → Create chunks
* Embedding Models → Generate vectors
* Vector Stores → Store and retrieve data

---

# 🧠 Summary

* Context improves LLM accuracy
* Context window has size limits
* RAG helps overcome these limits
* RAG works in two phases:

  * **Indexing (offline)**
  * **Retrieval (runtime)**

---

If you want, I can also convert this into:

* a **visual diagram (image)**
* or a **Notion-style doc**
* or even a **Medium blog version**
