````markdown
# Lesson: Overview of Chains

This lesson explores **LangChain’s high-level chain constructs** for handling common workflows like summarization and retrieval.

---

## 📚 Documentation Index
Fetch the complete documentation index here:  
https://notes.kodekloud.com/llms.txt  

Use this to explore all available pages before diving deeper.

---

## 🧠 What You’ll Learn

In this lesson, we cover two important built-in chain types:

- **Summarization**
  - Using `create_stuff_documents_chain`
- **Retrieval (RAG)**
  - Using `create_retrieval_chain`

These abstractions help you process multiple documents efficiently without building pipelines from scratch.

---

# 1. Summarization with Create Stuff Documents Chain

The **Stuff Documents Chain** works by:

- Combining multiple document chunks into a single prompt
- Sending that prompt to the LLM
- Generating a unified response (e.g., summary)

### ✅ Best Use Cases

- Summarizing multiple documents in one go
- Extracting insights across all inputs
- When data fits within the model’s context window

---

## 🧪 Example: Batch Summarization

```python
from langchain.chains import create_stuff_documents_chain
from langchain.llms import OpenAI
from langchain.schema import Document

# Initialize LLM
llm = OpenAI(model_name="gpt-4")

# Prepare document chunks
docs = [
    Document(page_content="Document text chunk 1..."),
    Document(page_content="Document text chunk 2..."),
]

# Create the chain
chain = create_stuff_documents_chain(llm=llm)

# Run summarization
summary = chain.run(input_documents=docs)

print("Summary:", summary)
````

### ⚠️ Important Note

* Ensure total tokens across all documents **do not exceed the model’s context window**
* You can use `tiktoken` to estimate token usage beforehand

---

# 2. Retrieval with Create Retrieval Chain (RAG)

When your data is too large to fit into a single prompt, use a **Retrieval Chain**.

### 🔄 How It Works

1. **Retriever** → Finds relevant document chunks
2. **Document Chain** → Formats them into a prompt
3. **LLM** → Generates the final answer

This is essentially a **basic RAG (Retrieval-Augmented Generation)** pipeline.

---

## 🧪 Example: Simple RAG Pipeline

```python
from langchain.chains import create_retrieval_chain
from langchain.llms import OpenAI
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings

# Initialize embeddings
embeddings = OpenAIEmbeddings()

# Create vector store
vectorstore = FAISS.from_texts(
    ["Doc text A", "Doc text B"],
    embedding=embeddings
)

# Build retrieval chain
retrieval_chain = create_retrieval_chain(
    llm=OpenAI(model_name="gpt-4"),
    retriever=vectorstore.as_retriever()
)

# Query the system
query = "What are the key takeaways from these documents?"
result = retrieval_chain.run(query)

print("Answer:", result)
```

### ⚠️ Important Note

* Always use the **same embedding model** during:

  * Document indexing
  * Query time
* This ensures consistent vector similarity results

---

# 📊 Comparison of Built-In Chains

| Chain Type                   | Use Case                                | Components                  | Key Benefit                        |
| ---------------------------- | --------------------------------------- | --------------------------- | ---------------------------------- |
| Create Stuff Documents Chain | Batch summarization, multi-doc analysis | LLM                         | Simple prompt stitching            |
| Create Retrieval Chain       | RAG over large datasets                 | Retriever + LLM + Doc Chain | Works beyond context window limits |

---

# 🚀 Next Steps

Now that you understand:

* Summarization Chains
* Retrieval Chains (RAG)

👉 Next, try implementing both with:

* PDFs
* Web pages
* Custom datasets

Experiment with:

* Different chunk sizes
* Embedding models
* LLM configurations

This is where real-world RAG systems start to take shape.


