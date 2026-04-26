```markdown
# Adding Memory to LLM Apps

This article explains how to add memory capabilities to LLM applications for maintaining conversational continuity.

---

## Overview

In this lesson, we'll enhance your LLM-based application by introducing **memory capabilities** that maintain conversational continuity. A central component in any LLM architecture is **history**, which keeps track of past exchanges between the user and the model.

---

## Why History Matters

Enterprise-grade chatbots — like ChatGPT — depend on stored context to resume conversations seamlessly. Since **LLMs are inherently stateless**, each new query loses all prior session details unless you explicitly include them.

>  If you omit previous messages from the input, the model treats every prompt in isolation. This can lead to **incoherent replies or hallucinations**.

To preserve context, every request must bundle in the earlier conversation thread. By doing so, the LLM can reference past discussions and produce coherent, context-aware responses.

---

## Memory Types

LLM apps generally leverage two forms of memory:

| Memory Type | Scope | Persistence |
|-------------|-------|-------------|
| **Short-term** | Single session | Volatile (stored in RAM) |
| **Long-term** | Cross sessions | Durable (persisted in external store) |

###  Short-Term Memory

- Lives only in **RAM** during a session
- Ideal for quick back-and-forth exchanges
- Data retention ends when the session closes

### Long-Term Memory

- Persisted in an **external database**
- Ensures conversations can resume even after days or weeks
- Backed by stores like **Redis**, **SQLite**, **PostgreSQL**, or **vector databases**

---

## Database Options

| Store | Speed | Persistence | Best For |
|-------|-------|-------------|----------|
| RAM | ⚡ Fastest |  Volatile | Single-session bots |
| Redis | ⚡ Very fast |  Optional TTL | High-traffic, short-lived sessions |
| SQLite |  Moderate |  Durable | Local dev / low-volume apps |
| PostgreSQL |  Moderate |  Durable | Production multi-user apps |
| Vector DB | Moderate | Durable | Semantic/contextual retrieval |

>  **Tip:** While Redis is a popular choice for its speed and simplicity, the same patterns apply to any persistent store.

---

## Next Steps

In the upcoming lesson, we'll walk through:

- Integration code for a Redis-backed memory store
- How to serialize conversation history
- Best practices for managing **token usage**
```