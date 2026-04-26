# LangChain Bootcamp - Tips, Tricks & Resources

This directory contains helpful tips, tricks, and resources for working with LangChain and LLM applications.

## Key Libraries

### Core LangChain Libraries

#### **langchain-core** (v1.3.0+)
The foundational library containing core abstractions for building LLM applications.

**Key Components:**
- `PromptTemplate` - Define and manage prompt templates with variable substitution
- `PydanticOutputParser` - Parse LLM output into structured Pydantic models
- `BaseOutputParser` - Base class for custom output parsers
- LLM base interfaces and runnables

```python
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

class ArticleSummary(BaseModel):
    title: str = Field(description="Title of the article")
    tags: list[str] = Field(description="List of relevant tags")

parser = PydanticOutputParser(pydantic_object=ArticleSummary)
```

#### **langchain** (v1.2.15+)
High-level abstractions and integrations for LLM applications.

**Key Components:**
- Chains and agents
- Memory systems
- Callbacks and logging
- Document loaders

#### **langchain-ollama** (v1.1.0+)
Integration for using Ollama models (local LLMs) with LangChain.

**Setup:**
```bash
# Install Ollama models
ollama pull gemma3:1b
ollama pull mistral
ollama pull llama2

# Start Ollama server
ollama serve
```

**Usage:**
```python
from langchain_ollama import OllamaLLM

llm = OllamaLLM(model="gemma3:1b")
```

### Supporting Libraries

#### **pydantic** (v2.13.3+)
Data validation and serialization using Python type hints.

**Key Features:**
- Model validation
- JSON schema generation
- Field descriptions and defaults

```python
from pydantic import BaseModel, Field

class Model(BaseModel):
    field_name: str = Field(description="Field description")
```

#### **requests** (v2.33.1+)
HTTP library for making API calls.

#### **pyyaml** (v6.0.3+)
YAML parsing and generation for configuration files.

---

## Best Practices

### 1. **Prompt Engineering**
- Always include format instructions in prompts
- Be explicit about expected output format
- Provide examples when possible
- Use `partial_variables` for reusable instructions

### 2. **Structured Output**
- Define schemas using Pydantic models with Field descriptions
- Use `PydanticOutputParser` for reliable parsing
- Always include format instructions in prompts
- Add error handling for parsing failures

```python
template = """{format_instructions}

Article: {article_text}

ONLY return valid JSON. No explanation."""

prompt = PromptTemplate(
    template=template,
    input_variables=["article_text"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)
```

### 3. **Chain Building**
- Use the pipe operator `|` to compose chains (modern approach)
- Call chains with `.invoke()` method (not deprecated `.run()`)
- Always wrap execution in try/except for error handling

```python
chain = prompt | llm | parser
result = chain.invoke({"article_text": "..."})
```

### 4. **Local Model Development**
- Use Ollama for local development and testing
- Start with smaller models (gemma3:1b) for faster iteration
- Test structured output support before deployment

---

## Common Issues & Solutions

### ModuleNotFoundError: No module named 'langchain.core'
**Solution:** Use `langchain_core` instead of `langchain.core`
```python
# ❌ Wrong
from langchain.core import PromptTemplate

# ✅ Correct
from langchain_core.prompts import PromptTemplate
```

### OutputParserException: Failed to parse output
**Causes:**
- Missing or incorrect format instructions in prompt
- LLM not outputting valid JSON
- Schema mismatch between expected and actual output

**Solutions:**
- Ensure `{format_instructions}` is in the template
- Use Pydantic models with Field descriptions
- Add explicit instruction: "ONLY return valid JSON"

### LLMChain not found
**Solution:** Use the modern runnable API with pipe operator
```python
# ❌ Old (deprecated)
chain = LLMChain(llm=llm, prompt=prompt)
output = chain.run(...)

# ✅ New (modern)
chain = prompt | llm | parser
result = chain.invoke({...})
```

---

## Environment Setup

### Virtual Environment
```bash
# Create
python3 -m venv langchain-env

# Activate
source langchain-env/bin/activate

# Install dependencies
pip install langchain langchain-core langchain-ollama pydantic
```

### Required Services
- **Ollama** - Download from [ollama.ai](https://ollama.ai) and run `ollama serve`

---

## Resources

- [LangChain Documentation](https://python.langchain.com)
- [Pydantic Documentation](https://docs.pydantic.dev)
- [Ollama GitHub](https://github.com/ollama/ollama)
- [LangChain Output Parsing](https://python.langchain.com/docs/modules/model_io/output_parsers)

---

## Quick Reference

| Task | Command/Code |
|------|--------------|
| Create Pydantic model | `class Model(BaseModel): field: type` |
| Create parser | `PydanticOutputParser(pydantic_object=Model)` |
| Build prompt | `PromptTemplate(template=..., input_variables=[...])` |
| Create chain | `prompt \| llm \| parser` |
| Run chain | `chain.invoke({...})` |
| Start Ollama | `ollama serve` |
| Pull model | `ollama pull model-name` |

---

Last Updated: April 24, 2026
