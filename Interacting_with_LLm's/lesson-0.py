from langchain_ollama import OllamaLLM
import os

# Initialize the Ollama model
llm = OllamaLLM(
    model="gemma3:1b",  # or "gemma3:1b" depending on your exact tag
)

text = "What would be a good company that makes toys for kids?"
print(llm.invoke(text))

