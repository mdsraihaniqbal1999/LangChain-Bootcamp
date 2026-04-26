"""
Lesson 03: Few-Shot Prompt Templates
====================================

This lesson demonstrates how to implement few-shot prompting using LangChain
to teach language models specific patterns through examples.

Few-shot prompting teaches a language model a specific pattern by providing
a handful of examples at runtime, allowing the LLM to infer the desired
transformation from context without explicit instructions.
"""

# ============================================================================
# PREREQUISITES
# ============================================================================
#
# Install the required packages:
# pip install langchain langchain_ollama
#
# Set your environment variables if needed (for Ollama, ensure it's running).
# Never commit API keys to version control. Use environment variables.

from langchain_core.prompts import (
    ChatPromptTemplate,
    FewShotChatMessagePromptTemplate,
)
from langchain_ollama import OllamaLLM

# ============================================================================
# DEFINING EXAMPLES
# ============================================================================
#
# Prepare a small list of example pairs. Each item maps an input (word)
# to its output (uppercase version). These examples teach the pattern.
# Using a simpler task that models typically handle well.

examples = [
    {"input": "hello",   "output": "HELLO"},
    {"input": "world",   "output": "WORLD"},
    {"input": "python",  "output": "PYTHON"},
    {"input": "langchain", "output": "LANGCHAIN"},
    {"input": "machine", "output": "MACHINE"},
]

# ============================================================================
# CREATING AN EXAMPLE PROMPT
# ============================================================================
#
# Use ChatPromptTemplate to describe how each example should appear in the
# conversation. This defines the format for each example pair.

example_prompt = ChatPromptTemplate.from_messages(
    [
        ("human", "{input}"),
        ("ai",    "{output}"),
    ]
)

# ============================================================================
# BUILDING THE FEW-SHOT PROMPT
# ============================================================================
#
# Combine your individual example template with the list of examples via
# FewShotChatMessagePromptTemplate. This creates the few-shot learning component.

few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=examples,
)

# ============================================================================
# ASSEMBLING THE CHAT PROMPT TEMPLATE
# ============================================================================
#
# Wrap the system instruction, the few-shot examples, and the final human query
# into a single ChatPromptTemplate. This creates the complete prompt structure.
# Updated system prompt to be more specific about the task.

prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a text processing assistant. Your task is to convert any word given to you to uppercase letters. Simply provide the uppercase version."),
        few_shot_prompt,
        ("human",  "{input}"),
    ]
)

# ============================================================================
# INSPECTING THE PROMPT STRUCTURE
# ============================================================================
#
# Print out the internal representation to verify the sequence of messages.
# This helps ensure your prompt layout and placeholders are correct.

print("--- Prompt Template Structure ---")
print(prompt_template)

# ============================================================================
# FORMATTING FOR INVOCATION
# ============================================================================
#
# Populate the template with a new input—e.g., "brazil"—to generate the messages
# you'll send to the LLM. This creates the actual message list.

messages = prompt_template.format_messages(input="brazil")
print("\n--- Formatted Messages ---")
for message in messages:
    print(message)

# ============================================================================
# INVOKING THE MODEL
# ============================================================================
#
# Pass the formatted messages to the OllamaLLM model and print the response.
# The model should infer the uppercase pattern from the examples.

model = OllamaLLM(model="gemma3:1b")
response = model.invoke(messages)
print("\n--- Model Response ---")
print(response)

# Expected result: 'BRAZIL'
# Notice the model inferred the uppercase pattern purely from examples—
# no explicit instruction was needed.