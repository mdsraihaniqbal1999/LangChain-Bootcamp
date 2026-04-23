"""
Lesson 02: Prompt Templates in LLM Applications
===============================================

This lesson explains why prompt templates are useful, what problems they
solve, and how to use different template types in LangChain.
"""

# ============================================================================
# WHAT ARE PROMPT TEMPLATES?
# ============================================================================
#
# Prompt templates are reusable text patterns that let you generate prompts
# dynamically by filling in variable values.
#
# They solve key problems such as:
#   - Repetition: Avoid rewriting the same prompt structure for every query
#   - Consistency: Keep system instructions and formatting uniform
#   - Maintainability: Change the prompt layout in one place
#   - Safety: Separate static instructions from dynamic user data
#
# In LangChain, prompt templates can be used for simple text prompts as well
# as structured multi-message chat prompts.

# ============================================================================
# PROMPT TEMPLATE TYPES
# ============================================================================
#
# The main prompt template types in LangChain are:
#
# 1. PromptTemplate
#    - Base template for plain text prompts
#    - Use when you only need a single text prompt with variables
#    - Example: "Summarize the following text: {text}"
#
# 2. ChatPromptTemplate
#    - High-level template for chat-style interactions
#    - Composes multiple message templates into a full conversation prompt
#    - Ideal for LLMs that expect role-based chat history
#
#    Chat message types inside ChatPromptTemplate:
#    -----------------------------------------------
#    a. SystemMessagePromptTemplate
#       - Defines the assistant role, rules, and behavior
#       - Example: "You are a helpful assistant."
#
#    b. HumanMessagePromptTemplate
#       - Defines the user query or request
#       - Example: "Explain {topic} in simple terms."
#
#    c. AIMessagePromptTemplate
#       - Defines example AI responses or expected output format
#       - Useful for few-shot examples and prompt scaffolding
#
# 3. Specialized prompt template patterns
#    - Few-shot templates: use examples to teach the model
#    - Conditional templates: change prompt text based on input
#    - Reusable templates: store templates for repeated use
#
# This typed hierarchy clarifies that:
#   - PromptTemplate is the basic text template type,
#   - ChatPromptTemplate is the chat-specific container,
#   - message prompt templates are the role-based building blocks.
#
# ============================================================================
# WHY USE PROMPT TEMPLATES?
# ============================================================================
#
# Benefits:
#   - Reduce copy/paste errors
#   - Enable dynamic prompt generation with variables
#   - Keep instructions separate from user content
#   - Support better prompt engineering and testing

# ============================================================================
# CODE EXAMPLES
# ============================================================================

from langchain_ollama import OllamaLLM
from langchain_core.prompts import (
    PromptTemplate,
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    AIMessagePromptTemplate,
)

# 1. Simple text prompt using PromptTemplate
simple_template = PromptTemplate(
    template="Write a short, friendly summary of the following topic: {topic}",
    input_variables=["topic"],
)

prompt_text = simple_template.format(topic="how solar panels work")
print("--- Simple PromptTemplate ---")
print(prompt_text)

# Initialize the Ollama model once for the next examples
llm = OllamaLLM(model="gemma3:1b")

# 2. Chat prompt with System and Human message templates
system_template = SystemMessagePromptTemplate.from_template(
    "You are a helpful assistant who explains concepts simply and clearly."
)

human_template = HumanMessagePromptTemplate.from_template(
    "Explain {topic} in simple terms for a beginner."
)

chat_prompt = ChatPromptTemplate.from_messages([system_template, human_template])
chat_messages = chat_prompt.format_messages(topic="reinforcement learning")

print("\n--- Chat PromptTemplate Messages ---")
for message in chat_messages:
    print(message)

response = llm.invoke(chat_messages)
print("\n--- Ollama Response ---")
print(response)

# 3. Using AIMessagePromptTemplate for examples or expected outputs
ai_example = AIMessagePromptTemplate.from_template(
    "Reinforcement learning is a type of machine learning where agents learn"
    " by receiving rewards for good actions."
)

example_prompt = ChatPromptTemplate.from_messages(
    [system_template, human_template, ai_example]
)
example_messages = example_prompt.format_messages(topic="reinforcement learning")

print("\n--- Chat Prompt with AI Example ---")
for message in example_messages:
    print(message)

# 4. Why structured templates matter
#    - System template sets the role and tone
#    - Human template provides the user query
#    - AI template can provide a sample response or expected format
#
# This structure makes it easier to build reliable prompts and maintain
# consistent behavior across requests.
