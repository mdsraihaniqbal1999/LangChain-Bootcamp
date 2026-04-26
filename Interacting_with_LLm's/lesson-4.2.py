"""
Lesson: JSON Output Parsing with LangChain + Ollama
===================================================
"""

# ============================================================================
# PREREQUISITES
# ============================================================================
#
# pip install langchain langchain_ollama
# ollama pull gemma3:1b

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_ollama import OllamaLLM
import json

# Initialize Ollama with Gemma4
llm = OllamaLLM(model="gemma4")

# ============================================================================
# 1. BASIC PROMPT WITHOUT PARSING
# ============================================================================
#
# First, create a simple prompt template to list three toy companies

prompt = PromptTemplate(
    template="List 3 good toy companies for kids in {country} and why they are good",
    input_variables=["country"]
)

raw_output = llm.invoke(input=prompt.format(country="USA"))
print("\n--- Raw Output (No Parser) ---")
print(raw_output)

# Example response:
# 1. LEGO - develops creativity
# 2. Mattel - trusted brands
# 3. Hasbro - fun learning
#
# While human-readable, this plain-text format is hard to consume programmatically.

# ============================================================================
# 2. DEFINING THE JSON OUTPUT PARSER
# ============================================================================
#
# Instantiate LangChain's JSON parser and get format instructions

output_parser = JsonOutputParser()
format_instructions = output_parser.get_format_instructions()
print("\n--- Format Instructions ---")
print(format_instructions)

# This prints instructions telling the LLM to return valid JSON

# ============================================================================
# 3. PROMPT TEMPLATE WITH FORMAT INSTRUCTIONS
# ============================================================================
#
# Embed the format instructions into your template to enforce JSON output

prompt = PromptTemplate(
    template=(
        "List 3 good toy companies for kids in {country}\n"
        "For each company, provide the name and reason\n"
        "{format_instructions}"
    ),
    input_variables=["country"],
    partial_variables={"format_instructions": format_instructions}
)

print("\n--- Formatted Prompt ---")
print(prompt.format(country="USA"))

# ============================================================================
# 4. INVOKING THE MODEL AND PARSING
# ============================================================================
#
# Invoke the LLM with the enhanced prompt and parse the JSON response

response = llm.invoke(input=prompt.format(country="USA"))
print("\n--- LLM Response (JSON) ---")
print(response)

# Parse the JSON response into a Python dictionary
countries = output_parser.parse(response)
print("\n--- Parsed Output ---")
print(f"Type: {type(countries)}")  # <class 'dict'>
print(f"Data: {json.dumps(countries, indent=2)}")

# ============================================================================
# 5. WORKING WITH THE PARSED OUTPUT
# ============================================================================
#
# Now you can work with the structured data programmatically

print("\n--- Working with Structured Data ---")
for company, reason in countries.items():
    print(f"Company: {company}")
    print(f"Reason: {reason}")
    print("---")

# ============================================================================
# 6. COMPLETE CHAIN EXAMPLE (MODERN SYNTAX)
# ============================================================================

print("\n--- Complete Chain Example ---")

chain = prompt | llm | output_parser
result = chain.invoke({"country": "Japan"})
print(json.dumps(result, indent=2))

# ============================================================================
# 7. ERROR HANDLING
# ============================================================================

print("\n--- Error Handling Example ---")

def safe_json_parse(response_text):
    """Safely parse JSON response with error handling"""
    try:
        return output_parser.parse(response_text)
    except Exception as e:
        print(f"JSON parsing failed: {e}")
        return {"error": "Failed to parse", "raw_response": response_text}

test_response = '{"LEGO": "Creativity", "Mattel": "Trusted"}'  # Valid JSON
safe_result = safe_json_parse(test_response)
print(f"Safe parse result: {safe_result}")

# ============================================================================
# COMPARISON OF LANgCHAIN OUTPUT PARSERS
# ============================================================================
#
# Parser Type                | Description                    | Example Usage
# ---------------------------|--------------------------------|------------------
# JsonOutputParser           | Ensures valid JSON output      | JsonOutputParser()
# CommaSeparatedListOutputParser | Parses CSV lists           | CommaSeparatedListOutputParser()
# PydanticOutputParser       | Validates against Pydantic model | PydanticOutputParser(pydantic_object=MyModel)
#
# For toy companies, JsonOutputParser is ideal because it gives you
# structured key-value pairs you can use immediately in code.