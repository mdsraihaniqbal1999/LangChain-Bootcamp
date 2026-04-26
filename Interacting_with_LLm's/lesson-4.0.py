from langchain_ollama import OllamaLLM
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from typing import List

# 1. Define your schema with Pydantic
class ArticleSummary(BaseModel):
    title: str = Field(description="Title of the article")
    tags: List[str] = Field(description="List of relevant tags")

# 2. Create the parser
parser = PydanticOutputParser(pydantic_object=ArticleSummary)

# 3. Build the prompt
template = """
Generate an article summary.

{format_instructions}

Article:
\"\"\"
{article_text}
\"\"\"
"""

prompt = PromptTemplate(
    template=template,
    input_variables=["article_text"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

# 4. Use LCEL (pipe syntax) instead of deprecated LLMChain
model = OllamaLLM(model="gemma4")
chain = prompt | model | parser

# 5. Invoke the chain
result = chain.invoke({"article_text": "Fruits"})

print(result.title)  # Access as object attributes
print(result.tags)