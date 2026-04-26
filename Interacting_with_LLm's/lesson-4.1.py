# Minimal working example - type this yourself!
from langchain_ollama import OllamaLLM
from langchain_core.output_parsers import CommaSeparatedListOutputParser
from langchain_core.prompts import PromptTemplate

llm = OllamaLLM(model="gemma4")
parser = CommaSeparatedListOutputParser()

prompt = PromptTemplate(
    template="List 3 {things}.\n{format_instructions}",
    input_variables=["things"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

chain = prompt | llm | parser
result = chain.invoke({"things": "countries that play football in the World Cup"})
print(result)  