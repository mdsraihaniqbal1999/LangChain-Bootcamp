# LCEL = LangChain Expression Language
# -----------------------------------
# LCEL is a domain-specific language (DSL) in LangChain used to compose,
# visualize, and maintain pipelines using a simple pipe operator (|).
#
# It allows chaining of components such as prompts, LLMs, parsers,
# retrievers, and custom logic in a clean and intuitive way.

# Core Concept
# Each LangChain component represents a stage in a pipeline:
#
#     Input -> Prompt -> LLM -> Parser -> Output
#
# The output of one stage becomes the input of the next,
# enabling clean, modular, and easy-to-debug workflows.

# Internal Working
# Every component follows a standard contract:
#     Input -> Processing -> Output
#
# When components are connected using the pipe operator (|),
# LCEL builds an executable graph under the hood.

# Benefits
# - Modularity:
#     Components can be swapped, reordered, or reused easily
#
# - Readability:
#     Provides a clear left-to-right flow similar to shell pipelines
#
# - Extensibility:
#     Supports custom components, retrievers, and parsers

# Runnable: Core Abstraction
# LCEL is built on the Runnable abstraction.
#
# By subclassing Runnable[InputType, OutputType], a component defines:
# - How input is received
# - How processing is performed
# - What output is returned
#
# This ensures compatibility with LCEL pipelines.

#from langchain_core.runnables import Runnable

#class MyCustomProcessor(Runnable[str, str]):
#    def invoke(self, input_text: str) -> str:
#        return input_text.upper()

# Production Usage
# LCEL is production-ready and supports large pipelines with multiple
# chained components running reliably at scale.

# Unix Pipeline Analogy
# LCEL follows a model similar to Unix pipes:
#
#     cat file.txt | grep "error" | wc -l
#
# Breakdown:
# - cat file.txt streams file content
# - grep "error" filters matching lines
# - wc -l counts the results
#
# Equivalent LCEL flow:
#     prompt | llm | parser

# Meta-Chains
# LCEL supports chaining of entire pipelines.
#
# The output of one chain can be passed as input to another,
# enabling multi-step reasoning and complex workflows.
#
# Example:
#     chain1 -> chain2 -> chain3

from langchain_core.prompts  import PromptTemplate
from langchain_ollama import OllamaLLM
from langchain_core.output_parsers import StrOutputParser

prompt = PromptTemplate(
    template="Q: {question}\nA:",
    input_variables=["question"]
)
llm = OllamaLLM(model="gemma4")
parser = StrOutputParser()

chain = prompt | llm | parser
result = chain.invoke({"question": "Tell me about the Godfather movie"})
print(result)
