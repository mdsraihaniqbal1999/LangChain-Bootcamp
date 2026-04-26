# LCEL Demo: Exploring Configurable Parameters
# --------------------------------------------
# This example demonstrates how to use configurable fields in LangChain
# to dynamically change runtime parameters (like model selection).
#
# Instead of hardcoding the model, we make it configurable so we can:
# - Switch models at runtime
# - Optimize cost vs performance
# - Build flexible pipelines


# Step 1: Imports
# ----------------
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import ConfigurableField
from langchain_ollama import OllamaLLM


# Step 2: Define Configurable Model
# ---------------------------------
# We expose "model" as a configurable parameter

model = OllamaLLM(model="gemma4").configurable_fields(
    model=ConfigurableField(
        id="model",
        name="Model Name",
        description="Select the LLM model to use at runtime",
    )
)

# Default model = gemma4


# Step 3: Build the Chain
# -----------------------
prompt = PromptTemplate.from_template(
    "Write a short poem about {subject}"
)

chain = prompt | model


# Step 4: Invoke with Default Model
# ---------------------------------
response_default = chain.invoke({
    "subject": "cat"
})

print(response_default)

# Uses default model: gemma4


# Step 5: Override Model at Runtime
# ---------------------------------
# Dynamically switch to another model (if available in Ollama)

response_override = chain.with_config(
    configurable={"model": "gemma3:1b"}
).invoke({
    "subject": "cat"
})

print(response_override)

# Now uses: gemma3:1b instead of gemma4


# Step 6: Multiple Runtime Configurations
# ---------------------------------------
# You can dynamically control behavior per request

subjects = ["AI", "space", "ocean"]

for s in subjects:
    result = chain.with_config(
        configurable={"model": "gemma4"}
    ).invoke({"subject": s})
    
    print(f"\nTopic: {s}")
    print(result)


# Notes:
# ------
# - configurable_fields() exposes parameters for runtime override
# - with_config() lets you change them dynamically
# - No need to rebuild the chain for different models
#
# - Useful for:
#   * Switching between cheap vs powerful models
#   * A/B testing models
#   * User-controlled settings


# Example Use Cases:
# ------------------
# - Use fast model for simple queries
# - Use powerful model for complex reasoning
#
# Example:
# if complex_query:
#     chain.with_config(configurable={"model": "gemma3:1b"})
# else:
#     chain.with_config(configurable={"model": "gemma4"})


# Summary:
# --------
# - ConfigurableField makes parameters dynamic
# - Default values are used unless overridden
# - with_config() allows runtime flexibility
# - Helps optimize cost, performance, and control