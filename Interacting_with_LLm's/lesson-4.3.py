from typing import List
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

# 1. Model Definition
class Ticket(BaseModel):
    """Schema for a movie ticket reservation."""
    date:    str = Field(description="show date")
    time:    str = Field(description="show time")
    theater: str = Field(description="theater name")
    count:   int = Field(description="number of tickets")
    movie:   str = Field(description="preferred movie")

# 2. Initialize Parser & Model
model = ChatOllama(model="gemma4")  # or gemma4:e2b if low VRAM
parser = PydanticOutputParser(pydantic_object=Ticket)

# 3. Build the Prompt
ticket_template = """\
Book us a movie ticket for two this Friday at 6:00 PM.
Choose any theater. Send the confirmation by email.
Our preferred movie is: {query}

Format instructions:
{format_instructions}
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Respond ONLY with valid JSON. No explanation, no markdown, no extra text."),
    ("human", ticket_template),
])

# 4. Inspect the Full Prompt (optional)
formatted_prompt = prompt.format_messages(
    query="Interstellar",
    format_instructions=parser.get_format_instructions()
)
for msg in formatted_prompt:
    print(f"[{msg.type}]: {msg.content}\n")

# 5. Build Chain & Invoke
chain = prompt | model | parser

reservation = chain.invoke({
    "query": "Interstellar",
    "format_instructions": parser.get_format_instructions()
})

# 6. Output
print("Reservation object:", reservation)
print("Type:", type(reservation))
print("\nField values:")
print(f"  Movie:   {reservation.movie}")
print(f"  Date:    {reservation.date}")
print(f"  Time:    {reservation.time}")
print(f"  Theater: {reservation.theater}")
print(f"  Count:   {reservation.count}")