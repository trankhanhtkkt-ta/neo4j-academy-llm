import os
from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate

llm = OpenAI(
    openai_api_key=os.getenv("OPENAI_API_KEY")
    )

template = PromptTemplate.from_template("""
You are a cockney fruit and vegetable seller.
Your role is to assist your customer with their fruit and vegetable needs.
Respond using cockney rhyming slang.

Tell me about the following fruit: {fruit}
""")

llm_chain = template | llm

response = llm_chain.invoke({"fruit": "apple"})

print(response)