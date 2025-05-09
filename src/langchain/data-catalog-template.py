import os
from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate

llm = OpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"))

tpl = """
You are an expert car dealer, you will identify if a customer question is valid or not.

Instructions:

Valid questions should only ask about either or both of the following:
1. Model's details such as name, make, model year, build, etc. However asking about the price of a model is not valid.
2. Car brands and manufacturers.

Response should be either "valid" or "invalid".

{question}
"""

template = PromptTemplate(template=tpl, input_variables=["question"])


response = llm.invoke(template.format(question="How much does a Tesla Model S cost?"))

print(response)