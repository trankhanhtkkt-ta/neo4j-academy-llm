import os
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import StrOutputParser

from langchain_neo4j import Neo4jGraph # graph
from langchain_neo4j import Neo4jChatMessageHistory # memory

from langchain_openai import ChatOpenAI # llm

from langchain_community.chat_message_histories import ChatMessageHistory

from langchain_core.runnables.history import RunnableWithMessageHistory

from uuid import uuid4


SESSION_ID = str(uuid4())
print(f"Session ID: {SESSION_ID}")

graph = Neo4jGraph(
    url=os.getenv("NEO4J_URI"),
    username=os.getenv("NEO4J_USERNAME"),
    password=os.getenv("NEO4J_PASSWORD")
)

chat_llm = ChatOpenAI(
    openai_api_key=os.getenv("OPENAI_API_KEY")
)


def get_memory(session_id):
    return Neo4jChatMessageHistory(session_id=session_id, graph=graph)


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a surfer dude, having a conversation about the surf conditions on the beach. Respond using surfer slang.",
        ),
        ("system", "{context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ]
)


chat_chain = prompt | chat_llm | StrOutputParser()

chat_with_message_history = RunnableWithMessageHistory(
    chat_chain,
    get_memory,
    input_messages_key="question",
    history_messages_key="chat_history",
)


current_weather = """
    {
        "surf": [
            {"beach": "Fistral", "conditions": "6ft waves and offshore winds"},
            {"beach": "Polzeath", "conditions": "Flat and calm"},
            {"beach": "Watergate Bay", "conditions": "3ft waves and onshore winds"}
        ]
    }"""

while (question := input("> ")) != "exit":

    response = chat_with_message_history.invoke(
        {
            "context": current_weather,
            "question": question,

        },
        config={
            "configurable": {"session_id": SESSION_ID}
        }
    )

    print(response)