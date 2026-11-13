from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

llm = ChatOllama(model="qwen3:4b-instruct")

prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer the user's question based ONLY from the following context. If the answer is not in the context, reply I DONT KNOW. context:\n\n{context}"),
    ("human","{question}"),
])

chain = prompt | llm

def generate_response(context, question):

    repsonse = chain.invoke({
        "context": context,
        "question": question
    })

    return repsonse