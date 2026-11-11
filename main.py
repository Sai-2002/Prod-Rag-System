from rag import RAG

def main():

    rag = RAG()
    print("Starting to Ingest and create embeddings")
    rag.ingest()
    rag.create_embeddings()
    print(len(rag.data_embeddings))

    # print(len(rag.embed("Hello world")["embeddings"]))
    print(rag.retrieve("My name is"))

    while True:
        question = input("Ask : ")
        context = "\n\n".join(rag.retrieve(question=question))

        prompt = f""" Answer the question only using the context below. If the answer does not exist in the context, please say I dont know.

        Context : {context}

        Question : {question}
"""
        rag.generate(prompt=prompt)
    


if __name__ == "__main__":
    main()
