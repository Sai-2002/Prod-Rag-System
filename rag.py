import ollama
import numpy as np

class RAG:
    def __init__(self):
        self.content_data = []
        self.data_embeddings = []
        self.index_embeddings = 0
            
    def ingest(self):
        with open("example_doc.md","r", encoding="utf-8") as file:
            content = file.read()

            content_data = content.split("\n\n")
        self.content_data=content_data

    def create_embeddings(self):
        # data_embeddings
        for content in self.content_data:
            embedded_content = self.embed(content)
            self.data_embeddings.append(embedded_content)

        self.index_embeddings=len(self.data_embeddings)

    @staticmethod
    def embed(content):
        embeded_chunks = ollama.embed(
            model='nomic-embed-text:latest',
            input=content
        )
        return embeded_chunks["embeddings"][0]

    def retrieve(self,question, k=3):
        q = self.embed(question)
        q = np.array(q)
        vectors = np.array(self.data_embeddings)
        similarity = vectors @ q / (np.linalg.norm(self.data_embeddings, axis=1)*np.linalg.norm(q))

        return [self.content_data[i] for i in np.argsort(similarity)[::-1][:k]]
        
    @staticmethod
    def generate(prompt):
        response = ollama.chat(
            model="qwen3:4b-instruct",
            messages=[{'role':'user','content':prompt}]
        )

        print(response['message']['content'])