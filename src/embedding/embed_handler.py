from langchain_ollama import OllamaEmbeddings
from langchain_postgres import PGVector
import os
from dotenv import load_dotenv

load_dotenv()

uname=os.getenv("POSTGRES_UNAME")
password=os.getenv("POSTGRES_PASS")

embeddings = OllamaEmbeddings(model="nomic-embed-text:latest", base_url="http://localhost:11434")

connection = f"postgresql://{uname}:{password}@localhost:5432/rag_system_db"
collection_name = "document_embeddings"

vector_store = PGVector(
    embeddings=embeddings,
    collection_name=collection_name,
    connection=connection,
    use_jsonb=True
)

def embed_chunk(encriched_text, documents):
    vectors = embeddings.embed_documents(encriched_text)

    vector_store.add_embeddings(
        texts=[doc.page_content for doc in documents],
        embeddings=vectors,
        metadatas=[doc.metadata for doc in documents],
        ids=[doc.id for doc in documents]
    )

def embed_query(user_query):
    return embeddings.embed_query(f"search_query: {user_query}")

def search_similar_chunks(user_query, k=4, metadata_filter=None):
    query_vector = embed_query(user_query=user_query)

    results = vector_store.similarity_search_with_score_by_vector(embedding=query_vector, k=k, filter=metadata_filter)

    return results