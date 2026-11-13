from ..embedding.embed_handler import search_similar_chunks

def get_similar_chunks(user_query):
    return search_similar_chunks(user_query=user_query)