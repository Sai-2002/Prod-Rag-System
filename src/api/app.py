from fastapi import FastAPI
from ..ingestion.loader import doc_loader
from ..retrieval.retrieval import get_similar_chunks
from pydantic import BaseModel
from ..generation.generation import generate_response

app = FastAPI()

version_url = "/api/v1"

class UserQuery(BaseModel):
    query: str

@app.get("/")
def home():
    return {"message": "Live"}


@app.get(f"{version_url}/load")
def load_document():
    content = doc_loader()

    return {"status":"200", "content":content}

@app.post(f"{version_url}/user_query")
def post_user_query(userQuery: UserQuery):

    results = get_similar_chunks(user_query=userQuery.query)

    # print(results);
    page_contents = [document.page_content for document,_ in results]

    context = "\n\n".join(page_contents)

    response = generate_response(context, userQuery.query)

    return {"response" : response.content}
