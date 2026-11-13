from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from uuid import uuid4

from ..embedding.embed_handler import embed_chunk

def md_splitter(content):
    headers_list = [
        ("#", "Header 1"),
        ("##", "Header 2")
    ]

    content_split = MarkdownHeaderTextSplitter(headers_to_split_on=headers_list)

    document = content_split.split_text(content)

    for doc in document:
        doc.id = str(uuid4())

    enriched_text = [enrich_document(doc) for doc in document]

    embed_chunk(encriched_text=enriched_text, documents=document)

    return {"message":"Successfully Created Embeddings", "status":"200"}


def enrich_document(doc):

    header = " > ".join(
        str(value) for key, value in doc.metadata.items() if key.startswith("Header")
    )

    enriched_text = f"""
    Section: {header}

    Content: {doc.page_content}
"""
    return enriched_text