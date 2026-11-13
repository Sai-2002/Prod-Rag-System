from langchain_community.document_loaders import TextLoader
# from md_splitter import md_splitter
from .md_splitter import md_splitter

def doc_loader():
    loader = TextLoader("data/processed/Flowtron_Commercial_Pricing_Framework_Expanded.md")
    content = loader.load()
    return md_splitter(content=content[0].page_content)