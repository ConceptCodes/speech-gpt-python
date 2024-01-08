from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

def create_vector_store(text_chunks, api_key):
    """Creates a FAISS vector store from text chunks."""
    vector_store = FAISS.from_texts(
        text_chunks, embedding=OpenAIEmbeddings(openai_api_key=api_key)
    )
    return vector_store.as_retriever()
