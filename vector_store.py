from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

def create_vector_store(text_chunks):
    vector_store = FAISS.from_texts(
        text_chunks, embedding=OpenAIEmbeddings()
    )
    return vector_store.as_retriever()
