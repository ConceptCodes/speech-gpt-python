import os
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

def create_vector_store(text_chunks):
    print("Creating vector store...")
    vector_store = FAISS.from_texts(
        text_chunks, embedding=OpenAIEmbeddings()
    )
    return vector_store

def does_vector_store_exist(name: str) -> bool:
    return os.path.exists(f"assets/vector_store/{name}")

def load_vector_store(name: str):
    return FAISS.load_local(f"assets/vector_store/{name}", embeddings=OpenAIEmbeddings())

def save_vector_store(vector_store, name: str):
    print("Saving vector store...")
    vector_store.save_local(f"assets/vector_store/{name}")