import whisper
import os

from halo import Halo
from prompts import DEFAULT_DOCUMENT_PROMPT
from langchain.schema import format_document


def load_whisper_model() -> whisper.Whisper:
    model = whisper.load_model("tiny")
    print("Model loaded!")
    return model


def transcribe_audio(model, filepath) -> str:
    spinner = Halo(text='Thinking...', spinner='dots')
    spinner.start()
    result = model.transcribe(filepath)
    spinner.stop()
    print("Transcription done!", end="\n\n")
    return result['text']


def chunk_text(text, chunk_size=500) -> list[str]:
    snippets = []
    start = 0
    end = chunk_size
    while start < len(text):
        snippet = text[start:end]
        snippets.append(snippet)
        start = end
        end += chunk_size
    return snippets


def combine_documents(docs, document_prompt=DEFAULT_DOCUMENT_PROMPT, document_separator="\n\n") -> str:
    doc_strings = [format_document(doc, document_prompt) for doc in docs]
    return document_separator.join(doc_strings)

def create_asset_dir() -> None:
    if not os.path.exists("assets"):
        os.makedirs("assets")
        os.makedirs("assets/vector_store")
