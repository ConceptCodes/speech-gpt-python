import whisper
import os
import subprocess

from halo import Halo
from prompts import DEFAULT_DOCUMENT_PROMPT
from langchain.schema import format_document


def load_whisper_model() -> whisper.Whisper:
    model = whisper.load_model("tiny")
    print("Model loaded!")
    return model


def transcribe_audio(model, file_path) -> str:
    spinner = Halo(text='Thinking...', spinner='dots')
    spinner.start()
    result = model.transcribe(file_path)
    spinner.stop()
    print("Transcription done!", end="\n\n")
    return result['text']

def convert_audio(file_path: str) -> None:
    file_name = os.path.basename(file_path).split('.')[0]
    subprocess.run([
        'ffmpeg',
        '-i',
        file_path,
        '-ar',
        '16000',
        '-ac',
        '1',
        '-c:a',
        'pcm_s16le',
        f'./assets/{file_name}.wav'
    ])

def M1_transcribe_audio(file_path) -> str:
    subprocess.run(['./whisper.cpp/main', '--model', 'models/ggml-base.en.bin', '--output-txt', '--file', file_path])

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
        os.makedirs("assets/output")
