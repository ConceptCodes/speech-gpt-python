import argparse
import sys
import os

from halo import Halo
from operator import itemgetter
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import get_buffer_string
from langchain_openai import ChatOpenAI
from colorama import init
from termcolor import cprint
from pyfiglet import figlet_format

from utils import combine_documents, load_whisper_model, transcribe_audio, chunk_text
from vector_store import create_vector_store
from memory import create_memory
from prompts import CONDENSE_QUESTION_PROMPT, ANSWER_PROMPT

init(strip=not sys.stdout.isatty())

parser = argparse.ArgumentParser(
    prog='Speech GPT',
    description='A GPT powered CLI chatbot to talk with your speeches',
    epilog='Enjoy the program! :)'
)

parser.add_argument('-f', '--filepath', type=str,
                    help='Path to audio file', required=True)

args = parser.parse_args()

API_KEY = "YOUR_API_KEY"


def chat_with_speech(filepath):
    cprint(figlet_format('Speech GPT', font='starwars'), attrs=['bold'])
    """Transcribes audio, prepares documents, and interacts with the chatbot."""

    if os.path.isfile(filepath) == False:
        print("File does not exist!")
        return

    model = load_whisper_model()
    transcribed_text = transcribe_audio(model, filepath)
    text_chunks = chunk_text(transcribed_text)
    vector_store = create_vector_store(text_chunks, API_KEY)
    memory = create_memory()

    loaded_memory = RunnablePassthrough.assign(
        chat_history=RunnableLambda(
            memory.load_memory_variables) | itemgetter("history")
    )

    standalone_question = {
        "standalone_question": {
            "question": lambda x: x["question"],
            "chat_history": lambda x: get_buffer_string(x["chat_history"]),
        }
        | CONDENSE_QUESTION_PROMPT
        | ChatOpenAI(temperature=0, api_key=API_KEY)
        | StrOutputParser(),
    }

    retrieved_documents = {
        "docs": itemgetter("standalone_question") | vector_store,
        "question": lambda x: x["standalone_question"],
    }

    final_inputs = {
        "context": lambda x: combine_documents(x["docs"]),
        "question": itemgetter("question"),
    }

    answer = {
        "answer": final_inputs | ANSWER_PROMPT | ChatOpenAI(api_key=API_KEY),
        "docs": itemgetter("docs"),
    }

    final_chain = loaded_memory | standalone_question | retrieved_documents | answer

    while True:
        print("\n>>> ", end="")
        question = input()
        if question == "exit":
            break
        spinner = Halo(text='\nThinking...', spinner='dots')
        spinner.start()
        result = final_chain.invoke({"question": question})
        spinner.stop()
        print("\nAnswer: {}".format(result["answer"]))
        memory.save_context({"question": question}, {
                            "answer": result["answer"].content})


chat_with_speech(args.filepath)
