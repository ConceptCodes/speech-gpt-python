from langchain.memory import ConversationBufferMemory

def create_memory():
    return ConversationBufferMemory(
        return_messages=True, output_key="answer", input_key="question"
    )
