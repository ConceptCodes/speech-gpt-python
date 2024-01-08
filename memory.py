from langchain.memory import ConversationBufferMemory

def create_memory():
    """Creates a ConversationBufferMemory for dialogue history."""
    return ConversationBufferMemory(
        return_messages=True, output_key="answer", input_key="question"
    )
