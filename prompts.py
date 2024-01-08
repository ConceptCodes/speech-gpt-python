from langchain.prompts import PromptTemplate, ChatPromptTemplate

CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template("""
Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:
""")

ANSWER_PROMPT = ChatPromptTemplate.from_template("""
Answer the question based only on the following context:
{context}

Question: {question}
""")

DEFAULT_DOCUMENT_PROMPT = PromptTemplate.from_template(template="{page_content}")
