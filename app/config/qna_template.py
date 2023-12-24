"""
This template outlines three prompt types for document assistance: SystemMessagePromptTemplate for system information,
HumanMessagePromptTemplate for user queries, and ChatPromptTemplate for integrated system-user interactions.
"""

from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

TEMPLATE = (
    "You are an AI tool specialized in extracting answers from PDF documents."
    " Your purpose is to assist users in finding specific information or answers to their "
    "questions by analyzing the content of PDF files"
)
system_message_prompt = SystemMessagePromptTemplate.from_template(TEMPLATE)

CONTEXTUAL_ANSWER_EXTRACTOR = """

In this task, I review the provided context and question to offer an answer based solely on the context,
 citing it if the exact answer is present and indicating 'Information not found' when confidence is low.
 Do not guess the answer if it is not present.

Context:
{context}

Question:
{question}

Please provide the answer as plain text, focusing on precision and adherence to these instructions.
"""
contextual_message_prompt = HumanMessagePromptTemplate.from_template(
    CONTEXTUAL_ANSWER_EXTRACTOR
)

chat_prompt = ChatPromptTemplate.from_messages(
    [system_message_prompt, contextual_message_prompt]
)
