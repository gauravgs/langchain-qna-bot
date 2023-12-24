"""
Module Description: 
Text and Query Handling: LangChain + OpenAI

Simplify PDF text extraction and Q&A with LangChain and OpenAI models.

Functions:
- extract_pdf_data(filepath): Open and divide PDF, return sections.
- generate_embeddings(docs): Convert text to embeddings using HuggingFace.
- process_and_embed(filepath): Process PDF and return embeddings.
- generate_answer(filepath, question): Use OpenAI models to answer questions.

Dependencies:
- LangChain (text and embeddings)
- HuggingFace's Transformers
- OpenAI's GPT models
- LangChain Core (basic operations)

Usage:
For systems extracting and analyzing PDFs with AI answers. Import and use functions.

Example:
Answer a PDF question:

sections = extract_pdf_data("example.pdf")
embeddings = generate_embeddings(sections)
answer = generate_answer("example.pdf", "Key subject?")
"""

from typing import List
import tiktoken
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain.chat_models import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

from app.config.constants import REQUESTS, TIME_PERIOD
from app.config.qna_template import chat_prompt
from ratelimit import limits, sleep_and_retry


def extract_pdf_data(file_path: str):
    """
    Load and split the data from a PDF file.
    :param file_path: Path to the PDF file.
    :return: List of document chunks.
    """
    pdf_loader = UnstructuredPDFLoader(file_path)
    document = pdf_loader.load()

    text_divider = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=10,
        separators=["\n\n", "\n", " ", ""],
        length_function=calculate_length,
    )
    document_chunks = text_divider.split_documents(document)
    return document_chunks


def generate_embeddings(doc_chunks: List[str]):
    """
    Generate embeddings for document chunks.
    :param doc_chunks: List of document chunks.
    :return: Embedding vector store.
    """
    embedding_model_name = "TaylorAI/gte-tiny"
    doc_embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)

    embedding_store = Chroma.from_documents(doc_chunks, doc_embeddings)

    return embedding_store


def process_and_embed(file_path: str):
    """
    Process the file and generate embeddings.
    :param file_path: Path to the file.
    :return: Embedding vector store.
    """
    chunks = extract_pdf_data(file_path)
    embedding_vector_store = generate_embeddings(chunks)
    return embedding_vector_store


def calculate_length(text):
    """
    Calculate the length of a text.
    :param text: Text to calculate the length of.
    :return: Length of the text.
    """
    encoder = tiktoken.get_encoding("cl100k_base")
    return len(encoder.encode(str(text)))


@sleep_and_retry
@limits(calls=REQUESTS, period=TIME_PERIOD)
async def generate_answer(file_path, user_query) -> str:
    """
    Generate an answer to a query based on a file's content.
    :param file_path: Path to the file.
    :param user_query: User's query.
    :return: Generated answer.
    """
    # Initialize the ChatOpenAI model
    chat_model = ChatOpenAI()

    # Load the embedding store from the given file path
    embed_store = process_and_embed(file_path)

    # Create a retriever using the embedding store
    data_retriever = embed_store.as_retriever(search_kwargs={"k": 7})

    # Set up the pipeline for generating the answer
    retrieval_setup = RunnableParallel(
        {"context": data_retriever, "question": RunnablePassthrough()}
    )
    processing_chain = retrieval_setup | chat_prompt | chat_model | StrOutputParser()

    # Generate the answer using the pipeline
    response = await processing_chain.ainvoke(user_query)

    # Return the generated answer
    return response
