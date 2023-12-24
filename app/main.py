"""
Main module
Serves the API and talks to the lib for QnA
"""
import os
import shutil
import json
from typing import Dict
import openai
import uvicorn
from fastapi import FastAPI, UploadFile, File

from app.lib import generate_answer
from app.config.constants import OPENAI_API_KEY

# Check if OPENAI_API_KEY environment variable is set, and raise an error if not
if OPENAI_API_KEY is None:
    raise ValueError("OPENAI_API_KEY environment variable is not set")

# Set the environment variable and OpenAI API key
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
openai.api_key = OPENAI_API_KEY

# Initialize the FastAPI app
app = FastAPI(title="Langchain QnA Chatbot", docs_url="/")


@app.post("/qna/")
async def process_queries(
    query_file: UploadFile = File(...), content_file: UploadFile = File(...)
) -> Dict[str, str]:
    """
    Process uploaded files to generate text responses for queries.
    :param query_file: File containing queries in JSON format.
    :param content_file: Document file to base the answers on.
    :return: Dictionary of queries and their corresponding answers.
    """
    # Load queries from the uploaded JSON file
    with open(query_file.filename, "wb") as temp_buffer:
        shutil.copyfileobj(query_file.file, temp_buffer)

    with open(query_file.filename, "r") as file:
        queries_json = json.load(file)
        queries = queries_json.get("questions", [])

    # Save the uploaded document to a temporary path
    temp_doc_path = f"temp_{content_file.filename}"
    with open(temp_doc_path, "wb") as temp_buffer:
        shutil.copyfileobj(content_file.file, temp_buffer)

    # Generate answers for each query
    responses = {}
    for query in queries:
        response = await generate_answer(temp_doc_path, query)
        responses[query] = response

    # Clean up temporary files
    os.remove(query_file.filename)
    os.remove(temp_doc_path)

    return responses


# Run the application with Uvicorn
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
