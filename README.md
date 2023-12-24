# Question-Answering Bot with Langchain

## Problem Statement

Create a powerful Question-Answering (QA) bot using the Langchain framework, capable of answering questions based on the content of a document. This bot will utilize the advanced capabilities of the OpenAI GPT-3.5 Turbo model. To achieve this, follow the steps outlined in the Langchain documentation: [Langchain Documentation](https://python.langchain.com/docs/use_cases/question_answering/).

## Implementation

Implement the backend API using either FastAPI, Flask, or Django. Ensure that the API supports two types of input for answering questions:

1. **JSON**: File containing a list of questions.
2. **PDF**: File containing the document over which the questions will be answered.

## Input Requirements

Provide two input files for the QA bot:

1. A file containing a list of questions (JSON).
2. A file containing the document over which the questions will be answered PDF.

## Output Format

The output should be a structured JSON blob that pairs each question with its corresponding answer. The format should be as follows:

```json
[
  {"question": "answer"},
  {"question": "answer"},
  {"question": "answer"}
]
```

## Technology Requirements

Make use of the following technologies for the implementation:

- Python 3.x
- LangChain (Python)
- OpenAI (gpt-3.5-turbo model)
- VectorDB

## Getting Started

Clone the project first and navigate to the project directory, then:

1. Install the required dependencies:
   ```bash
   brew install tesseract
   brew install poppler
   pip install -r requirements.txt
   ```
2. Create a new .env  file in the root directory with your OPEN_AI_TOKEN:
   ```
   OPENAI_API_KEY = "<your_token>"
   ```

3. Run the application:
   ```bash
   python app.py
   ```

4. Access the API endpoints to perform question-answering based on the provided input files.

Note: Known Issue

If you encounter an issue with NLTK punkt not found, do the following, in a separate terminal window, open the python IDLE via typing `python`:

```
import nltk

nltk.download('punkt')

```

## Directory Structure

```
langchain-qna
|   .env
|   .gitignore
|   Dockerfile
|   img.png
|   img_1.png
|   README.md
|   requirements.txt
|
+---app
|   |   lib.py
|   |   main.py
|   |   __init__.py
|   |
|   +---config
|   |       constants.py
|   |       qna_template.py
|   |       __init__.py
|   |
|   \---__pycache__
|           lib.cpython-311.pyc
|           main.cpython-311.pyc
|           __init__.cpython-311.pyc
|
\---assets
        test.json
        village.pdf

```



## API Endpoints
### http://<server_url:<port>/qna/

![image](https://github.com/gauravgs/langchain-qna-bot/assets/40033910/3f677a64-2545-46a0-911d-82dae7a6ae0f)


## Example Usage
```
   curl -X 'POST' \
  'http://127.0.0.1:8000/qna/' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'query_file=@test.json;type=application/json' \
  -F 'content_file=@village.pdf;type=application/pdf'
```
![image](https://github.com/gauravgs/langchain-qna-bot/assets/40033910/02258156-52d7-41c9-8cc3-ece603f26521)





## _**Note: Sample files can be found under dir /assets_
