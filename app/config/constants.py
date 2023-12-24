"""
This module used in main.py.
"""

import os
from dotenv import load_dotenv

# Load the .env file
load_dotenv()

#  OpenAI API key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


# Define the rate limit (e.g., 10 requests per minute)
REQUESTS = 10
TIME_PERIOD = 60
