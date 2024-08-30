# src/config.py
from dotenv import load_dotenv
import os

load_dotenv()

# Get OpenAI and Pinecone API keys and settings from the environment
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
PINECONE_DIMENSION = int(os.getenv("PINECONE_DIMENSION"))  
PINECONE_CLOUD = os.getenv("PINECONE_CLOUD")  
PINECONE_REGION = os.getenv("PINECONE_REGION")  


if not OPENAI_API_KEY:
    raise ValueError("OpenAI API Key is not set in the environment.")

if not PINECONE_API_KEY or not PINECONE_INDEX:
    raise ValueError("Pinecone API Key or Index is not set in the environment.")
