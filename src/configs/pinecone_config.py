# src/pinecone_config.py
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import Pinecone as LangchainPinecone
from langchain_openai import OpenAIEmbeddings
from langchain.tools.retriever import create_retriever_tool
from .config import PINECONE_API_KEY, PINECONE_INDEX, PINECONE_ENVIRONMENT, PINECONE_DIMENSION, PINECONE_CLOUD, PINECONE_REGION, OPENAI_API_KEY

# Initialize Pinecone client
pc = Pinecone(
    api_key=PINECONE_API_KEY,
    environment=PINECONE_ENVIRONMENT
)

# Check if the index exists, otherwise create it
if PINECONE_INDEX not in pc.list_indexes().names():
    pc.create_index(
        name=PINECONE_INDEX,
        dimension=PINECONE_DIMENSION,
        metric='cosine',
        spec=ServerlessSpec(
            cloud=PINECONE_CLOUD,
            region=PINECONE_REGION
        )
    )

pinecone_index = pc.Index(PINECONE_INDEX)

# Initialize embeddings
embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY, model='text-embedding-3-large')

# Create LangChain Pinecone vector store from existing index
vector_store = LangchainPinecone.from_existing_index(
    index_name=PINECONE_INDEX,
    embedding=embeddings
)
