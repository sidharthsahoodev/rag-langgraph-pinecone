#src/query_transformations.py
import logging
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from src.configs.config import OPENAI_API_KEY

MODEL_NAME = "gpt-4o-mini"

# Set up logging
logger = logging.getLogger(__name__)

# Query Rewriting
re_write_llm = ChatOpenAI(temperature=0, model_name=MODEL_NAME, openai_api_key=OPENAI_API_KEY)
query_rewrite_template = """You are an AI assistant tasked with reformulating user queries to improve retrieval in a RAG system. 
Given the original query, rewrite it to be more specific, detailed, and likely to retrieve relevant information.

Original query: {original_query}

Rewritten query:"""
query_rewrite_prompt = PromptTemplate(input_variables=["original_query"], template=query_rewrite_template)
query_rewriter = query_rewrite_prompt | re_write_llm

def rewrite_query(original_query: str) -> str:
    try:
        logger.info(f"Rewriting query: {original_query}")
        response = query_rewriter.invoke(original_query)
        logger.info(f"Rewritten query: {response.content}")
        return response.content
    except Exception as e:
        logger.error(f"Error in query rewriting: {str(e)}")
        return original_query  # Fallback to the original query if an error occurs

# Step-back Prompting
step_back_llm = ChatOpenAI(temperature=0, model_name=MODEL_NAME)
step_back_template = """You are an AI assistant tasked with generating broader, more general queries to improve context retrieval in a RAG system.
Given the original query, generate a step-back query that is more general and can help retrieve relevant background information.

Original query: {original_query}

Step-back query:"""
step_back_prompt = PromptTemplate(input_variables=["original_query"], template=step_back_template)
step_back_chain = step_back_prompt | step_back_llm

def generate_step_back_query(original_query: str) -> str:
    try:
        logger.info(f"Generating step-back query for: {original_query}")
        response = step_back_chain.invoke(original_query)
        logger.info(f"Step-back query: {response.content}")
        return response.content
    except Exception as e:
        logger.error(f"Error in step-back query generation: {str(e)}")
        return original_query  # Fallback to the original query if an error occurs

# Sub-query Decomposition
sub_query_llm = ChatOpenAI(temperature=0, model_name=MODEL_NAME)
subquery_decomposition_template = """You are an AI assistant tasked with breaking down complex queries into simpler sub-queries for a RAG system.
Given the original query, decompose it into 2-4 simpler sub-queries that, when answered together, would provide a comprehensive response to the original query.

Original query: {original_query}

example: What are the impacts of climate change on the environment?

Sub-queries:
1. What are the impacts of climate change on biodiversity?
2. How does climate change affect the oceans?
3. What are the effects of climate change on agriculture?
4. What are the impacts of climate change on human health?"""
subquery_decomposition_prompt = PromptTemplate(input_variables=["original_query"], template=subquery_decomposition_template)
subquery_decomposer_chain = subquery_decomposition_prompt | sub_query_llm

def decompose_query(original_query: str) -> list[str]:
    try:
        logger.info(f"Decomposing query: {original_query}")
        response = subquery_decomposer_chain.invoke(original_query)
        sub_queries = [q.strip() for q in response.content.split('\n') if q.strip() and not q.strip().startswith('Sub-queries:')]
        logger.info(f"Decomposed sub-queries: {sub_queries}")
        return sub_queries
    except Exception as e:
        logger.error(f"Error in query decomposition: {str(e)}")
        return []  # Return an empty list if an error occurs
