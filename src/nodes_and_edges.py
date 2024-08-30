# src/nodes_and_edges

import logging
from langchain_core.messages import HumanMessage
from src.query_transformations import rewrite_query, generate_step_back_query, decompose_query
from src.configs.pinecone_config import vector_store
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from src.agent_state import AgentState

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_NAME = "gpt-4o-mini"

def query_rewriting_node(state: AgentState) -> AgentState:
    logger.info("Starting Query Rewriting Node.")
    original_query = state["messages"][0].content

    # Query Rewriting
    state["initial_query"] = original_query
    rewritten_query = rewrite_query(original_query)
    logger.info(f"Rewritten Query: {rewritten_query}")
    state["messages"].append(HumanMessage(content=rewritten_query))
    state["rewritten_query"] = HumanMessage(content=rewritten_query)

    logger.info(f"State after query rewriting: {state}")
    return state

def step_back_prompting_node(state: AgentState) -> AgentState:
    logger.info("Starting Step-back Prompting Node.")
    rewritten_query = state["messages"][-1].content

    # Step-back Prompting
    step_back_query = generate_step_back_query(rewritten_query)
    logger.info(f"Step-back Query: {step_back_query}")
    state["messages"].append(HumanMessage(content=step_back_query))

    logger.info(f"State after step-back prompting: {state}")
    return state

def sub_query_decomposition_node(state: AgentState) -> AgentState:
    logger.info("Starting Sub-query Decomposition Node.")
    rewritten_query = state["messages"][-2].content

    # Sub-query Decomposition
    sub_queries = decompose_query(rewritten_query)
    sub_queries = [query.strip() for query in sub_queries if query.strip() and not query.startswith("Sub-queries for the original query:")]
    logger.info(f"Filtered Sub-queries: {sub_queries}")

    # Save sub-queries in the state
    state["sub_queries"] = sub_queries

    logger.info(f"State after sub-query decomposition: {state}")
    return state

def retrieval_node(state: AgentState) -> AgentState:
    logger.info("Starting Retrieval Node.")
    
    sub_queries = state.get("sub_queries", [])
    if not sub_queries:
        logger.error("Sub-queries are missing or empty, cannot proceed with retrieval.")
        return state
    
    summarized_content = []
    for sub_query in sub_queries:
        logger.info(f"Retrieving documents for sub-query: {sub_query}")
        search_results = vector_store.similarity_search(query=sub_query, k=2)

        if not search_results:
            logger.warning(f"No documents retrieved for sub-query: {sub_query}")
            continue

        docs = "\n\n".join([doc.page_content for doc in search_results])
        logger.info(f"Documents retrieved for sub-query '{sub_query}': {docs}")

        summarized_content.append(docs)

    state["summarized_content"] = summarized_content

    logger.info(f"State after retrieval: {state}")
    return state

def summarization_node(state: AgentState) -> AgentState:
    logger.info("Starting Summarization Node.")
    
    summarized_content = state.get("summarized_content", [])
    if not summarized_content:
        logger.error("No documents to summarize.")
        return state

    summarized_context = "\n\n".join(summarized_content)
    logger.info(f"Summarized context: {summarized_context}")

    summarized_output = []
    for sub_query, docs in zip(state["sub_queries"], summarized_content):
        summarization_prompt = PromptTemplate(
            template="""You are an AI assistant tasked with summarizing the content retrieved for each sub-query. 
            Given the retrieved documents, provide a concise summary that captures the essential information.

            Sub-query: {sub_query}
            Retrieved Documents: {documents}

            Summary:""",
            input_variables=["sub_query", "documents"],
        )

        summary_llm = ChatOpenAI(model_name=MODEL_NAME, temperature=0, streaming=True)
        summary_chain = summarization_prompt | summary_llm | StrOutputParser()

        try:
            summary_response = summary_chain.invoke({"sub_query": sub_query, "documents": docs})
            summary_text = summary_response.content if hasattr(summary_response, 'content') else str(summary_response)

            if not summary_text.strip():
                logger.warning(f"Summarization failed for sub-query: {sub_query}")
                continue

            summarized_output.append(summary_text)
            logger.info(f"Summary for sub-query '{sub_query}': {summary_text}")
        except Exception as e:
            logger.error(f"Error during summarization for sub-query '{sub_query}': {str(e)}")

    state["summarized_content"] = summarized_output

    logger.info(f"State after summarization: {state}")
    return state

def final_generation_node(state: AgentState) -> AgentState:
    logger.info("Generating the final answer based on summarized content.")
    
    summarized_content = state.get("summarized_content", [])
    
    if not summarized_content or len(summarized_content) == 0:
        logger.error("No summarized content available. Exiting.")
        return state
    
    summarized_context = "\n\n".join(summarized_content)

    prompt = PromptTemplate(
    template="""You are an expert AI assistant specializing in generating accurate, functional code and creating file structures based on user specifications. Your task is to produce code snippets, file templates, or complete file structures that align precisely with the user's requirements. Follow these guidelines:

    1. **Technical Accuracy**: Ensure that the code you generate is syntactically correct, follows best practices, and is ready for execution or integration.
    2. **Contextual Relevance**: Your generated output must align with the context provided in the summarized content. Include only relevant code or file structures that directly address the user's query.
    3. **Comprehensive Coverage**: Address all aspects of the user's request. If multiple files or code components are needed, ensure each part is clearly defined and well-organized.
    4. **Clarity and Readability**: Write code and file structures that are easy to understand and maintain. Use appropriate comments and consistent formatting.
    5. **Error Handling**: Where applicable, include basic error handling or validation checks to make the code robust and reliable.
    6. **Explicit Limitations**: If the provided context does not contain enough information to generate the required output, clearly state what additional details are needed or explain any assumptions made.
    7. **Professional Standards**: Maintain a professional and formal tone in any accompanying explanations or comments.

    Use the following question and context to generate the appropriate code or file structure:
    Question: {question}
    Summarized Context: {context}
    Output:""",
    input_variables=["question", "context"],
    )

    llm = ChatOpenAI(model_name=MODEL_NAME, temperature=0, streaming=True)
    rag_chain = prompt | llm | StrOutputParser()

    question = state["initial_query"]

    logger.info("Invoking the RAG chain.")
    try:
        response = rag_chain.invoke({"context": summarized_context, "question": question})
        
        logger.info("Final response generated.")
        state["final_response"] = response.content if hasattr(response, 'content') else str(response)
        
        state["messages"].append(HumanMessage(content=state["final_response"]))
    except Exception as e:
        logger.error(f"Error during final response generation: {str(e)}")

    logger.info(f"Final state after generation: {state}")
    return state
