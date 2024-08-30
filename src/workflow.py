#src/workflow.py
from langgraph.graph import END, StateGraph, START
from src.nodes_and_edges import (
    query_rewriting_node,
    step_back_prompting_node,
    sub_query_decomposition_node,
    retrieval_node,
    summarization_node,
    final_generation_node
)

def create_workflow(agent_state_class):
    workflow = StateGraph(agent_state_class)
    
    # Define nodes and edges
    workflow.add_node("query_rewriting", query_rewriting_node)
    workflow.add_node("step_back_prompting", step_back_prompting_node)
    workflow.add_node("sub_query_decomposition", sub_query_decomposition_node)
    workflow.add_node("retrieval", retrieval_node)
    #workflow.add_node("summarization", summarization_node)
    workflow.add_node("final_generation", final_generation_node)
    
    # Define the flow of nodes
    workflow.add_edge(START, "query_rewriting")
    workflow.add_edge("query_rewriting", "step_back_prompting")
    workflow.add_edge("step_back_prompting", "sub_query_decomposition")
    workflow.add_edge("sub_query_decomposition", "retrieval")
    #workflow.add_edge("retrieval", "summarization")
    workflow.add_edge("retrieval", "final_generation")
    workflow.add_edge("final_generation", END)

    return workflow
