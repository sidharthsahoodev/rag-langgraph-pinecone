# src/agent_state.py
from typing import Annotated, Sequence, TypedDict, List, Optional
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    sub_queries: Optional[List[str]]  # List of sub-queries
    summarized_content: Optional[List[str]]  # Summarized content or retrieved documents
    final_response: Optional[str]  # Final generated response
    rewritten_query: Optional[str]
    initial_query: Optional[str]
