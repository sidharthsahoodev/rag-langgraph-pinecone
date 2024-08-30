from src.workflow import create_workflow
from src.agent_state import AgentState

def initialize_workflow():
    # Create and compile the workflow graph
    workflow = create_workflow(AgentState)
    graph = workflow.compile()
    return graph
