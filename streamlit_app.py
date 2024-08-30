import streamlit as st
import logging
from workflow_setup import initialize_workflow
from llm_utils import auto_populate_fields
from state_management import initialize_session_state, get_session_state

# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Streamlit App
st.title("🦜🔗 Langchain Agentic RAG Search")

# Initialize workflow
graph = initialize_workflow()

# Initialize session state
initialize_session_state()

# User input outside the form for auto-populate
user_input = st.text_area("Enter your query:", help="Describe your query or the information you seek in detail.")

# Button to auto-populate the fields outside the form
if st.button("Auto Populate"):
    if user_input.strip():
        st.session_state.auto_populated_data = auto_populate_fields(user_input)
        if st.session_state.auto_populated_data:
            st.info("Fields auto-populated based on your query.")
    else:
        st.warning("Please enter a query first to auto-populate the fields.")

# Define the available content types
content_type_options = [
    "General Information", 
    "Technical Explanation", 
    "Troubleshooting Guide", 
    "Step-by-Step Tutorial"
]

# Get session state data
data = get_session_state(content_type_options)

# User input form with fields focused on understanding the user's query
with st.form("query_form"):
    # Populate the fields with auto-suggested values or leave them for manual entry
    purpose = st.text_area("Purpose/Use Case:", value=data['purpose'], help="What is the main objective of your query? What do you hope to achieve?")
    context = st.text_area("Context:", value=data['context'], help="Provide any background information or context that will help in understanding your query better.")
    expected_outcome = st.text_area("Expected Outcome:", value=data['expected_outcome'], help="What are you expecting as a result of this query? What kind of information or solution are you looking for?")
    keywords = st.text_input("Keywords:", value=data['keywords'], help="List any specific terms or concepts that are central to your query.")
    content_type = st.selectbox("Content Type:", content_type_options, index=content_type_options.index(data['normalized_content_type']), help="What type of content do you think would best answer your query?")
    sample_questions = st.text_area("Sample Questions:", value=data['sample_questions'], help="List some specific questions that relate to your query. These should help in narrowing down the information you're seeking.")
    
    submit_button = st.form_submit_button("Submit")

    if submit_button and user_input:
        st.write(f"User Input: {user_input}")
        st.write(f"Purpose/Use Case: {purpose}")
        st.write(f"Context: {context}")
        st.write(f"Expected Outcome: {expected_outcome}")
        st.write(f"Keywords: {keywords}")
        st.write(f"Content Type: {content_type}")
        st.write(f"Sample Questions: {sample_questions}")

        # Construct the message to improve LLM response logic
        message_content = (
            f"**Query:** {user_input}\n"
            f"**Purpose:** {purpose}\n"
            f"**Context:** {context}\n"
            f"**Expected Outcome:** {expected_outcome}\n"
            f"**Sample Questions:** {sample_questions}"
        )

        # Combine inputs into messages and metadata
        inputs = {
            "messages": [("user", message_content)],
            "metadata": {
                "keywords": keywords.split(', '),
                "content_type": content_type,
            }
        }

        final_state = None
        for output in graph.stream(inputs):
            final_state = output  # Store the last state
            for key, value in output.items():
                st.write(f"Output from node '{key}':")
                st.write(value)

        if final_state:
            # Display the final response if it exists
            if "final_response" in final_state.get("final_generation", {}):
                st.subheader("Final Response:")
                st.write(final_state["final_generation"]["final_response"])
            else:
                st.error("No final response generated.")
        else:
            st.error("No final state available.")
