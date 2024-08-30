import streamlit as st

def initialize_session_state():
    if 'auto_populated_data' not in st.session_state:
        st.session_state.auto_populated_data = {}

def get_session_state(content_type_options):
    # Normalize the "Content Type" to match the options
    content_type = st.session_state.auto_populated_data.get("Content Type", "General Information")
    normalized_content_type = next((ct for ct in content_type_options if ct.lower() in content_type.lower()), "General Information")
    
    return {
        'purpose': st.session_state.auto_populated_data.get("Purpose", ""),
        'context': st.session_state.auto_populated_data.get("Context", ""),
        'expected_outcome': st.session_state.auto_populated_data.get("Expected Outcome", ""),
        'keywords': st.session_state.auto_populated_data.get("Keywords", ""),
        'normalized_content_type': normalized_content_type,
        'sample_questions': st.session_state.auto_populated_data.get("Sample Questions", "")
    }
