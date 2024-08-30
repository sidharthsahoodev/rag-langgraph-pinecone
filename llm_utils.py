import json
import logging
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Set up logging
logger = logging.getLogger(__name__)
MODEL_NAME = "gpt-4o-mini"

# LLM and Prompt Setup for Auto Populate
llm = ChatOpenAI(model_name=MODEL_NAME, temperature=0)
output_parser = StrOutputParser()  # To parse the LLM output

auto_populate_prompt_template = PromptTemplate(
    template="""Based on the user's query below, predict the following:
    1. Purpose/Use Case: What is the user likely trying to accomplish?
    2. Context: What background information or context is relevant to this query?
    3. Expected Outcome: What result or solution is the user likely expecting?
    4. Keywords: List key terms that represent the core concepts.
    5. Content Type: What type of content (e.g., tutorial, explanation) would best answer the query?
    6. Sample Questions: Generate specific questions that relate to the query.

    User Query: {user_input}

    Output (JSON format):
    {{
        "Purpose": "<Predicted Purpose>",
        "Context": "<Predicted Context>",
        "Expected Outcome": "<Predicted Expected Outcome>",
        "Keywords": "<Predicted Keywords>",
        "Content Type": "<Predicted Content Type>",
        "Sample Questions": "<Predicted Sample Questions>"
    }}
    """,
    input_variables=["user_input"]
)

def auto_populate_fields(user_input):
    try:
        # Create a sequence of the prompt template and LLM
        sequence = auto_populate_prompt_template | llm | output_parser
        
        # Run the sequence with the user input
        response = sequence.invoke({"user_input": user_input})
        
        # Log the raw response to check its content
        logger.info(f"Raw LLM response: {response}")

        # Strip out the ```json markers if they exist
        response = response.strip().strip('```').strip('json').strip()

        # Parse JSON from the response
        parsed_data = json.loads(response)
        return parsed_data
    except json.JSONDecodeError as e:
        logger.error(f"JSON decoding error: {str(e)}")
        return {}
    except Exception as e:
        logger.error(f"Error during auto population: {str(e)}")
        return {}
