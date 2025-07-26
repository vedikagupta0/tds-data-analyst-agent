from langchain_core.messages import HumanMessage, SystemMessage
import json
import re

def extract_info_from_description(description, model):
    """
    Uses the AI model to extract URL and questions from a text description.

    Args:
        description (str): The text description containing the task details.
        model: The initialized AI model (e.g., Gemini).

    Returns:
        dict: A dictionary containing the extracted 'url' and 'questions',
              or an error message if parsing failed.
    """
    prompt_text = f"""Analyze the following description of a data analysis task.
Extract the website URL and the list of questions to be answered based on the data from that URL.
Output the extracted information as a JSON object with two keys: "url" (string) and "questions" (array of strings).
Ensure the output is valid JSON and contains only the JSON object.

Description:
{description}
"""

    messages = [
        SystemMessage("You are a helpful assistant that extracts structured information from text."),
        HumanMessage(prompt_text),
    ]

    try:
        response = model.invoke(messages).content
        # Clean the response content by removing markdown code block (```json) and ``` wrappers
        cleaned_response_content = re.sub(r'```json|```', '', response).strip()
        extracted_info = json.loads(cleaned_response_content)

        # Validate the extracted information
        if 'url' in extracted_info and 'questions' in extracted_info and isinstance(extracted_info['questions'], list):
            return extracted_info
        else:
            return {"error": "AI model did not return the expected JSON format."}

    except json.JSONDecodeError:
        return {"error": "Could not parse the AI model's response as JSON."}
    except Exception as e:
        return {"error": f"An error occurred: {e}"}

