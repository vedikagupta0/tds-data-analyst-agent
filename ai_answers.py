from langchain_core.messages import HumanMessage, SystemMessage

def answer_questions_with_ai(dataframes, questions, model):
    """
    Uses the AI model to answer questions based on the extracted dataframes.

    Args:
        dataframes (list): A list of pandas DataFrames containing the extracted data.
        questions (list): A list of strings representing the questions to ask.
        model: The initialized AI model (e.g., Gemini).

    Returns:
        dict: A dictionary where keys are questions and values are the AI's answers.
    """
    if not dataframes:
        return {"error": "No data available to answer questions."}

    # Convert dataframes to string format that can be included in the prompt
    # For simplicity, we'll convert the first dataframe to a string.
    # For more complex data, you might need a more sophisticated representation.
    data_string = dataframes[0].to_string()

    answers = {}
    for question in questions:
        prompt_text = f"""Based on the following data, answer the question:

Data:
{data_string}

Question: {question}

Provide a concise answer based *only* on the provided data.
"""

        messages = [
            SystemMessage("You are a helpful assistant that answers questions based on provided data."),
            HumanMessage(prompt_text),
        ]

        try:
            response = model.invoke(messages)
            answers[question] = response.content
        except Exception as e:
            answers[question] = f"Error getting answer from AI: {e}"
            continue

    return answers
