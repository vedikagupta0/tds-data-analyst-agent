import pandas as pd
from bs4 import BeautifulSoup
from IPython.display import display # Import display
from langchain_core.messages import HumanMessage, SystemMessage # Import necessary classes for AI model
import re

def extract_tables_from_html(html_content, description, model):

    if html_content is None:
        print("No HTML content provided to extract tables.")
        return []

    soup = BeautifulSoup(html_content, 'html.parser')
    tables = soup.find_all('table')

    if not tables:
        print("No tables found in the HTML content.")
        return []

    print(f"Found {len(tables)} potential tables.")

    # Use the AI model to identify the most relevant table(s)
    table_summaries = []
    for i, table in enumerate(tables):
        # Create a summary of the table (e.g., first few rows, headers) to send to the AI
        # This is a simplified summary; a more sophisticated approach might be needed
        table_html_snippet = str(table)[:1000]  # Send up to 1000 characters of the table HTML
        table_summaries.append(f"Table {i+1}:\n{table_html_snippet}...\n")

    prompt_text = f"""Analyze the following description and the provided table summaries.
Identify the table(s) that are most relevant to answering the questions in the description.
Respond with a comma-separated list of the table numbers (e.g., "1, 3, 5") that are most relevant.
If no tables seem relevant, respond with "none".

Description:
{description}

Table Summaries:
{''.join(table_summaries)}  # Corrected multiline string handling here
"""

    messages = [
        SystemMessage("You are a helpful assistant that identifies relevant tables based on a task description and table summaries."),
        HumanMessage(prompt_text),
    ]

    relevant_table_indices = []
    try:
        response = model.invoke(messages).content
        # Attempt to parse the AI's response to get the table numbers
        # This assumes the AI responds with a comma-separated list of numbers
        table_numbers_str = response.strip().lower()
        if table_numbers_str != "none":
            try:
                # Extract numbers from the string, handling potential non-digit characters
                relevant_table_indices = [int(num) - 1 for num in re.findall(r'\d+', table_numbers_str)]
            except ValueError:
                print(f"Warning: Could not parse AI response for relevant tables: {response}")
                # If parsing fails, fall back to assuming the first table is relevant (or handle as needed)
                relevant_table_indices = [0]  # Fallback to first table
        else:
            print("AI determined no tables are relevant.")
            return []  # Return empty list if AI says no tables are relevant


    except Exception as e:
        print(f"Error getting relevant tables from AI: {e}")
        print("Falling back to processing the first table.")
        relevant_table_indices = [0]


    dataframes = []
    print(f"Processing {len(relevant_table_indices)} potentially relevant tables.")
    for i in relevant_table_indices:
        if 0 <= i < len(tables):  # Ensure index is within bounds
            table = tables[i]
            try:
                # Use StringIO for newer pandas versions to read from string
                from io import StringIO
                df = pd.read_html(StringIO(str(table)))[0]
                dataframes.append(df)

                # Display the head of the dataframe
                print(f"\nHead of Extracted Table {i+1}:")
                display(df.head())

            except Exception as e:
                print(f"Could not convert Table {i+1} to DataFrame: {e}")
                continue
        else:
            print(f"Warning: AI returned out-of-bounds table index: {i+1}")


    return dataframes
