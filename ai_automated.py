# Assuming extract_info_from_description, scrape_website, extract_tables_from_html are defined in previous cells
# Assuming 'model' is initialized from a previous cell (ensure cell VSrE7F_eRdSB is executed)
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
import re
import json
from langchain_core.messages import HumanMessage, SystemMessage
from scipy import stats
from IPython.display import display
from web_scrape import scrape_website
from tables_scrape import extract_tables_from_html
from extract_desc_info import extract_info_from_description
from model import con 
def automated_ai_assistant(description, model):
    print(con)
    """
    Runs the AI assistant pipeline starting from a text description,
    including data manipulation, analysis, and visualization where requested,
    by using the AI to help generate and execute Python code.

    Args:
        description (str): The text description containing the task details (URL and questions).
        model: The initialized AI model (e.g., Gemini).

    Returns:
        dict: A dictionary where keys are questions and values are the AI's answers or results (e.g., plot URI),
              or an error message if the process failed.
    """
    print("Extracting info from description...")
    extracted_data = extract_info_from_description(description, model)

    if "error" in extracted_data:
        return {"error": f"Failed to extract info from description: {extracted_data['error']}"}

    url_to_scrape = extracted_data.get('url')
    user_questions = extracted_data.get('questions')

    if not url_to_scrape or not user_questions:
        return {"error": "Could not extract URL or questions from the description."}

    print(f"Scraping website: {url_to_scrape}")
    html_content = scrape_website(url_to_scrape)

    if not html_content:
        return {"error": f"Failed to scrape website: {url_to_scrape}"}

    print("Extracting tables...")
    dataframes = extract_tables_from_html(html_content, description, model)
    print("Tables extracted.")

    if not dataframes:
        print("No tables found, returning error.")
        return {"error": f"No tables found on the page: {url_to_scrape}"}

    data_df = dataframes[0].copy()
    print("DataFrame copied.")
    # duckdb_db_path = "scraped_data.duckdb"
    # print('01')
    # if os.path.exists(duckdb_db_path):
    #     print(f"Removing existing DuckDB file: {duckdb_db_path}")
    #     os.remove(duckdb_db_path)
    # try:
    #     import time
    #     time.sleep(5)

    #     print("Connecting to DuckDB...")
    #     con = duckdb.connect(duckdb_db_path)
    print("DuckDB connection established.")

    print('02')
    print("About to store DataFrame in DuckDB...")
    con.register('data_df', data_df)
    con.execute("CREATE OR REPLACE TABLE scraped_table AS SELECT * FROM data_df")
    data_df = con.execute("SELECT * FROM scraped_table").df()
    print("Tables in DuckDB now:")
    print(data_df.head())
    print(con.execute("SHOW TABLES").df())
    print("DataFrame stored in DuckDB.")
 

    # Get sample and schema for AI context
    sample_df = con.execute("SELECT * FROM scraped_table LIMIT 5").df()
    schema = con.execute("DESCRIBE scraped_table").df().to_string()
    print("Sample and schema extracted.")

    data_string_for_ai = sample_df.to_string()
    data_info_for_ai_string = schema

    prompt_text = f"""
You are a helpful assistant that generates Python code to analyze and visualize data based on user questions.
You have access to a pandas DataFrame named `data_df`.
Based on the following questions and the provided dataframe snippet and info, generate Python code to answer each question.
Store the answer for each question in a dictionary called `question_answers`, where keys are the question strings and values are the answers or results.
For questions requiring data analysis (e.g., correlation, counts, filtering), generate code that performs the analysis and stores the answer in `question_answers`.
For questions requiring visualization (e.g., scatterplot), generate code to create the plot, save it as a base64 encoded data URI, and store the data URI in `question_answers`.
Ensure the generated code is valid Python and can be executed.
Include necessary data cleaning steps in the generated code, such as converting columns to numeric types, handling potential errors during conversion, and removing irrelevant characters (like '$', ',', 'T').
For the scatterplot question, make sure to handle potential non-numeric values in the 'Rank' and 'Peak' columns before attempting to calculate correlation or plot.

Dataframe snippet (first few rows and columns):
{data_string_for_ai}

Dataframe columns and their inferred data types:
{data_info_for_ai_string}

Questions to answer:
{user_questions}

Generate the Python code for each question below:
"""

    messages = [
        SystemMessage("You are a helpful assistant that generates Python code to answer questions based on a pandas DataFrame. Store results in a dictionary called `question_answers`."),
        HumanMessage(prompt_text),
    ]


    generated_code = ""
    try:
        print("Generating code for analysis and visualization using AI...")
        response = model.invoke(messages)
        generated_code = response.content

        # Remove all markdown code fences (``` or ```python)
        generated_code = re.sub(r"^```.*?$", "", generated_code, flags=re.MULTILINE)
        generated_code = generated_code.strip()

        # Remove any redefinition of question_answers in the generated code
        generated_code = re.sub(r"question_answers\s*=\s*{}", "", generated_code)

     

        # Prepare the environment for executing the generated code
        exec_globals = {
            'con': con,  # DuckDB connection
            'pd': pd,
            'np': np,
            'plt': plt,
            'io': io,
            'base64': base64,
            'display': display,
            'stats': stats,
            'data_df':data_df,
            'question_answers': {}
        }
        exec_locals = {}

        # Execute the generated code
        exec(generated_code, exec_globals, exec_locals)

        # Retrieve the answers from the question_answers dictionary
        results = exec_globals.get('question_answers', {})

        # --- Save results as JSON and images if present ---
        with open("results.json", "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2, default=convert_np)

        # Save any base64 images to files and update the answer to the filename
        for q, a in list(results.items()):
            if isinstance(a, str) and a.startswith("data:image/png;base64,"):
                img_data = a.split(",", 1)[1]
                img_filename = f"{q[:30].replace(' ', '_').replace('/', '_')}.png"
                with open(img_filename, "wb") as img_file:
                    img_file.write(base64.b64decode(img_data))
                results[q] = img_filename

        # Update the JSON file with image filenames if any were changed
        with open("results.json", "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2, default=convert_np)
    except Exception as e:
        print("Exception occurred:", e)
        results = {}
        results["Code Generation/Execution Error"] = f"An error occurred: {e}\n\nGenerated code:\n{generated_code}"
        return results

    return results

def convert_np(obj):
    import numpy as np
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    return str(obj)