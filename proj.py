from web_scrape import scrape_website
from model import model  # Ensure you have the model initialized as per your setup
from tables_scrape import extract_tables_from_html
from ai_answers import answer_questions_with_ai
from extract_desc_info import extract_info_from_description
from ai_automated import automated_ai_assistant
description_test_case = """Scrape the list of highest grossing films from Wikipedia. It is at the URL:
https://en.wikipedia.org/wiki/List_of_highest-grossing_films

Answer the following questions:

1. How many $2 bn movies were released before 2020?
2. Which is the earliest film that grossed over $1.5 bn?
3. What's the correlation between the Rank and Peak?
4. Draw a scatterplot of Rank and Peak along with a dotted red regression line through it.
   Return as a base-64 encoded data URI, `"data:image/png;base64,iVBORw0KG..."` under 100,000 bytes.
"""

if description_test_case:
    print("Running automated AI assistant with the provided description...")
    results = automated_ai_assistant(description_test_case, model)

    print("\n--- Answers ---")
    if isinstance(results, dict) and "error" in results:
        print(f"Error: {results['error']}")
    else:
        for q, a in results.items():
            print(f"Q: {q}")
            print(f"A: {a}\n")
else:
    print("Please provide a description to test.")