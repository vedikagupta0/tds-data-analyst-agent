import getpass
import os
from langchain.chat_models import init_chat_model

os.environ["GOOGLE_API_KEY"] = "AIzaSyCbKIzFhKpKmSmexPDqwmgGjHit_KAvnOY"

from langchain.chat_models import init_chat_model

model = init_chat_model("gemini-2.0-flash", model_provider="google_genai")
import duckdb
print("Connecting to DuckDB (minimal test)...")
con = duckdb.connect("scraped_data.duckdb")
print("DuckDB connection established (minimal test).")