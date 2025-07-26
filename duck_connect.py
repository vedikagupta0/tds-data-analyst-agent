import duckdb
def connect_db():
    try:
        con = duckdb.connect("scraped_data.duckdb")
        print("Connected to DuckDB!")
        # Check if the table exists
        con.execute('SELECT * FROM scraped_table LIMIT 1')
        print("Table 'scraped_table' exists.")
        return con
    except Exception as e:
        print("DuckDB connect error:", e)
        return {"error": f"DuckDB connect error: {e}"}
print(connect_db())
