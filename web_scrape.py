import requests
from bs4 import BeautifulSoup

def scrape_website(url):
    
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
        return response.text
    except requests.exceptions.RequestException as e:
        print(f"Error scraping website: {e}")
        return None