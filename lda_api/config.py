import os

from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("LDA_API")
BASE_URL = "https://lda.senate.gov/api/v1/"  # Confirm exact URL
HEADERS = {"Authorization": f"Token {API_KEY}"}
DEFAULT_PAGE_SIZE = 25
