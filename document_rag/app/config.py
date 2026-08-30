import os

from dotenv import load_dotenv


load_dotenv()


GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

EMBEDDING_MODEL = "gemini-embedding-001"

LLM_MODEL = "gemini-3.5-flash"

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


TOP_K = 3

EMBEDDINGS_FILE = "storage/embeddings.json"


if not GEMINI_API_KEY:
    raise ValueError(
        "GEMINI_API_KEY is missing. "
        "Add it to your .env file."
    )