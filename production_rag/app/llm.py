import time

from google import genai
from google.genai.errors import ServerError

from app.config import GEMINI_API_KEY, LLM_MODEL


client = genai.Client(api_key=GEMINI_API_KEY)


def generate_answer(question, context):

    prompt = f"""
You are a helpful AI assistant.

Answer the user's question using ONLY the provided context.

If the answer is not present in the context, say:
"I don't know based on the provided documents."

Context:
{context}

User Question:
{question}

Answer:
"""

    max_retries = 3

    for attempt in range(max_retries):

        try:

            response = client.models.generate_content(
                model=LLM_MODEL,
                contents=prompt
            )

            return response.text

        except ServerError as error:

            if attempt < max_retries - 1:

                print(
                    f"\nGemini temporarily unavailable. "
                    f"Retrying... ({attempt + 1}/{max_retries})"
                )

                time.sleep(2)

            else:

                return (
                    "The retrieved documents were found correctly, "
                    "but the Gemini model is temporarily unavailable."
                )