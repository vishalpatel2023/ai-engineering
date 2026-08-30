from google import genai
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

    response = client.models.generate_content(
        model=LLM_MODEL,
        contents=prompt
    )

    return response.text