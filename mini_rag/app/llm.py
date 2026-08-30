from google import genai

from app.config import GEMINI_API_KEY, LLM_MODEL


client = genai.Client(api_key=GEMINI_API_KEY)


def generate_answer(
    question: str,
    context: str
) -> str:

    prompt = f"""
You are a helpful question-answering assistant.

Answer the user's question using ONLY the provided context.

If the context does not contain enough information,
say that you do not have enough information to answer.

Context:
{context}

Question:
{question}
"""

    response = client.models.generate_content(
        model=LLM_MODEL,
        contents=prompt
    )

    return response.text