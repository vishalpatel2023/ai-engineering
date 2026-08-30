from app.config import TOP_K
from app.embeddings import get_embedding
from app.llm import generate_answer
from app.vector_store import search


def retrieve(
    question: str,
    documents: list[dict],
    top_k: int = TOP_K
) -> list[dict]:

    query_vector = get_embedding(question)

    results = search(
        query_vector=query_vector,
        documents=documents,
        top_k=top_k
    )

    return results


def build_context(
    results: list[dict]
) -> str:

    return "\n\n".join(
        result["text"]
        for result in results
    )


def answer_question(
    question: str,
    documents: list[dict]
):

    results = retrieve(
        question,
        documents
    )

    context = build_context(results)

    answer = generate_answer(
        question,
        context
    )

    return results, answer