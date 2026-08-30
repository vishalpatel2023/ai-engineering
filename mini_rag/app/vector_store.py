import json
import math
from pathlib import Path


def cosine_similarity(
    vec1: list[float],
    vec2: list[float]
) -> float:
    """
    Calculate cosine similarity between two vectors.
    """

    dot_product = sum(
        a * b
        for a, b in zip(vec1, vec2)
    )

    magnitude1 = math.sqrt(
        sum(a * a for a in vec1)
    )

    magnitude2 = math.sqrt(
        sum(b * b for b in vec2)
    )

    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0

    return dot_product / (
        magnitude1 * magnitude2
    )


def save_embeddings(
    documents: list[dict],
    filepath: str
):
    """
    Save documents and their embeddings to disk.
    """

    path = Path(filepath)
    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(path, "w", encoding="utf-8") as file:
        json.dump(
            documents,
            file,
            indent=2
        )


def load_embeddings(filepath: str) -> list[dict]:
    """
    Load documents and embeddings from disk.
    """

    with open(
        filepath,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


def search(
    query_vector: list[float],
    documents: list[dict],
    top_k: int = 3
) -> list[dict]:
    """
    Brute-force vector similarity search.

    Compares the query against every stored vector.
    """

    results = []

    for document in documents:

        score = cosine_similarity(
            query_vector,
            document["embedding"]
        )

        results.append({
            "id": document["id"],
            "text": document["text"],
            "score": score
        })

    results.sort(
        key=lambda item: item["score"],
        reverse=True
    )

    return results[:top_k]