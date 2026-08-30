import json
import os
import math


STORAGE_FILE = "storage/vectors.json"


def save_vectors(chunks):

    os.makedirs("storage", exist_ok=True)

    with open(STORAGE_FILE, "w", encoding="utf-8") as file:
        json.dump(chunks, file, ensure_ascii=False, indent=2)

    print(f"Saved {len(chunks)} chunks to {STORAGE_FILE}")


def load_vectors():

    if not os.path.exists(STORAGE_FILE):
        return None

    with open(STORAGE_FILE, "r", encoding="utf-8") as file:
        chunks = json.load(file)

    print(f"Loaded {len(chunks)} chunks from {STORAGE_FILE}")

    return chunks


def cosine_similarity(vec1, vec2):

    dot_product = 0
    magnitude1 = 0
    magnitude2 = 0

    for a, b in zip(vec1, vec2):

        dot_product += a * b
        magnitude1 += a * a
        magnitude2 += b * b

    magnitude1 = math.sqrt(magnitude1)
    magnitude2 = math.sqrt(magnitude2)

    if magnitude1 == 0 or magnitude2 == 0:
        return 0

    return dot_product / (magnitude1 * magnitude2)


def search(query_vector, chunks, top_k=2):

    results = []

    for chunk in chunks:

        score = cosine_similarity(
            query_vector,
            chunk["embedding"]
        )

        results.append({
            "chunk": chunk,
            "score": score
        })

    results.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    return results[:top_k]