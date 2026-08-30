from app.loader import load_text_file
from app.chunker import chunk_text
from app.embeddings import get_embedding


def ingest_document(filepath):

    text = load_text_file(filepath)

    chunks = chunk_text(text)

    for chunk in chunks:

        print(f"Embedding {chunk['id']}...")

        chunk["embedding"] = get_embedding(chunk["text"])

    return chunks