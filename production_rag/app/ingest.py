from app.loader import load_documents
from app.chunker import chunk_text
from app.embeddings import get_embedding


def ingest_documents(folder_path):

    all_chunks = []

    documents = load_documents(folder_path)

    chunk_id = 0

    for document in documents:

        chunks = chunk_text(document["text"])

        for index, chunk in enumerate(chunks):

            # chunk can be either a string or a dictionary
            if isinstance(chunk, dict):
                text = chunk["text"]
            else:
                text = chunk

            print(
                f"Embedding {document['source']} "
                f"chunk {index}..."
            )

            embedding = get_embedding(text)

            all_chunks.append({
                "id": f"chunk_{chunk_id}",
                "text": text,
                "embedding": embedding,
                "source": document["source"],
                "page": document["page"],
                "chunk_index": index
            })

            chunk_id += 1

    return all_chunks