import chromadb


CHROMA_PATH = "storage/chroma"
COLLECTION_NAME = "documents"


# Create persistent ChromaDB client
client = chromadb.PersistentClient(
    path=CHROMA_PATH
)


# Create or load collection
collection = client.get_or_create_collection(
    name=COLLECTION_NAME
)


def save_vectors(chunks):

    # Avoid duplicate documents
    if collection.count() > 0:
        collection.delete(
            where={}
        )

    ids = []
    documents = []
    embeddings = []
    metadatas = []

    for chunk in chunks:

        ids.append(chunk["id"])
        documents.append(chunk["text"])
        embeddings.append(chunk["embedding"])

        metadatas.append({
            "source": chunk["source"],
            "page": chunk["page"] if chunk["page"] else -1,
            "chunk_index": chunk["chunk_index"]
        })

    collection.add(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas
    )

    print(
        f"Saved {len(chunks)} chunks to ChromaDB."
    )


def search(query_vector, top_k=2):

    results = collection.query(
        query_embeddings=[query_vector],
        n_results=top_k
    )

    retrieved = []

    for i in range(len(results["ids"][0])):

        retrieved.append({
            "id": results["ids"][0][i],
            "text": results["documents"][0][i],
            "score": results["distances"][0][i],
            "metadata": results["metadatas"][0][i]
        })

    return retrieved


def count():

    return collection.count()