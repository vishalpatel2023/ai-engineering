import json

from app.config import EMBEDDINGS_FILE
from app.embeddings import get_embedding
from app.rag import answer_question
from app.vector_store import (
    load_embeddings,
    save_embeddings
)


DOCUMENTS_FILE = "data/documents.json"


def load_documents():

    with open(
        DOCUMENTS_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


def build_vector_store():

    documents = load_documents()

    print("Creating embeddings...")

    for document in documents:

        print(
            f"Embedding {document['id']}..."
        )

        document["embedding"] = get_embedding(
            document["text"]
        )

    save_embeddings(
        documents,
        EMBEDDINGS_FILE
    )

    print("\nEmbeddings saved successfully.")


def main():

    try:

        documents = load_embeddings(
            EMBEDDINGS_FILE
        )

        print(
            f"Loaded {len(documents)} documents."
        )

    except FileNotFoundError:

        print(
            "No saved embeddings found."
        )

        build_vector_store()

        documents = load_embeddings(
            EMBEDDINGS_FILE
        )

    print("\n==============================")
    print("        MINI RAG")
    print("==============================")

    while True:

        question = input(
            "\nAsk a question (type 'exit' to quit): "
        )

        if question.lower() == "exit":
            break

        results, answer = answer_question(
            question,
            documents
        )

        print("\n------------------------------")
        print("RETRIEVED DOCUMENTS")
        print("------------------------------")

        for i, result in enumerate(
            results,
            start=1
        ):

            print(
                f"\nRank {i}"
            )

            print(
                f"Score: {result['score']:.4f}"
            )

            print(
                f"Text: {result['text']}"
            )

        print("\n------------------------------")
        print("AI ANSWER")
        print("------------------------------")

        print(answer)


if __name__ == "__main__":
    main()