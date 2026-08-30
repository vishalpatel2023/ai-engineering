from app.ingest import ingest_document
from app.vector_store import (
    save_vectors,
    load_vectors,
    search
)
from app.embeddings import get_embedding
from app.llm import generate_answer


def main():

    chunks = load_vectors()

    if chunks is None:

        print("No saved vectors found.")
        print("Creating embeddings...\n")

        chunks = ingest_document("data/notes.txt")

        save_vectors(chunks)

    else:

        print("Using existing embeddings.")


    print("\n==============================")
    print("        DOCUMENT RAG")
    print("==============================")

    while True:

        question = input(
            "\nAsk a question (type 'exit' to quit): "
        )

        if question.lower() == "exit":
            print("\nGoodbye!")
            break

        print("\nCreating query embedding...")

        query_vector = get_embedding(question)

        results = search(
            query_vector,
            chunks,
            top_k=2
        )

        print("\n------------------------------")
        print("RETRIEVED CHUNKS")
        print("------------------------------")


        for rank, result in enumerate(results, start=1):

            chunk = result["chunk"]
            score = result["score"]

            print(f"\nRank {rank}")
            print(f"Score: {score:.4f}")
            print(f"ID: {chunk['id']}")
            print(f"Text: {chunk['text']}")

        context = "\n\n".join(
            result["chunk"]["text"]
            for result in results
        )

        print("\n------------------------------")
        print("GENERATING ANSWER")
        print("------------------------------")

        answer = generate_answer(
            question,
            context
        )

        print("\n------------------------------")
        print("AI ANSWER")
        print("------------------------------")

        print(answer)


if __name__ == "__main__":
    main()