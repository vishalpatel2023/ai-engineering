from app.ingest import ingest_documents
from app.vector_store import (
    save_vectors,
    search,
    count
)
from app.embeddings import get_embedding
from app.llm import generate_answer


def main():

    # 1. Check ChromaDB

    total = count()

    if total == 0:

        print("No documents found in ChromaDB.")
        print("Creating embeddings...\n")

        chunks = ingest_documents("data")

        save_vectors(chunks)

    else:

        print(
            f"Loaded {total} chunks from ChromaDB."
        )


    print("\n----------------------------")
    print("        DOCUMENT RAG")
    print("------------------------------")


    # 2. Question loop
    while True:

        question = input(
            "\nAsk a question (type 'exit' to quit): "
        )


        if question.lower() == "exit":

            print("\nGoodbye!")

            break

        # 3. Embed question

        print("\nCreating query embedding...")

        query_vector = get_embedding(question)

        # 4. Search ChromaDB

        # results = search(
        #     query_vector,
        #     top_k=2
        # )

        results = search(query_vector, top_k=3)

        results = [
            result
            for result in results
            if result["score"] < 0.8
        ]

        # 5. Display results

        print("\n------------------------------")
        print("RETRIEVED CHUNKS")
        print("------------------------------")

        for rank, result in enumerate(results, start=1):

            metadata = result["metadata"]

            print(f"\nRank {rank}")
            print(f"Source: {metadata['source']}")

            if metadata["page"] != -1:
                print(f"Page: {metadata['page']}")

            print(f"Distance: {result['score']:.4f}")
            print(f"Text: {result['text']}")

        # 6. Build context

        context_parts = []

        for result in results:

            metadata = result["metadata"]

            source = metadata["source"]

            if metadata["page"] != -1:
                source += f", page {metadata['page']}"

            context_parts.append(
                f"[Source: {source}]\n{result['text']}"
            )

        context = "\n\n".join(context_parts)

        # context = "\n\n".join(
        #     result["text"]
        #     for result in results
        # )

        # 7. Generate answer

        print("\n------------------------------")
        print("GENERATING ANSWER")
        print("------------------------------")

        answer = generate_answer(
            question,
            context
        )

        # 8. Display answer
        
        print("\n------------------------------")
        print("AI ANSWER")
        print("------------------------------")

        print(answer)


if __name__ == "__main__":
    main()