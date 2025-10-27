from retrieval.retrieval_manager import RetrievalManager


import logging

def main():
    logging.basicConfig(level=logging.INFO)
    # Initialize retrieval system
    retriever = RetrievalManager()

    # Execute query
    results = retriever.retrieve(
        "Quelles étaient les températures de l'eau en juillet 1832 sur le navire Neptune?",
        top_k=5,
        strategy=['vector', 'keyword']
    )

    # print(results)
    # Display results
    for i, result in enumerate(results):
        print(f"\nResult {i+1}:")
        print(f"Source: {result['metadata']['book']} (Page {result['metadata']['page']})")
        print(f"Score: {result.get('score', 0):.3f}")
        print(f"Content: {result['content'][:200]}...")

if __name__ == "__main__":
    main()