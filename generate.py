from typing import List

from generation.generation_manager import GenerationManager
from generation.query_classifier.keyword_classifier import KeywordClassifier
from retrieval.retrieval_manager import RetrievalManager

from datetime import datetime

# Initialize components
retriever = RetrievalManager()
generator = GenerationManager()
query_classifier = KeywordClassifier()

# Process query
# query = "Quelles étaient les températures de l'eau en juillet 1853?"
query = "Quelles étaient les conditions à Bord en juillet 1853?"

retrieval_strategy = query_classifier.get_retrieval_strategy(query)

query_type = query_classifier.classify(query)

results = retriever.retrieve(query, top_k=5, strategy=retrieval_strategy)

print("### Retrived contents:")
print(results)

# Prepare context
context = "\n\n".join([r['content'] for r in results])
query_metadata = {
    # "temporal": {"start": datetime(1832, 7, 1), "end": datetime(1832, 7, 31)},
    # "entities": {"measurement": "température"}
}

# Generate response
response = generator.generate_response(query, results, query_metadata)

print("### Response:")
print(response['response'])

# Example feedback
# generator.process_feedback(
#     feedback_id=response['feedback_id'],
#     accuracy=4,
#     completeness=5,
#     relevance=5,
#     hallucination=False,
#     comments="Good use of context data"
# )
