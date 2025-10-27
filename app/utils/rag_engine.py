# app/utils/rag_engine.py
from generation.generation_manager import GenerationManager
from retrieval.retrieval_manager import RetrievalManager
from generation.query_classifier.keyword_classifier import KeywordClassifier
from datetime import datetime

class RAGEngine:
    def __init__(self):
        self.retriever = RetrievalManager()
        self.generator = GenerationManager()
        self.query_classifier = KeywordClassifier()

    def process_query(self, query, start_year=None, end_year=None, document_types=None):
        # Determine retrieval strategy
        retrieval_strategy = self.query_classifier.get_retrieval_strategy(query)

        # Prepare query metadata
        query_metadata = {}

        # Add temporal constraints if provided
        if start_year and end_year:
            query_metadata["temporal"] = {
                "start": datetime(start_year, 1, 1),
                "end": datetime(end_year, 12, 31)
            }

        # Retrieve documents
        results = self.retriever.retrieve(query, top_k=5, strategy=retrieval_strategy)

        # Prepare context
        context = "\n\n".join([r['content'] for r in results])

        # Generate response
        response = self.generator.generate_response(query, results, query_metadata)

        # Format retrieved documents for display
        retrieved_documents = []
        for r in results:
            doc = {
                'content': r['content'],
                'source': r['metadata'].get('book', 'Unknown'),
                'date': r['metadata'].get('primary_year', 'Unknown'),
                'page': r['metadata'].get('page', 'N/A'),
                'score': r.get('score', 0),
                'title': f"{r['metadata'].get('book', 'Document')} - Page {r['metadata'].get('page', 'N/A')}"
            }
            retrieved_documents.append(doc)

        return {
            'response': response['response'],
            'context': context,
            'retrieved_documents': retrieved_documents,
            'query_metadata': query_metadata,
            'retrieval_strategy': retrieval_strategy
        }