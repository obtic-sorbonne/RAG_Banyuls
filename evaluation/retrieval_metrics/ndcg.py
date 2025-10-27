import numpy as np

class NDCGCalculator:
    def calculate(self, retrieved_docs: list, relevance_grades: dict, k: int = 5) -> float:
        """Calculate Normalized Discounted Cumulative Gain at K"""
        # Get relevance scores for the top K documents
        dcg = 0
        for i, doc in enumerate(retrieved_docs[:k]):
            rel = relevance_grades.get(doc['id'], 0)
            rank = i + 1
            dcg += rel / np.log2(rank + 1)
        
        # Calculate ideal DCG
        ideal_relevance = sorted(relevance_grades.values(), reverse=True)[:k]
        idcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(ideal_relevance))
        
        return dcg / idcg if idcg > 0 else 0
