class MRRCalculator:
    def calculate(self, retrieved_docs: list, relevant_docs: list) -> float:
        """Calculate Mean Reciprocal Rank"""
        for i, doc in enumerate(retrieved_docs):
            if doc['id'] in relevant_docs:
                return 1 / (i + 1)
        return 0
    
    def calculate_batch(self, queries: dict) -> float:
        """Calculate MRR for a batch of queries"""
        total = 0
        for query, results in queries.items():
            total += self.calculate(results['retrieved'], results['relevant'])
        return total / len(queries) if queries else 0
