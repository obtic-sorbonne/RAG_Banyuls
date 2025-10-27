class PrecisionRecallCalculator:
    def calculate(self, retrieved_docs: list, relevant_docs: list, k: int = 5) -> dict:
        """Calculate precision and recall at K"""
        top_k = retrieved_docs[:k]
        relevant_in_k = [doc for doc in top_k if doc['id'] in relevant_docs]
        
        precision = len(relevant_in_k) / k if k > 0 else 0
        recall = len(relevant_in_k) / len(relevant_docs) if relevant_docs else 0
        
        return {
            "precision@k": precision,
            "recall@k": recall,
            "f1@k": self._f1_score(precision, recall)
        }
    
    def _f1_score(self, precision: float, recall: float) -> float:
        if precision + recall == 0:
            return 0
        return 2 * (precision * recall) / (precision + recall)
    
    def calculate_curve(self, retrieved_docs: list, relevant_docs: list, max_k: int = 10) -> dict:
        """Calculate precision and recall at each K"""
        results = {}
        for k in range(1, max_k + 1):
            metrics = self.calculate(retrieved_docs, relevant_docs, k)
            results[f"k={k}"] = metrics
        return results
