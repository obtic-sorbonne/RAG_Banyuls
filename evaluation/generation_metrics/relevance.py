from sentence_transformers import CrossEncoder

class RelevanceEvaluator:
    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model = CrossEncoder(model_name)
    
    def calculate(self, response: str, query: str) -> float:
        """Calculate relevance of response to query"""
        scores = self.model.predict([(query, response)])
        return float(scores[0])
