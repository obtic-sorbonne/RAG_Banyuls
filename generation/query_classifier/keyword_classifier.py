# query_classifier.py
import re
from typing import Dict, List, Tuple

class KeywordClassifier:
    def __init__(self):
        self.patterns = {
            'temperature': r'température|températures|chaud|froid|degrés',
            'crew': r'équipage|marins|matelots|personnel|à bord',
            'location': r'localité|lieu|endroit|région|zone|cap',
            'date': r'juillet|août|janvier|février|mars|avril|mai|juin|septembre|octobre|novembre|décembre|\d{4}',
            'measurement': r'dragage|profondeur|mesure|résultat',
            'voyage': r'sortie|voyage|expédition|mission',
        }

    def classify(self, query: str) -> Dict[str, float]:
        """Classify query and return confidence scores for each category"""
        scores = {}
        query_lower = query.lower()

        for category, pattern in self.patterns.items():
            matches = re.findall(pattern, query_lower)
            scores[category] = len(matches) / len(query_lower.split())

        return scores

    def get_retrieval_strategy(self, query: str) -> List[str]:
        """Determine appropriate retrieval strategies based on query type"""
        scores = self.classify(query)
        strategies = ['vector']

        # if scores.get('date', 0) > 0.1:
        #     strategies.append('temporal')
        if scores.get('sql', 0) > 0.1:
            strategies.append('sql')
        if any(score > 0.2 for score in scores.values()):
            strategies.append('keyword')

        return strategies