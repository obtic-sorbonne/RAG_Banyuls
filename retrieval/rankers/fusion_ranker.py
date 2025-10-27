from typing import List, Dict, Any
import numpy as np
import logging

# Initialize logger
logger = logging.getLogger(__name__)

class ReciprocalRankFusion:
    def __init__(self, k: int = 60, weights: Dict[str, float] = None):
        self.k = k
        self.weights = weights or {"vector": 1.0, "keyword": 0.9, "temporal": 0.8}
    
    def fuse(self, results_list: List[List[Dict]]) -> List[Dict]:
        """Weighted RRF with score normalization"""
        if not results_list:
            return []
            
        fused_scores = {}
        total_ranks = len(results_list)
        
        for rank_list, results in enumerate(results_list):
            result_type = results[0]["type"] if results else "unknown"
            weight = self.weights.get(result_type, 0.5)
            
            for rank, doc in enumerate(results):
                doc_id = doc["id"]
                if doc_id not in fused_scores:
                    fused_scores[doc_id] = {
                        "score": 0.0,
                        "doc": doc,
                        "contributions": []
                    }
                
                # Calculate weighted contribution
                contribution = weight / (self.k + rank + 1)
                fused_scores[doc_id]["score"] += contribution
                fused_scores[doc_id]["contributions"].append((
                    result_type, rank+1, contribution
                ))
        
        # Apply normalization
        max_score = max(item["score"] for item in fused_scores.values())
        if max_score > 0:
            for item in fused_scores.values():
                item["score"] /= max_score
                # Add debug info to metadata
                item["doc"]["metadata"]["fusion_info"] = {
                    "score": item["score"],
                    "contributions": item["contributions"]
                }
        
        # Sort by fused score
        sorted_docs = sorted(
            fused_scores.values(), 
            key=lambda x: x["score"], 
            reverse=True
        )
        
        return [item["doc"] for item in sorted_docs]