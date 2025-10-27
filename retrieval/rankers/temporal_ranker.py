from typing import List, Dict, Optional
import logging

# Initialize logger
logger = logging.getLogger(__name__)

class TemporalRanker:
    def __init__(self, beta: float = 0.4):
        self.beta = beta  # Weight for temporal relevance
    
    def rank(self, results: List[Dict], time_filters: Dict) -> List[Dict]:
        """Temporal re-ranking with fallback strategy"""
        if not results or not time_filters:
            return results
            
        try:
            start_ts = time_filters["start"].timestamp()
            end_ts = time_filters["end"].timestamp()
            query_duration = max(1, end_ts - start_ts)  # Prevent division by zero
        except Exception as e:
            logger.error(f"Invalid time filters: {time_filters} - {e}")
            return results
            
        for result in results:
            try:
                metadata = result.get("metadata", {})
                
                # Calculate temporal score (0 if missing temporal data)
                if "start_ts" in metadata and "end_ts" in metadata:
                    doc_start = metadata["start_ts"]
                    doc_end = metadata["end_ts"]
                    temporal_score = self._calculate_score(
                        doc_start, doc_end, start_ts, end_ts, query_duration
                    )
                else:
                    temporal_score = 0
                
                # Store for explainability
                result["metadata"]["temporal_score"] = temporal_score
                
                # Update overall score
                if "score" in result:
                    result["score"] = (1 - self.beta) * result["score"] + self.beta * temporal_score
                else:
                    result["score"] = temporal_score
                    
            except Exception as e:
                logger.warning(f"Temporal ranking failed for result {result.get('id')}: {e}")
                result.setdefault("score", 0)
        
        return sorted(results, key=lambda x: x.get("score", 0), reverse=True)
    
    def _calculate_score(self, 
                       doc_start: float, 
                       doc_end: float, 
                       query_start: float, 
                       query_end: float,
                       query_duration: float) -> float:
        """Calculate temporal relevance with multiple factors"""
        # Calculate coverage (how much of doc is covered by query)
        overlap_start = max(doc_start, query_start)
        overlap_end = min(doc_end, query_end)
        overlap_duration = max(0, overlap_end - overlap_start)
        coverage = overlap_duration / (doc_end - doc_start) if doc_end > doc_start else 0
        
        # Calculate containment (how much of query is covered by doc)
        containment = overlap_duration / query_duration
        
        # Calculate recency boost (more recent = higher score)
        recency = 1 - min(1, (query_end - doc_end) / (100 * 365 * 24 * 3600))  # 100-year range
        
        # Combine factors
        return 0.5 * coverage + 0.3 * containment + 0.2 * recency