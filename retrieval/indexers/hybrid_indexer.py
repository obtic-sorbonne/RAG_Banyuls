from .vector_indexer import VectorIndexer
from .keyword_indexer import KeywordIndexer
from .temporal_indexer import TemporalIndexer
from typing import List, Dict, Optional
from datetime import datetime
import numpy as np
import logging

# Initialize logger
logger = logging.getLogger(__name__)

class HybridIndexer:
    def __init__(self, config: Dict, data_dir: str = "./data/dump",
                 ):
        self.config = config

        self.vector_indexer = VectorIndexer(config=config)
        self.keyword_indexer = KeywordIndexer(config=config, data_dir=data_dir)
        self.temporal_indexer = TemporalIndexer(config=config, data_dir=data_dir)
    
    def search(self, query: str, query_embedding: np.ndarray,
            filters: Dict = None, temporal_range: Dict = None,
            top_k: int = 10) -> List[Dict]:
        """Hybrid search with filter-aware retrieval"""
        results = []
        
        # Vector search - only if we have embedding
        if query_embedding is not None and query_embedding.size > 0:
            try:
                vector_results = self.vector_indexer.search(
                    query=query,
                    query_embedding=query_embedding,
                    top_k=top_k,
                    filters=filters
                )
                results.extend(vector_results)
            except Exception as e:
                logger.error(f"Vector search failed: {e}")
        
        # Keyword search - only if query exists
        if query and query.strip():
            try:
                keyword_results = self.keyword_indexer.search(
                    query=query,
                    filters=filters,
                    top_k=top_k
                )
                results.extend(keyword_results)
            except Exception as e:
                logger.error(f"Keyword search failed: {e}")
        
        # Temporal search - only if date range provided
        if temporal_range.get("start") and temporal_range.get("end"):
            try:
                temporal_results = self.temporal_indexer.search(
                    # temporal_range["start"],
                    # temporal_range["end"],
                    top_k=top_k,
                    filters=temporal_range  # Pass filters to temporal indexer
                )
                results.extend(temporal_results)
            except Exception as e:
                logger.error(f"Temporal search failed: {e}")
        
        return self._deduplicate_results(results)
    
    def _deduplicate_results(self, results: List[Dict]) -> List[Dict]:
        """Remove duplicate results based on document ID"""
        seen = set()
        unique_results = []
        for res in results:
            if res["id"] not in seen:
                seen.add(res["id"])
                unique_results.append(res)
        return unique_results