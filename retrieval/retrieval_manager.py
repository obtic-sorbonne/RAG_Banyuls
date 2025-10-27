from .query_processor import QueryProcessor
from .indexers import HybridIndexer, VectorIndexer, KeywordIndexer, TemporalIndexer
from .rankers import SimilarityRanker, TemporalRanker, ReciprocalRankFusion
from typing import List, Dict, Tuple
import numpy as np
import yaml
import logging

# Initialize logger
logger = logging.getLogger(__name__)

class RetrievalManager:
    def __init__(self):

        with open("config/retrieval_settings.yaml", "r") as f:
            self.config = yaml.safe_load(f)

        data_dir = self.config.get("data_dir", "./data/dump")
        # keyword_index_dir = self.config.get("keyword_dir", "./data/indexer")
        # vector_dir = self.config.get("vector_dir", "./data/indexer")

        self.query_processor = QueryProcessor(self.config.get("embedding_model", "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"))
        self.indexer = HybridIndexer(self.config.get("indexer"), data_dir)
        self.vector_indexer = VectorIndexer(self.config.get("indexer"))
        self.keyword_indexer = KeywordIndexer(self.config.get("indexer"),data_dir=data_dir)
        self.temporal_indexer = TemporalIndexer(self.config.get("indexer"), data_dir)
        # self.indexer = KeywordIndexer(self.config.get("indexer"),data_dir=data_dir)
        # self.indexer = VectorIndexer(self.config.get("indexer"))
        self.similarity_ranker = SimilarityRanker(self.config.get("ranker"))
        self.temporal_ranker = TemporalRanker()
        self.fusion_ranker = ReciprocalRankFusion()
        self.embedding_model = self.query_processor.embedding_model  # Reuse from processor
    
    def retrieve(self, query: str, top_k: int = 10, strategy : List[str]=None) -> List[Dict]:
        """End-to-end retrieval process with optimized ranking"""
        # Process query
        if strategy is None:
            strategy = ['vector']

        clean_query, query_embedding, filters = self.query_processor.process(query)
        logger.info(f"Processed query: {clean_query} with filters: {filters}")
        
        # Retrieve from multiple indices
        results = []
        if 'vector' in strategy:
            vector_results = self.vector_indexer.search(query=clean_query,
                                       query_embedding=query_embedding,
                                       filters=filters["filters"],
                                       top_k=top_k)
            results.extend(vector_results)
        if 'keyword' in strategy:
            keyword_results = self.keyword_indexer.search(query=clean_query,
                                       filters=filters["filters"],
                                       top_k=top_k)
            results.extend(keyword_results)
        if 'temporal' in strategy:
            temporal_results = self.temporal_indexer.search(query =query,
                                         filters=filters["filters"])
            results.extend(temporal_results)

        # self.indexer.search(
        #     query=clean_query,
        #     query_embedding=query_embedding,
        #     filters=filters["filters"],
        #     temporal_range=filters["temporal"],
        #     top_k=top_k * 5  # Retrieve more for better re-ranking
        # )
        
        # Early exit if no results
        if not results:
            return []
        
        # Multi-stage ranking pipeline
        results = self._apply_ranking_pipeline(
            results, 
            query_embedding, 
            filters["temporal"]
        )
        
        return results[:top_k]
    
    def _apply_ranking_pipeline(self, 
                              results: List[Dict], 
                              query_embedding: np.ndarray,
                              temporal_filters: Dict) -> List[Dict]:
        """Optimized ranking pipeline with parallelizable stages"""
        # Stage 1: Similarity ranking
        results = self.similarity_ranker.rank(
            results, query_embedding, self.embedding_model
        )
        
        # # Stage 2: Temporal ranking
        # results = self.temporal_ranker.rank(results, temporal_filters)
        
        # Stage 3: Fusion ranking
        result_types = {}
        for r in results:
            result_types.setdefault(r.get("type", "unknown"), []).append(r)
        
        if len(result_types) > 1:
            return self.fusion_ranker.fuse(list(result_types.values()))
        
        return results