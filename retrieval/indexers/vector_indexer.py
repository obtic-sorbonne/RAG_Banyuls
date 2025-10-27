import chromadb
from chromadb.config import Settings
from typing import List, Dict, Optional
import numpy as np
import logging

# Initialize logger
logger = logging.getLogger(__name__)

class VectorIndexer:
    def __init__(self, config: Dict):
        try:
            if config.get("chromadb_type", "local") == "http":
                self.client = chromadb.HttpClient(
                    host=config.get("chromadb_host",'localhost'),
                    port=config.get("chromadb_port" ,8000),
                    # settings=Settings(chroma_db_impl="duckdb+parquet")
                )
            elif config.get("chromadb_type", "local") == "local":
                chroma_persist_dir = config.get("chromadb_persist_dir", "./data/vectordb/chromadb")
                self.client = chromadb.PersistentClient(
                    path=chroma_persist_dir,
                    settings=Settings(
                        anonymized_telemetry=False,
                        allow_reset=True
                    )
                )

            self.collection = self.client.get_collection(config.get("chroma_db_collection", "ocean_observations"))
            logger.info("Connected to ChromaDB vector store")
        except Exception as e:
            logger.critical(f"ChromaDB connection failed: {e}")
            raise
    
    def search(self, query: str = None, query_embedding: np.ndarray = None,
                filters: Dict = None, temporal_range: Dict = None,
                top_k: int = 10) -> List[Dict]:
        """Robust vector search with filter validation"""
        if not filters:
            filters = {}
        
        try:
            # Convert numpy array to list
            embedding_list = query_embedding.tolist()
            
            # Handle ChromaDB filter format
            chroma_filters = {}
            for k, v in filters.items():
                if "$cintains" in v:
                    chroma_filters[k] = {"$cintains": v["$in"]}

            logger.info(f"Chroma filters: {chroma_filters}")

            results = self.collection.query(
                query_embeddings=[embedding_list],
                n_results=top_k,
                where=chroma_filters if chroma_filters else None
            )

            logger.info(f"VectorBase retrieval resaults: {results}")

            return self._format_results(results)
        except Exception as e:
            logger.error(f"Vector search error: {e}")
            return []
    
    def _format_results(self, results) -> List[Dict]:
        """Standardize result format"""
        formatted = []
        for i in range(len(results["ids"][0])):
            formatted.append({
                "id": results["ids"][0][i],
                "content": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "score": 1 - results["distances"][0][i],  # Convert distance to similarity
                "type": "vector"
            })
        return formatted