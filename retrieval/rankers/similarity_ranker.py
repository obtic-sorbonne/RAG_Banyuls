from typing import List, Dict, Optional
import numpy as np
import logging
from sentence_transformers.util import cos_sim
from sentence_transformers import SentenceTransformer


# Initialize logger
logger = logging.getLogger(__name__)

class SimilarityRanker:
    def __init__(self, config: Dict, alpha: float = 0.7):
        self.alpha = alpha  # Weight for semantic similarity
        self.embedding_model = SentenceTransformer(config.get("embedding_model", "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"))
    
    def rank(self, results: List[Dict], query_embedding: np.ndarray, 
             embedding_model = None) -> List[Dict]:
        """Hybrid re-ranking with fallback handling"""
        if not results or query_embedding.size == 0:
            return results

        if embedding_model == None:
            embedding_model = self.embedding_model

        # Separate rerankable and non-rerankable results
        result_texts = [str(result) for result in results]
        embeddings = embedding_model.encode(result_texts)

        # update
        for result, embedding in zip(results, embeddings):
            result['embedding'] = embedding

        # rerankable = []
        # others = []
        # for r in results:
        #     if "embedding" in r and r["embedding"] is not None:
        #         rerankable.append(r)
        #     else:
        #         others.append(r)
        #
        # if not rerankable:
        #     return results
            
        try:
            # Calculate similarities in batch
            similarities = cos_sim([query_embedding], embeddings)[0].numpy().tolist()
            
            # Update scores
            for i, r in enumerate(results):
                semantic_score = similarities[i]
                
                if "score" in r:
                    # Combine existing score with semantic similarity
                    r["score"] = (1 - self.alpha) * r["score"] + self.alpha * semantic_score
                else:
                    r["score"] = semantic_score
                
                # Store semantic score for explainability
                r["metadata"]["semantic_score"] = semantic_score
        except Exception as e:
            logger.error(f"Similarity ranking failed: {e}")
            return results
        
        # Combine and sort
        ranked = sorted(results, key=lambda x: x["score"], reverse=True)
        return ranked