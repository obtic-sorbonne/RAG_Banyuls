import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict
from sentence_transformers import SentenceTransformer


class EmbeddingGenerator:
    def __init__(self, model_name: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2", dim: int = 768):
        self.model = SentenceTransformer(model_name)
        self.dim = dim  # Dimension for mpnet-base
    
    def generate(self, chunks: List[Dict]) -> List[Dict]:
        """Generate embeddings for text chunks"""
        texts = [chunk["content"] for chunk in chunks]
        embeddings = self.model.encode(texts, show_progress_bar=True)
        
        for i, chunk in enumerate(chunks):
            chunk["embedding"] = embeddings[i].tolist()
        
        return chunks
    
    def batch_generate(self, chunks: List[Dict], batch_size=32) -> List[Dict]:
        """Generate embeddings in batches"""
        embeddings = []
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i+batch_size]
            texts = [chunk["content"] for chunk in batch]
            batch_embeddings = self.model.encode(texts)
            embeddings.extend(batch_embeddings)
        
        for i, chunk in enumerate(chunks):
            chunk["embedding"] = embeddings[i].tolist()
        
        return chunks
