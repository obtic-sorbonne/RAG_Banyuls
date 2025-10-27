import re
from typing import List, Dict

class ChunkingStrategy:
    def chunk(self, content: str, metadata: dict) -> List[Dict]:
        raise NotImplementedError

class PageChunker(ChunkingStrategy):
    """Treat each page as a single chunk"""
    def chunk(self, content: str, metadata: dict) -> List[Dict]:
        return [{
            "content": content,
            "metadata": metadata,
            "chunk_type": "full_page"
        }]

class RecordChunker(ChunkingStrategy):
    """Split pages into individual observation records"""
    def __init__(self, max_length=500):
        self.max_length = max_length
    
    def chunk(self, content: str, metadata: dict) -> List[Dict]:
        # Split by record markers
        chunks = []
        current_chunk = []
        
        # Record separators: signatures, timestamps, section headers
        separators = [
            r"Signature:\s*\w+",
            r"\d{1,2}h\d{0,2}\s*[\-à]\s*\d{1,2}h\d{0,2}",
            r"Observation\s*\d+:",
            r"^\*{3,}\s*$"
        ]
        
        # Split content into lines and group records
        lines = content.split('\n')
        for line in lines:
            if any(re.search(sep, line) for sep in separators) and current_chunk:
                chunks.append('\n'.join(current_chunk))
                current_chunk = [line]
            else:
                current_chunk.append(line)
        
        if current_chunk:
            chunks.append('\n'.join(current_chunk))
        
        # Further split large chunks
        final_chunks = []
        for chunk in chunks:
            if len(chunk) > self.max_length:
                # Split by sentences
                sub_chunks = re.split(r'(?<=[.!?])\s+', chunk)
                current_sub = []
                for sub in sub_chunks:
                    if len(' '.join(current_sub + [sub])) > self.max_length and current_sub:
                        final_chunks.append(' '.join(current_sub))
                        current_sub = [sub]
                    else:
                        current_sub.append(sub)
                if current_sub:
                    final_chunks.append(' '.join(current_sub))
            else:
                final_chunks.append(chunk)
        
        return [{
            "content": chunk,
            "metadata": metadata,
            "chunk_type": "record"
        } for chunk in final_chunks]

class SemanticChunker(ChunkingStrategy):
    """AI-powered semantic chunking (placeholder for future implementation)"""
    def chunk(self, content: str, metadata: dict) -> List[Dict]:
        # Would use NLP to identify topic boundaries
        return [{
            "content": content,  # Temporary implementation
            "metadata": metadata,
            "chunk_type": "semantic"
        }]
