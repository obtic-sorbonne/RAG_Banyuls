import json
import os
import re
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from intervaltree import Interval, IntervalTree
import logging
import numpy as np

# Precompile regex patterns for efficiency
DATE_PATTERNS = [
    (re.compile(r"\d{1,2}\s+[A-Za-z]+\s+\d{4}"), "full_date"),  # 14 juillet 1832
    (re.compile(r"\d{4}-\d{2}-\d{2}"), "iso_date"),             # 1832-07-14
    (re.compile(r"\d{1,2}/\d{1,2}/\d{4}"), "numeric_date")      # 14/07/1832
]
YEAR_PATTERN = re.compile(r"\b(18\d{2}|19\d{2}|20\d{2})\b")

# Initialize logger
logger = logging.getLogger(__name__)

class TemporalIndexer:
    def __init__(self, config: Dict, data_dir: str):
        self.tree = IntervalTree()
        self.build_index(data_dir)
        logger.info(f"Built temporal index with {len(self.tree)} intervals")
    
    def build_index(self, data_dir: str):
        """Efficiently build temporal index with chunk validation"""
        if not os.path.exists(data_dir):
            logger.error(f"Data directory not found: {data_dir}")
            return
            
        for book_file in os.listdir(data_dir):
            if not book_file.endswith(".json"):
                continue
                
            try:
                with open(os.path.join(data_dir, book_file), 'r') as f:
                    book_data = json.load(f)
                
                book_name = book_data.get("book_name", "unknown")
                book_year = self._parse_book_year(book_data)
                
                for page in book_data.get("pages", []):
                    for chunk in page.get("chunks", []):
                        self._index_chunk(chunk, book_name, book_year, page.get("page_number"))
            except Exception as e:
                logger.error(f"Error processing {book_file}: {e}")
    
    def _parse_book_year(self, book_data: Dict) -> int:
        """Extract and validate book year from metadata"""
        year_str = book_data.get("metadata", {}).get("year", "")
        if not year_str:
            return 0
            
        try:
            year_match = YEAR_PATTERN.search(year_str)
            return int(year_match.group(0)) if year_match else 0
        except:
            return 0
    
    def _index_chunk(self, chunk: Dict, book_name: str, book_year: int, page_number: int):
        """Index a single chunk with date extraction"""
        if not chunk.get("content"):
            return
            
        dates = self._extract_dates(chunk["content"])
        if not dates:
            return  # Skip chunks without dates
            
        start_date = min(dates)
        end_date = max(dates)
        
        # Create interval data
        interval_data = {
            "id": chunk["id"],
            "content": chunk["content"],
            "metadata": {
                "book": book_name,
                "year": book_year,
                "page": page_number,
                "start_ts": start_date.timestamp(),
                "end_ts": end_date.timestamp()
            }
        }
        
        # Add to interval tree
        self.tree.addi(
            start_date.timestamp(), 
            end_date.timestamp(), 
            interval_data
        )
    
    def _extract_dates(self, text: str) -> List[datetime]:
        """Efficient date extraction with pattern-specific parsing"""
        dates = []
        year_matches = YEAR_PATTERN.findall(text)
        
        # Add standalone years
        for year_str in year_matches:
            try:
                dates.append(datetime(int(year_str), 1, 1))
            except:
                continue
        
        # Parse full date formats
        for pattern, ptype in DATE_PATTERNS:
            for match in pattern.finditer(text):
                try:
                    date_str = match.group(0)
                    if ptype == "full_date":
                        dates.append(self._parse_french_date(date_str))
                    elif ptype == "iso_date":
                        dates.append(datetime.fromisoformat(date_str))
                    elif ptype == "numeric_date":
                        day, month, year = date_str.split('/')
                        dates.append(datetime(int(year), int(month), int(day)))
                except Exception as e:
                    logger.debug(f"Failed to parse {date_str}: {e}")
                    continue
                    
        return dates
    
    def _parse_french_date(self, date_str: str) -> datetime:
        """Parse French date format (e.g., '14 juillet 1832')"""
        month_map = {
            'janvier': 1, 'février': 2, 'mars': 3, 'avril': 4,
            'mai': 5, 'juin': 6, 'juillet': 7, 'août': 8,
            'septembre': 9, 'octobre': 10, 'novembre': 11, 'décembre': 12
        }
        day, month_name, year = date_str.split()
        return datetime(int(year), month_map[month_name.lower()], int(day))
    
    def search(self, query: str = None, query_embedding: np.ndarray = None,
                filters: Dict = None, temporal_range: Dict = None,
                top_k: int = 10) -> List[Dict]:
        """Temporal search with filter support and scoring"""
        start_date=filters["temporal"].get("start")
        end_date=filters["temporal"].get("end")
        
        if not self.tree or not (start_date & end_date):
            return []
        
        

        start_ts = start_date.timestamp()
        end_ts = end_date.timestamp()
        
        # Find overlapping intervals
        results = []
        for interval in self.tree.overlap(start_ts, end_ts):
            if self._matches_filters(interval.data, filters):
                results.append({
                    "id": interval.data["id"],
                    "content": interval.data["content"],
                    "metadata": interval.data["metadata"],
                    "temporal_score": self._calculate_temporal_score(
                        interval.begin, interval.end, start_ts, end_ts
                    ),
                    "type": "temporal"
                })
        
        # Sort and limit results
        results.sort(key=lambda x: x["temporal_score"], reverse=True)
        return results[:top_k]
    
    def _matches_filters(self, data: Dict, filters: Dict) -> bool:
        """Check if document matches given filters"""
        if not filters:
            return True
            
        meta = data["metadata"]
        for field, condition in filters.items():
            if field not in meta:
                return False
                
            if "$in" in condition:
                if meta[field] not in condition["$in"]:
                    return False
                    
        return True
    
    def _calculate_temporal_score(self, 
                                doc_start: float, 
                                doc_end: float, 
                                query_start: float, 
                                query_end: float) -> float:
        """Calculate normalized temporal relevance score"""
        # Calculate overlap
        overlap_start = max(doc_start, query_start)
        overlap_end = min(doc_end, query_end)
        overlap = max(0, overlap_end - overlap_start)
        
        # Calculate coverage ratio
        doc_duration = doc_end - doc_start
        if doc_duration <= 0:
            return 0
            
        coverage = overlap / doc_duration
        
        # Calculate position score (how centered is the query in the doc)
        doc_center = (doc_start + doc_end) / 2
        query_center = (query_start + query_end) / 2
        position_score = 1 - min(1, abs(doc_center - query_center) / doc_duration)
        
        return 0.7 * coverage + 0.3 * position_score