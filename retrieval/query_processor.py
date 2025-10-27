import re
import numpy as np
from datetime import datetime
from dateutil import parser
from typing import Tuple, Dict, Any, List
import logging
from sentence_transformers import SentenceTransformer

# Initialize logger
logger = logging.getLogger(__name__)

class QueryProcessor:
    def __init__(self, embedding_model: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"):
        self.embedding_model = SentenceTransformer(embedding_model)
        self.date_patterns = [
            (r"\d{1,2}\s+[A-Za-z]+\s+\d{4}", "full_date"),  # 14 juillet 1832
            (r"\d{4}-\d{2}-\d{2}", "iso_date"),             # 1832-07-14
            (r"\d{1,2}/\d{1,2}/\d{4}", "numeric_date"),     # 14/07/1832
            (r"\d{4}", "year")                               # 1832
        ]
        self.entity_patterns = {
            # "year": r"\b(18\d{2}|19\d{2}|20\d{2})\b",
            "year": r"\b(?:18\d{2}|19\d{2}|20\d{2})\b",
            "book": r"(book|livre|volume)\s+['\"]?([\w\s]+)['\"]?",
            "page": r"(page|pg\.?)\s+(\d+)"
        }
    
    def process(self, query: str) -> Tuple[str, np.ndarray, Dict]:
        """Process query with enhanced entity extraction"""
        # Extract entities
        entities = self._extract_entities(query)
        logger.debug(f"Extracted entities: {entities}")
        
        # Extract temporal filters
        time_filters = self._extract_temporal_filters(query)
        
        # Clean query
        clean_query = self._clean_query(query, time_filters, entities)
        
        # Generate embedding
        query_embedding = self.embedding_model.encode([clean_query], 
                                                     show_progress_bar=False)[0]
        
        # Prepare filters
        filters = self._build_filters(entities)
        
        return clean_query, query_embedding, {
            "temporal": time_filters,
            "filters": filters
        }
    
    def _extract_temporal_filters(self, query: str) -> Dict:
        """Extract date ranges with pattern-specific parsing"""
        dates = []
        for pattern, ptype in self.date_patterns:
            for match in re.finditer(pattern, query):
                try:
                    date_str = match.group(0)
                    if ptype == "year":
                        dates.append(datetime(int(date_str), 1, 1))
                    else:
                        dates.append(parser.parse(date_str, dayfirst=("numeric_date" in ptype)))
                except Exception as e:
                    logger.warning(f"Failed to parse {date_str}: {e}")
                    continue
        
        return self._create_date_range(dates)
    
    def _create_date_range(self, dates: List[datetime]) -> Dict:
        """Create normalized date range from extracted dates"""
        if not dates:
            return {}
        
        start_date = min(dates)
        end_date = max(dates)
        
        # Handle single date case
        if len(dates) == 1:
            start_date = datetime(start_date.year, 1, 1)
            end_date = datetime(end_date.year, 12, 31)
        
        return {"start": start_date, "end": end_date}
    
    def _extract_entities(self, query: str) -> Dict[str, List]:
        """Enhanced entity extraction using configurable patterns"""
        entities = {}
        
        for entity_type, pattern in self.entity_patterns.items():
            matches = re.findall(pattern, query, re.IGNORECASE)
            if not matches:
                continue
                
            if entity_type == "year":
                # entities[entity_type] = [int(match) for match, *_ in matches]
                entities[entity_type] = [int(match) for match in matches]
            elif entity_type == "book":
                entities[entity_type] = [book.strip() for _, book in matches]
            elif entity_type == "page":
                entities[entity_type] = [int(page) for _, page in matches]

        # logger.info("Query Entities: {entities}")        
        # print("Query Entities: {entities}")
        return entities
    
    def _build_filters(self, entities: Dict) -> Dict:
        """Construct unified filter dictionary"""
        filters = {}
        
        if "year" in entities:
            filters["years"] = {"$contains": entities["year"]}
        if "book" in entities:
            filters["book_name"] = {"$in": entities["book"]}
        if "page" in entities:
            filters["page"] = {"$in": entities["page"]}
        
        return filters
    
    def _clean_query(self, query: str, 
                   temporal: Dict, 
                   entities: Dict) -> str:
        """Remove filter-related terms with context awareness"""
        # # Remove temporal patterns
        # for pattern, _ in self.date_patterns:
        #     query = re.sub(pattern, "", query)
        #
        # # Remove entity mentions
        # for entity_type in entities:
        #     if entity_type == "book":
        #         query = re.sub(rf"(book|livre|volume)\s+['\"]?{entities['book'][0]}['\"]?", "", query, flags=re.IGNORECASE)
        #     elif entity_type == "page":
        #         query = re.sub(rf"(page|pg\.?)\s+{entities['page'][0]}", "", query, flags=re.IGNORECASE)
        #
        # Final cleanup
        return re.sub(r"\s+", " ", query).strip()