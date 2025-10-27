import os
import json
import re
from tqdm import tqdm
from whoosh.index import open_dir, create_in
from whoosh.qparser import MultifieldParser, OrGroup
from whoosh import scoring
from whoosh.fields import SchemaClass, TEXT, ID, NUMERIC, KEYWORD, DATETIME
from whoosh.query import Term
from typing import List, Dict, Optional
import logging
import numpy as np

# Initialize logger
logger = logging.getLogger(__name__)

# class KeywordIndexSchema(SchemaClass):
#     """Structured schema for keyword indexing"""
#     id = ID(stored=True, unique=True)
#     content = TEXT(stored=True)
#     book = TEXT(stored=True)
#     years = KEYWORD(stored=True, sortable=True)
#     date = DATETIME(stored=True, sortable=True)
#     page = NUMERIC(stored=True)
#     # chunk_type = TEXT(stored=True)
#     entities = TEXT(stored=True)  # JSON string of entities

class KeywordIndexSchema(SchemaClass):
    """Structured schema for keyword indexing"""
    id = ID(stored=True, unique=True)
    book = ID(stored=True)
    page = NUMERIC(stored=True)
    content = TEXT(stored=True)
    entities = KEYWORD(stored=True, commas=True, lowercase=True)
    years = KEYWORD(stored=True, commas=True, lowercase=True)

class KeywordIndexer:
    def __init__(self, config: Dict, data_dir: str = "./data/dump"):
        try:
            self.schema = KeywordIndexSchema()
            self.config = config
            keyword_cache_dir = self.config.get("keyword_cache_dir", "./data/indexer")
            if not os.path.exists(keyword_cache_dir):
                os.makedirs(keyword_cache_dir)
                self.index = create_in(keyword_cache_dir, self.schema)
                self.build_index(data_dir)
            else:
                self.index = open_dir(keyword_cache_dir)
                
            logger.info(f"Loaded keyword index from {keyword_cache_dir}")
        except Exception as e:
            logger.critical(f"Keyword index initialization failed: {e}")
            raise

    def build_index(self, data_dir: str):
        """Build keyword index from processed JSON files"""
        writer = self.index.writer()

        for book_file in tqdm(os.listdir(data_dir), desc="Indexing books"):
            if not book_file.endswith(".json"):
                continue
            path = os.path.join(data_dir, book_file)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    book_data = json.load(f)

                book_name = book_data.get("book_name", os.path.splitext(book_file)[0])
                years = ",".join(book_data.get("metadata", {}).get("years", []))

                for page in book_data.get("pages", []):
                    page_number = int(page.get("page_number", 0))
                    content = page.get("cleaned_content", "")
                    entity_dates = ",".join(page.get("entities", {}).get("dates", []))

                    # Create unique id (book_page)
                    doc_id = f"{book_name}_{page_number:04d}"

                    writer.add_document(
                        id=doc_id,
                        book=book_name,
                        page=page_number,
                        content=content,
                        entities=entity_dates,
                        years=years
                    )
            except Exception as e:
                logger.error(f"Failed indexing {book_file}: {e}")

        writer.commit()
        logger.info("Index built successfully.")
    def search(self, query: str, filters: Dict = None,
               top_k: int = 10) -> List[Dict]:
        """Keyword BM25 search with optional filters"""
        if not query or not self._is_valid_query(query):
            return []

        try:
            with self.index.searcher(weighting=scoring.BM25F()) as searcher:
                safe_query = self._sanitize_query(query)
                if not safe_query.strip():
                    return []

                parser = MultifieldParser(
                    ["content", "book", "entities", "years"],
                    schema=self.index.schema,
                    group=OrGroup
                )
                parsed_query = parser.parse(safe_query)

                # Apply filters
                filtered_query = self._apply_filters(parsed_query, filters)

                results = searcher.search(filtered_query, limit=top_k)
                return self._format_results(results)
        except Exception as e:
            logger.error(f"Keyword search failed: {e}")
            return []
    
    def _is_valid_query(self, query: str) -> bool:
        """Check if query contains meaningful search terms"""
        # Should have at least 2 alphanumeric characters
        return len(re.sub(r'\W+', '', query)) >= 2
    
    # def _sanitize_query(self, query: str) -> str:
    #     """Remove problematic characters and keywords"""
    #     # Remove filter keywords that might cause errors
    #     for kw in ["AND", "OR", "NOT", "TO"]:
    #         query = re.sub(rf'\b{kw}\b', '', query, flags=re.IGNORECASE)
    #
    #     # Remove special characters that break Whoosh
    #     return re.sub(r'[^\w\sàâçéèêëîïôûùüÿñæœ]', ' ', query)
    def _sanitize_query(self, query: str) -> str:
        """Escape potentially problematic characters"""
        return re.sub(r"[^\w\s]", " ", query).strip()

    def _apply_filters(self, query, filters: Optional[Dict]):
        """Apply structured filters to query"""
        if not filters:
            return query
        from whoosh.query import And, Term
        clauses = [query]
        if "book" in filters:
            clauses.append(Term("book", filters["book"]))
        if "year" in filters:
            clauses.append(Term("years", str(filters["year"])))
        if "entity" in filters:
            clauses.append(Term("entities", filters["entity"]))
        return And(clauses)

    def _format_results(self, results) -> List[Dict]:
        """Standardize result format"""
        formatted = []
        for r in results:
            formatted.append({
                "id": r["id"],
                "content": r["content"],
                "metadata": {
                    "book": r.get("book"),
                    "page": r.get("page"),
                    "entities": r.get("entities"),
                    "years": r.get("years")
                },
                # "score": r.score,
                "type": "keyword"
            })
        return formatted