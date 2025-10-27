class CoverageCalculator:
    def calculate(self, retrieved_docs: list, corpus_size: int) -> float:
        """Calculate corpus coverage of retrieved documents"""
        unique_docs = {doc['id'] for doc in retrieved_docs}
        return len(unique_docs) / corpus_size if corpus_size > 0 else 0
    
    def temporal_coverage(self, retrieved_docs: list, date_range: tuple) -> float:
        """Calculate coverage of temporal range"""
        if not date_range:
            return 0
        
        min_date, max_date = date_range
        doc_dates = []
        
        for doc in retrieved_docs:
            if 'start_date' in doc['metadata']:
                doc_dates.append(doc['metadata']['start_date'])
        
        if not doc_dates:
            return 0
        
        min_doc_date = min(doc_dates)
        max_doc_date = max(doc_dates)
        
        covered_range = (max_doc_date - min_doc_date).days
        total_range = (max_date - min_date).days
        
        return covered_range / total_range if total_range > 0 else 0
