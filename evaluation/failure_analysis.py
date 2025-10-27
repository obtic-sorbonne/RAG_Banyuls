class FailureAnalyzer:
    def __init__(self, eval_db_path="evaluation.db"):
        self.conn = sqlite3.connect(eval_db_path)
    
    def analyze_retrieval_failures(self, min_failures=10):
        """Identify patterns in retrieval failures"""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT query, context, response, feedback
            FROM evaluations
            WHERE relevance < 3 OR accuracy < 3
        """)
        failures = cursor.fetchall()
        
        if len(failures) < min_failures:
            return {"status": "insufficient_data", "count": len(failures)}
        
        # Analyze common characteristics
        patterns = {
            "temporal_mismatch": sum(1 for f in failures if "date" in f[0] and "date" not in f[1]),
            "entity_missing": sum(1 for f in failures if any(ent in f[0] for ent in ["ship", "vessel"]) and "ship" not in f[1]),
            "measurement_absence": sum(1 for f in failures if "temperature" in f[0] and "°" not in f[1]),
            "long_queries": sum(1 for f in failures if len(f[0].split()) > 10)
        }
        
        total = len(failures)
        for key in patterns:
            patterns[key] = patterns[key] / total
        
        return {
            "total_failures": total,
            "patterns": patterns,
            "sample_failures": failures[:5]
        }
    
    def analyze_generation_hallucinations(self, min_cases=5):
        """Analyze hallucination patterns in responses"""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT query, response, context
            FROM evaluations
            WHERE accuracy < 3
        """)
        hallucinations = cursor.fetchall()
        
        if len(hallucinations) < min_cases:
            return {"status": "insufficient_data", "count": len(hallucinations)}
        
        # Classify hallucination types
        hallucination_types = {
            "invented_measurements": 0,
            "incorrect_dates": 0,
            "nonexistent_entities": 0,
            "contradictory_info": 0
        }
        
        for _, response, context in hallucinations:
            if re.search(r"\d+\.\d+°[CF]", response) and not re.search(r"\d+\.\d+°[CF]", context):
                hallucination_types["invented_measurements"] += 1
            if re.search(r"\d{4}-\d{2}-\d{2}", response) and not re.search(r"\d{4}-\d{2}-\d{2}", context):
                hallucination_types["incorrect_dates"] += 1
            if re.search(r"Navire \w+", response) and not re.search(r"Navire \w+", context):
                hallucination_types["nonexistent_entities"] += 1
        
        total = len(hallucinations)
        for key in hallucination_types:
            hallucination_types[key] = hallucination_types[key] / total
        
        return {
            "total_hallucinations": total,
            "hallucination_types": hallucination_types,
            "sample_cases": hallucinations[:3]
        }
