class HumanEvaluationFramework:
    def __init__(self, db_path="human_eval.db"):
        self.conn = sqlite3.connect(db_path)
        self._create_schema()
    
    def _create_schema(self):
        cursor = self.conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS evaluations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query TEXT NOT NULL,
                response TEXT NOT NULL,
                context TEXT,
                relevance INTEGER CHECK(relevance BETWEEN 1 AND 5),
                fluency INTEGER CHECK(fluency BETWEEN 1 AND 5),
                informativeness INTEGER CHECK(informativeness BETWEEN 1 AND 5),
                accuracy INTEGER CHECK(accuracy BETWEEN 1 AND 5),
                evaluator_id TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self.conn.commit()
    
    def submit_evaluation(self, query: str, response: str, context: str,
                          relevance: int, fluency: int, informativeness: int,
                          accuracy: int, evaluator_id: str = "anonymous") -> None:
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO evaluations (
                query, response, context, relevance, 
                fluency, informativeness, accuracy, evaluator_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (query, response, context, relevance, fluency, informativeness, accuracy, evaluator_id))
        self.conn.commit()
    
    def calculate_aggregates(self, min_evals=10) -> dict:
        """Calculate average human evaluation metrics"""
        cursor = self.conn.cursor()
        cursor.execute(f"""
            SELECT 
                AVG(relevance) as avg_relevance,
                AVG(fluency) as avg_fluency,
                AVG(informativeness) as avg_informativeness,
                AVG(accuracy) as avg_accuracy
            FROM evaluations
            WHERE (
                SELECT COUNT(*) FROM evaluations
            ) >= ?
        """, (min_evals,))
        
        result = cursor.fetchone()
        return {
            "relevance": result[0] or 0,
            "fluency": result[1] or 0,
            "informativeness": result[2] or 0,
            "accuracy": result[3] or 0
        }
    
    def close(self):
        self.conn.close()
