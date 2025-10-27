class UserSatisfactionTracker:
    def __init__(self, db_path="satisfaction.db"):
        self.conn = sqlite3.connect(db_path)
        self._create_schema()
    
    def _create_schema(self):
        cursor = self.conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS satisfaction (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                query TEXT NOT NULL,
                rating INTEGER CHECK(rating BETWEEN 1 AND 5),
                feedback TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self.conn.commit()
    
    def record_rating(self, session_id: str, query: str, rating: int, feedback: str = "") -> None:
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO satisfaction (session_id, query, rating, feedback)
            VALUES (?, ?, ?, ?)
        """, (session_id, query, rating, feedback))
        self.conn.commit()
    
    def calculate_satisfaction_score(self) -> float:
        cursor = self.conn.cursor()
        cursor.execute("SELECT AVG(rating) FROM satisfaction")
        result = cursor.fetchone()
        return result[0] or 0 if result else 0
    
    def analyze_feedback(self) -> dict:
        """Analyze feedback text for common themes"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT feedback FROM satisfaction WHERE feedback != ''")
        feedbacks = [row[0] for row in cursor.fetchall()]
        
        # Simple text analysis - would use NLP in production
        themes = {
            "accuracy": sum(1 for f in feedbacks if "inaccurate" in f.lower()),
            "completeness": sum(1 for f in feedbacks if "incomplete" in f.lower()),
            "relevance": sum(1 for f in feedbacks if "irrelevant" in f.lower()),
            "hallucination": sum(1 for f in feedbacks if "hallucinat" in f.lower()),
            "positive": sum(1 for f in feedbacks if "good" in f.lower() or "great" in f.lower())
        }
        
        total = len(feedbacks)
        if total > 0:
            for key in themes:
                themes[key] = themes[key] / total
        
        return themes
