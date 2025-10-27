import sqlite3
from contextlib import contextmanager
import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import uuid

class FeedbackManager:
    def __init__(self, db_path="./data/feedback/feedback.db"):
        self.db_path = db_path
        self.init_db()

    @contextmanager
    def get_connection(self):
        """Context manager for database connections"""
        conn = sqlite3.connect(
            self.db_path,
            check_same_thread=False,
            isolation_level=None
        )
        conn.row_factory = sqlite3.Row  # Enable row factory for easier access
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            yield conn
        finally:
            conn.close()

    def init_db(self):
        with self.get_connection() as conn:
            cursor = conn.cursor()

            # Create records table to store generation outputs
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS generation_records (
                    id TEXT PRIMARY KEY,
                    query TEXT NOT NULL,
                    response TEXT NOT NULL,
                    augmented_context TEXT,
                    prompt TEXT,
                    query_type TEXT,
                    retrieval_strategy TEXT,
                    used_sources TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Create feedback table for human evaluations
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS human_feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    record_id TEXT NOT NULL,
                    accuracy INTEGER CHECK(accuracy BETWEEN 1 AND 5),
                    completeness INTEGER CHECK(completeness BETWEEN 1 AND 5),
                    relevance INTEGER CHECK(relevance BETWEEN 1 AND 5),
                    hallucination BOOLEAN,
                    comments TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (record_id) REFERENCES generation_records (id)
                )
            """)

            # Create evaluation table for LLM judge evaluations
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS llm_evaluations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    record_id TEXT NOT NULL,
                    evaluation_type TEXT NOT NULL,
                    score REAL,
                    details TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (record_id) REFERENCES generation_records (id)
                )
            """)

            conn.commit()

    def record_record(self, query: str, response: str,
                      augmented_context: str = None, prompt: str = None,
                      query_type: str = None, retrieval_strategy: str = None,
                      used_sources: List[Dict] = None) -> str:
        """Record a generation record for later evaluation"""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            record_id = str(uuid.uuid4())
            # Filter out none serializable embedding
            used_sources_json = json.dumps([{k: v for k, v in item.items() if k != 'embedding'} for item in used_sources])

            cursor.execute("""
                INSERT INTO generation_records 
                (id, query, response, augmented_context, prompt, query_type, retrieval_strategy, used_sources)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (record_id, query, response, augmented_context, prompt,
                  query_type, retrieval_strategy, used_sources_json))

            conn.commit()
            return record_id

    def record_human_feedback(self, record_id: str, accuracy: int,
                              completeness: int, relevance: int,
                              hallucination: bool, comments: str = "") -> None:
        """Record human feedback for a generation record"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO human_feedback 
                (record_id, accuracy, completeness, relevance, hallucination, comments)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (record_id, accuracy, completeness, relevance, int(hallucination), comments))
            conn.commit()

    def record_llm_evaluation(self, record_id: str, evaluation_type: str,
                              score: float, details: Dict) -> None:
        """Record LLM judge evaluation for a generation record"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            details_json = json.dumps(details)
            cursor.execute("""
                INSERT INTO llm_evaluations 
                (record_id, evaluation_type, score, details)
                VALUES (?, ?, ?, ?)
            """, (record_id, evaluation_type, score, details_json))
            conn.commit()

    def get_record(self, record_id: str) -> Optional[Dict]:
        """Retrieve a generation record by ID"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM generation_records WHERE id = ?
            """, (record_id,))

            row = cursor.fetchone()
            if not row:
                return None

            record = dict(row)

            # Parse JSON fields
            if record.get('used_sources'):
                record['used_sources'] = json.loads(record['used_sources'])

            # Get feedback if available
            feedback = self.get_feedback_for_record(record_id)
            if feedback:
                record['feedback'] = feedback

            return record

    def get_feedback_for_record(self, record_id: str) -> Optional[Dict]:
        """Get human feedback for a specific record"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM human_feedback 
                WHERE record_id = ?
                ORDER BY timestamp DESC
                LIMIT 1
            """, (record_id,))

            row = cursor.fetchone()
            if row:
                return dict(row)
            return None

    def get_records_with_feedback(self, limit: int = 50) -> List[Dict]:
        """Get records along with their feedback (if available)"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT 
                    gr.*,
                    hf.accuracy, hf.completeness, hf.relevance, 
                    hf.hallucination, hf.comments as feedback_comments,
                    hf.timestamp as feedback_timestamp
                FROM generation_records gr
                LEFT JOIN human_feedback hf ON gr.id = hf.record_id
                ORDER BY gr.timestamp DESC
                LIMIT ?
            """, (limit,))

            rows = cursor.fetchall()
            records = []

            for row in rows:
                record = dict(row)

                # Parse JSON fields
                if record.get('used_sources'):
                    record['used_sources'] = json.loads(record['used_sources'])

                # Check if feedback exists
                has_feedback = any([
                    record.get('accuracy') is not None,
                    record.get('completeness') is not None,
                    record.get('relevance') is not None
                ])

                if has_feedback:
                    record['feedback'] = {
                        'accuracy': record.pop('accuracy', None),
                        'completeness': record.pop('completeness', None),
                        'relevance': record.pop('relevance', None),
                        'hallucination': record.pop('hallucination', None),
                        'comments': record.pop('feedback_comments', None),
                        'timestamp': record.pop('feedback_timestamp', None)
                    }

                records.append(record)

            return records

    def get_quality_metrics(self, last_n=100) -> Dict:
        """Calculate average quality metrics from feedback"""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            # Human feedback metrics
            cursor.execute(f"""
                SELECT 
                    AVG(accuracy) as avg_accuracy,
                    AVG(completeness) as avg_completeness,
                    AVG(relevance) as avg_relevance,
                    AVG(hallucination) as hallucination_rate,
                    COUNT(*) as feedback_count
                FROM human_feedback
                WHERE id IN (
                    SELECT id FROM human_feedback ORDER BY timestamp DESC LIMIT ?
                )
            """, (last_n,))

            human_result = cursor.fetchone()
            human_metrics = {
                "accuracy": human_result[0] or 0,
                "completeness": human_result[1] or 0,
                "relevance": human_result[2] or 0,
                "hallucination_rate": human_result[3] or 0,
                "feedback_count": human_result[4] or 0
            }

            # LLM evaluation metrics
            cursor.execute(f"""
                SELECT evaluation_type, AVG(score) as avg_score, COUNT(*) as eval_count
                FROM llm_evaluations 
                WHERE id IN (
                    SELECT id FROM llm_evaluations ORDER BY timestamp DESC LIMIT ?
                )
                GROUP BY evaluation_type
            """, (last_n,))

            llm_results = cursor.fetchall()
            llm_metrics = {}
            for result in llm_results:
                llm_metrics[result[0]] = {
                    "avg_score": result[1],
                    "eval_count": result[2]
                }

            return {
                "human_metrics": human_metrics,
                "llm_metrics": llm_metrics
            }

    def get_feedback_trends(self, days: int = 30) -> List[Dict]:
        """Get feedback trends over time"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT 
                    DATE(timestamp) as date,
                    AVG(accuracy) as avg_accuracy,
                    AVG(completeness) as avg_completeness,
                    AVG(relevance) as avg_relevance,
                    AVG(hallucination) as hallucination_rate,
                    COUNT(*) as feedback_count
                FROM human_feedback
                WHERE timestamp >= DATE('now', ?)
                GROUP BY DATE(timestamp)
                ORDER BY date DESC
            """, (f"-{days} days",))

            rows = cursor.fetchall()
            return [dict(row) for row in rows]

    def get_recent_records(self, limit=10) -> List[Dict]:
        """Get recent generation records for evaluation"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM generation_records 
                ORDER BY timestamp DESC LIMIT ?
            """, (limit,))

            rows = cursor.fetchall()
            records = []

            for row in rows:
                record = dict(row)
                if record.get('used_sources'):
                    record['used_sources'] = json.loads(record['used_sources'])

                # Get feedback if available
                feedback = self.get_feedback_for_record(record['id'])
                if feedback:
                    record['feedback'] = feedback

                records.append(record)

            return records