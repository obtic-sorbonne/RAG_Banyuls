# app/utils/database.py
import sqlite3
import json
from datetime import datetime

def init_db():
    """Initialize the database with required tables"""
    conn = sqlite3.connect('rag_app.db')
    cursor = conn.cursor()

    # Create interactions table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS interactions (
            id TEXT PRIMARY KEY,
            query TEXT NOT NULL,
            context TEXT NOT NULL,
            response TEXT NOT NULL,
            retrieved_docs TEXT NOT NULL,
            metadata TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Create feedback table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            interaction_id TEXT NOT NULL,
            accuracy INTEGER,
            completeness INTEGER,
            relevance INTEGER,
            hallucination BOOLEAN,
            comments TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (interaction_id) REFERENCES interactions (id)
        )
    """)

    conn.commit()
    conn.close()

def save_interaction(interaction_id, query, context, response, retrieved_docs, metadata=None):
    """Save an interaction to the database"""
    conn = sqlite3.connect('rag_app.db')
    cursor = conn.cursor()

    retrieved_docs_json = json.dumps(retrieved_docs)
    metadata_json = json.dumps(metadata) if metadata else None

    cursor.execute("""
        INSERT INTO interactions (id, query, context, response, retrieved_docs, metadata)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (interaction_id, query, context, response, retrieved_docs_json, metadata_json))

    conn.commit()
    conn.close()

def save_feedback(interaction_id, accuracy, completeness, relevance, hallucination, comments=None):
    """Save feedback for an interaction"""
    conn = sqlite3.connect('rag_app.db')
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO feedback (interaction_id, accuracy, completeness, relevance, hallucination, comments)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (interaction_id, accuracy, completeness, relevance, int(hallucination), comments))

    feedback_id = cursor.lastrowid
    conn.commit()
    conn.close()

    return feedback_id

def get_interactions(limit=50):
    """Retrieve recent interactions from the database"""
    conn = sqlite3.connect('rag_app.db')
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute("""
        SELECT * FROM interactions 
        ORDER BY timestamp DESC 
        LIMIT ?
    """, (limit,))

    rows = cursor.fetchall()
    interactions = []

    for row in rows:
        interaction = dict(row)
        interaction['retrieved_docs'] = json.loads(interaction['retrieved_docs'])
        if interaction['metadata']:
            interaction['metadata'] = json.loads(interaction['metadata'])
        interactions.append(interaction)

    conn.close()
    return interactions

def get_feedback_stats():
    """Get feedback statistics"""
    conn = sqlite3.connect('rag_app.db')
    cursor = conn.cursor()

    cursor.execute("""
        SELECT 
            COUNT(*) as total_feedback,
            AVG(accuracy) as avg_accuracy,
            AVG(completeness) as avg_completeness,
            AVG(relevance) as avg_relevance,
            AVG(hallucination) as hallucination_rate
        FROM feedback
    """)

    result = cursor.fetchone()
    stats = {
        'total_feedback': result[0],
        'avg_accuracy': result[1],
        'avg_completeness': result[2],
        'avg_relevance': result[3],
        'hallucination_rate': result[4]
    }

    conn.close()
    return stats