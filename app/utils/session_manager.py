import streamlit as st
import json
from datetime import datetime

class SessionManager:
    def __init__(self):
        self.session_file = "data/session_data.json"

    def save_session(self):
        """Save session data to file"""
        session_data = {
            "query_history": st.session_state.get("query_history", []),
            "feedback_data": st.session_state.get("feedback_data", []),
            "last_saved": datetime.now().isoformat()
        }

        with open(self.session_file, 'w') as f:
            json.dump(session_data, f, default=str)

    def load_session(self):
        """Load session data from file"""
        try:
            with open(self.session_file, 'r') as f:
                session_data = json.load(f)

                # Convert string dates back to datetime objects
                for item in session_data.get("query_history", []):
                    if "timestamp" in item and isinstance(item["timestamp"], str):
                        item["timestamp"] = datetime.fromisoformat(item["timestamp"])

                st.session_state.query_history = session_data.get("query_history", [])
                st.session_state.feedback_data = session_data.get("feedback_data", [])

        except FileNotFoundError:
            # Initialize empty session
            st.session_state.query_history = []
            st.session_state.feedback_data = []

    def export_data(self, format_type="json"):
        """Export session data in specified format"""
        session_data = {
            "query_history": st.session_state.get("query_history", []),
            "feedback_data": st.session_state.get("feedback_data", []),
            "export_date": datetime.now().isoformat()
        }

        if format_type == "json":
            return json.dumps(session_data, indent=2, default=str)
        elif format_type == "csv":
            # Convert to CSV format
            import pandas as pd
            feedback_df = pd.DataFrame(session_data["feedback_data"])
            history_df = pd.DataFrame(session_data["query_history"])
            return feedback_df.to_csv(index=False), history_df.to_csv(index=False)

        return None

    def clear_session(self):
        """Clear all session data"""
        st.session_state.query_history = []
        st.session_state.feedback_data = []
        self.save_session()