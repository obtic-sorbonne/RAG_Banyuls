import streamlit as st
from datetime import datetime

class HistoryViewer:
    def __init__(self):
        if 'rag_engine' not in st.session_state:
            from generation.generation_manager import GenerationManager
            st.session_state.rag_engine = GenerationManager()

    def render(self):
        st.header("Query History")

        # Get recent records with feedback
        records = st.session_state.rag_engine.feedback.get_records_with_feedback(limit=50)

        if not records:
            st.info("No query history yet. Use the Query Interface to get started.")
            return

        # Filter options
        col1, col2, col3 = st.columns(3)
        with col1:
            query_filter = st.text_input("Filter by query content")
        with col2:
            type_filter = st.selectbox(
                "Filter by query type",
                ["All", "Temperature", "Crew", "Voyage", "General"]
            )
        with col3:
            feedback_filter = st.selectbox(
                "Filter by feedback status",
                ["All", "With Feedback", "Without Feedback"]
            )

        # Apply filters
        filtered_records = records
        if query_filter:
            filtered_records = [r for r in filtered_records if query_filter.lower() in r['query'].lower()]
        if type_filter != "All":
            filtered_records = [r for r in filtered_records if r.get('query_type', 'general').lower() == type_filter.lower()]
        if feedback_filter == "With Feedback":
            filtered_records = [r for r in filtered_records if r.get('feedback')]
        elif feedback_filter == "Without Feedback":
            filtered_records = [r for r in filtered_records if not r.get('feedback')]

        # Display summary stats
        with_feedback = sum(1 for r in filtered_records if r.get('feedback'))
        st.caption(f"Showing {len(filtered_records)} records ({with_feedback} with feedback)")

        # Display records
        for record in filtered_records:
            self._render_record(record)

    def _render_record(self, record):
        # Create a container for each record
        has_feedback = bool(record.get('feedback'))

        # Create a colored border based on feedback presence
        border_color = "#4CAF50" if has_feedback else "#e0e0e0"

        with st.container():
            st.markdown(f"""
                <div style="border: 2px solid {border_color}; border-radius: 5px; padding: 10px; margin: 10px 0;">
                    <h4>{record['query']}</h4>
                    <p><strong>Time:</strong> {record['timestamp']} | <strong>Type:</strong> {record.get('query_type', 'general')}</p>
                </div>
            """, unsafe_allow_html=True)

            # Show feedback badge if available
            if has_feedback:
                feedback = record['feedback']
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Accuracy", f"{feedback['accuracy']}/5")
                with col2:
                    st.metric("Completeness", f"{feedback['completeness']}/5")
                with col3:
                    st.metric("Relevance", f"{feedback['relevance']}/5")
                with col4:
                    st.metric("Hallucination", "Yes" if feedback['hallucination'] else "No")

                if feedback.get('comments'):
                    st.info(f"Feedback Comments: {feedback['comments']}")

            # Expandable section for details
            with st.expander("View Details"):
                col1, col2 = st.columns([3, 1])

                with col1:
                    st.subheader("Response")
                    st.info(record['response'])

                    if record.get('augmented_context'):
                        st.subheader("Context")
                        with st.expander("View Full Context"):
                            st.text(record['augmented_context'])

                with col2:
                    st.subheader("Metadata")
                    st.write(f"**Type:** {record.get('query_type', 'general')}")
                    st.write(f"**Strategy:** {record.get('retrieval_strategy', 'vector')}")
                    st.write(f"**Date:** {record['timestamp']}")

                    # Show used sources if available
                    if record.get('used_sources'):
                        st.subheader("Sources")
                        for i, source in enumerate(record['used_sources'][:]):  # Show first n sources
                            metadata = source.get('metadata', {})
                            st.write(f"{i+1}. {metadata.get('book', 'Unknown')} - Page {metadata.get('page', 'N/A')}")

            st.markdown("---")