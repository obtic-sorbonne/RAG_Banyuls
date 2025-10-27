import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta

class FeedbackPanel:
    def __init__(self):
        if 'rag_engine' not in st.session_state:
            from generation.generation_manager import GenerationManager
            st.session_state.rag_engine = GenerationManager()

    def render(self):
        st.header("Feedback Analysis")

        # Get quality metrics
        metrics = st.session_state.rag_engine.get_quality_metrics(last_n=100)

        # Display metrics
        col1, col2, col3, col4, col5 = st.columns(5)

        human_metrics = metrics['human_metrics']
        with col1:
            st.metric("Accuracy", f"{human_metrics['accuracy']:.2f}/5",
                      help="Average accuracy rating from user feedback")
        with col2:
            st.metric("Completeness", f"{human_metrics['completeness']:.2f}/5",
                      help="Average completeness rating from user feedback")
        with col3:
            st.metric("Relevance", f"{human_metrics['relevance']:.2f}/5",
                      help="Average relevance rating from user feedback")
        with col4:
            st.metric("Hallucination Rate", f"{human_metrics['hallucination_rate']*100:.1f}%",
                      help="Percentage of responses flagged as containing hallucinations")
        with col5:
            st.metric("Feedback Count", human_metrics['feedback_count'],
                      help="Total number of feedback submissions")

        # Time period filter
        st.subheader("Feedback Trends Over Time")
        time_period = st.selectbox(
            "Time Period",
            ["Last 7 days", "Last 30 days", "Last 90 days", "All time"],
            index=1,
            key="feedback_time_period"
        )

        # Get feedback trends
        if time_period == "Last 7 days":
            days = 7
        elif time_period == "Last 30 days":
            days = 30
        elif time_period == "Last 90 days":
            days = 90
        else:
            days = 365  # All time (approx 1 year)

        trends = st.session_state.rag_engine.feedback.get_feedback_trends(days=days)

        if trends:
            # Convert to DataFrame for visualization
            df = pd.DataFrame(trends)
            df['date'] = pd.to_datetime(df['date'])

            # Create trend charts
            col1, col2 = st.columns(2)

            with col1:
                # Ratings trend
                fig_ratings = px.line(df, x='date', y=['avg_accuracy', 'avg_completeness', 'avg_relevance'],
                                      title='Average Ratings Over Time',
                                      labels={'value': 'Rating', 'variable': 'Metric', 'date': 'Date'})
                fig_ratings.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
                st.plotly_chart(fig_ratings, use_container_width=True)

            with col2:
                # Hallucination rate trend
                fig_hallucination = px.line(df, x='date', y='hallucination_rate',
                                            title='Hallucination Rate Over Time',
                                            labels={'hallucination_rate': 'Hallucination Rate', 'date': 'Date'})
                st.plotly_chart(fig_hallucination, use_container_width=True)

        # Detailed feedback table
        st.subheader("Detailed Feedback Records")

        # Get records with feedback
        records_with_feedback = st.session_state.rag_engine.feedback.get_records_with_feedback(limit=50)
        records_with_feedback = [r for r in records_with_feedback if r.get('feedback')]

        if records_with_feedback:
            # Create a DataFrame for the table
            table_data = []
            for record in records_with_feedback:
                feedback = record['feedback']
                table_data.append({
                    'Query': record['query'][:50] + "..." if len(record['query']) > 50 else record['query'],
                    'Timestamp': record['timestamp'],
                    'Accuracy': feedback['accuracy'],
                    'Completeness': feedback['completeness'],
                    'Relevance': feedback['relevance'],
                    'Hallucination': 'Yes' if feedback['hallucination'] else 'No',
                    'Comments': feedback.get('comments', '')[:30] + "..." if feedback.get('comments') and len(feedback.get('comments')) > 30 else feedback.get('comments', ''),
                    'Record ID': record['id']
                })

            df_feedback = pd.DataFrame(table_data)

            # Add filters
            col1, col2 = st.columns(2)
            with col1:
                min_accuracy = st.slider("Minimum Accuracy", 1, 5, 1, key="min_accuracy")
            with col2:
                hallucination_filter = st.selectbox(
                    "Hallucination Filter",
                    ["All", "With Hallucinations", "Without Hallucinations"],
                    key="hallucination_filter"
                )

            # Apply filters
            filtered_df = df_feedback[df_feedback['Accuracy'] >= min_accuracy]
            if hallucination_filter == "With Hallucinations":
                filtered_df = filtered_df[filtered_df['Hallucination'] == 'Yes']
            elif hallucination_filter == "Without Hallucinations":
                filtered_df = filtered_df[filtered_df['Hallucination'] == 'No']

            # Display the table
            st.dataframe(
                filtered_df,
                use_container_width=True,
                column_config={
                    "Record ID": st.column_config.TextColumn("Record ID", width="small"),
                    "Query": st.column_config.TextColumn("Query", width="large"),
                    "Timestamp": st.column_config.DatetimeColumn("Time"),
                    "Accuracy": st.column_config.NumberColumn("Accuracy", format="%d"),
                    "Completeness": st.column_config.NumberColumn("Completeness", format="%d"),
                    "Relevance": st.column_config.NumberColumn("Relevance", format="%d"),
                    "Hallucination": st.column_config.TextColumn("Hallucination"),
                    "Comments": st.column_config.TextColumn("Comments")
                },
                hide_index=True
            )

            # Add option to view details for a specific record
            selected_record_id = st.selectbox(
                "Select a record to view details",
                options=[""] + [r['id'] for r in records_with_feedback],
                format_func=lambda x: next((r['query'][:50] + "..." for r in records_with_feedback if r['id'] == x), ""),
                key="record_selector"
            )

            if selected_record_id:
                record = next((r for r in records_with_feedback if r['id'] == selected_record_id), None)
                if record:
                    with st.expander("Record Details", expanded=True):
                        st.write("**Query:**", record['query'])
                        st.write("**Response:**", record['response'])
                        st.write("**Feedback:**")
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("Accuracy", record['feedback']['accuracy'])
                        col2.metric("Completeness", record['feedback']['completeness'])
                        col3.metric("Relevance", record['feedback']['relevance'])
                        col4.metric("Hallucination", "Yes" if record['feedback']['hallucination'] else "No")

                        if record['feedback'].get('comments'):
                            st.write("**Comments:**", record['feedback']['comments'])
        else:
            st.info("No feedback records available yet. Use the Query Interface and provide feedback on responses.")