import streamlit as st
import uuid
from datetime import datetime

class QueryInterface:
    def __init__(self):
        # Initialize RAG components if not already in session state
        if 'rag_engine' not in st.session_state:
            from generation.generation_manager import GenerationManager
            st.session_state.rag_engine = GenerationManager()

        if 'retriever' not in st.session_state:
            from retrieval.retrieval_manager import RetrievalManager
            st.session_state.retriever = RetrievalManager()

        if 'query_classifier' not in  st.session_state:
            from generation.query_classifier.keyword_classifier import KeywordClassifier
            st.session_state.query_classifier = KeywordClassifier()


        # Initialize feedback form state
        if 'feedback_form' not in st.session_state:
            st.session_state.feedback_form = {}

    def render(self):
        st.header("Query Interface")

        # Query input section
        query = st.text_area(
            "Enter your question about historical maritime data:",
            placeholder="e.g., Quelles étaient les conditions à Bord en juillet 1853?",
            height=100
        )

        # Advanced options
        with st.expander("Advanced Options"):
            col1, col2 = st.columns(2)
            with col1:
                top_k = st.slider("Number of documents to retrieve", 1, 100, 5)
            with col2:
                retrieval_strategy = st.selectbox(
                    "Retrieval Strategy",
                    ["Auto", "Vector", "Keyword",
                     # "Hybrid"
                     ],
                    index=0
                )

        # Process query
        if st.button("Generate Response", type="primary") and query:
            with st.spinner("Searching historical documents and generating response..."):
                # Determine retrieval strategy
                if retrieval_strategy == "Auto":
                    strategy = st.session_state.query_classifier.get_retrieval_strategy(query)
                else:
                    strategy = [retrieval_strategy.lower()]

                # Retrieve documents
                results = st.session_state.retriever.retrieve(query, top_k=top_k, strategy=strategy)

                # Generate response
                response = st.session_state.rag_engine.generate_response(query, results)

                # Store in session state
                st.session_state.last_response = response
                st.session_state.show_feedback = True

        # Display results if available
        if 'last_response' in st.session_state:
            response = st.session_state.last_response

            # Display the response
            st.subheader("Generated Response")
            st.info(response['response'])

            # Display retrieved documents
            st.subheader("Retrieved Documents")
            for i, doc in enumerate(response['retrieved_docs']):
                with st.expander(f"Document {i+1}: {doc.get('metadata', {}).get('book', 'Unknown')} - Page {doc.get('metadata', {}).get('page', 'N/A')}"):
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.write(doc['content'])
                    with col2:
                        st.caption("Metadata")
                        st.write(f"**Source:** {doc.get('metadata', {}).get('book', 'Unknown')}")
                        st.write(f"**Page:** {doc.get('metadata', {}).get('page', 'N/A')}")
                        st.write(f"**Year:** {doc.get('metadata', {}).get('primary_year', 'Unknown')}")
                        st.write(f"**Relevance Score:** {doc.get('score', 0):.3f}")

            # Feedback section
            if st.session_state.get('show_feedback', False):
                self._render_feedback_form(response['record_id'])

    def _render_feedback_form(self, record_id):
        st.subheader("Provide Feedback")

        with st.form("feedback_form"):
            st.write("How would you rate this response?")

            col1, col2, col3 = st.columns(3)

            with col1:
                accuracy = st.slider(
                    "Accuracy",
                    1, 5, 3,
                    help="How accurate is the information compared to the source documents?"
                )
            with col2:
                completeness = st.slider(
                    "Completeness",
                    1, 5, 3,
                    help="How complete is the answer relative to what's available in the documents?"
                )
            with col3:
                relevance = st.slider(
                    "Relevance",
                    1, 5, 3,
                    help="How relevant is the answer to your original question?"
                )

            # Hallucination detection
            hallucination = st.radio(
                "Does the response contain any hallucinations or incorrect information?",
                ["No", "Yes"],
                index=0
            )

            # Specific issue checkboxes
            st.write("Select any specific issues you noticed:")
            col1, col2 = st.columns(2)
            with col1:
                missing_info = st.checkbox("Missing important information")
                outdated_info = st.checkbox("Outdated or incorrect information")
            with col2:
                irrelevant_info = st.checkbox("Irrelevant information")
                confusing_structure = st.checkbox("Confusing structure or language")

            comments = st.text_area(
                "Additional comments",
                placeholder="Any additional feedback about this response..."
            )

            submitted = st.form_submit_button("Submit Feedback")

            if submitted:
                # Record feedback
                st.session_state.rag_engine.record_feedback(
                    record_id=record_id,
                    accuracy=accuracy,
                    completeness=completeness,
                    relevance=relevance,
                    hallucination=(hallucination == "Yes"),
                    comments=comments
                )

                st.success("Thank you for your feedback! It helps us improve the system.")

                # Clear the form
                st.session_state.show_feedback = False
                if 'last_response' in st.session_state:
                    del st.session_state.last_response