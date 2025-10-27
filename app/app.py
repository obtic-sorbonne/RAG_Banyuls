# app.py
# Environment
import sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import yaml
from components.query_interface import QueryInterface
from components.history_viewer import HistoryViewer
from components.feedback_panel import FeedbackPanel
from components.annotation_interface import AnnotationInterface

import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from generation.generation_manager import GenerationManager


class OceanRAGApp:
    def __init__(self):
        self.setup_page_config()
        self.load_config()

    def setup_page_config(self):
        st.set_page_config(
            page_title="Ocean Observation RAG System",
            page_icon="🌊",
            layout="wide",
            initial_sidebar_state="expanded"
        )

        # Custom CSS
        try:
            with open("app/assets/styles.css") as f:
                st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
        except FileNotFoundError:
            st.warning("Custom CSS file not found. Using default styles.")

    def load_config(self):
        try:
            with open("config/ui_settings.yaml") as f:
                self.config = yaml.safe_load(f)
        except FileNotFoundError:
            self.config = {}
            st.warning("UI configuration file not found. Using default settings.")

        self.mode = self.config.get("mode", "RAG")

    def render_annotation_sidebar(self):
        """Render sidebar for Annotation mode"""
        with st.sidebar:
            try:
                st.image("app/assets/icons/logo.jpg", width=200)
            except FileNotFoundError:
                st.title("OCR Annotation Tool")

            # Navigation - Only annotation interface in this mode
            app_mode = "OCR Annotation"

            # Annotation statistics
            st.divider()
            st.subheader("Annotation Progress")

            try:
                annotation_interface = AnnotationInterface()
                stats = annotation_interface.get_annotation_stats()

                col1, col2 = st.columns(2)
                col1.metric("Total Books", stats["total_books"])
                col2.metric("Total Images", stats["total_images"])

                st.divider()

                col3, col4 = st.columns(2)
                col3.metric("Annotated", stats["annotated_images"],
                            f"{stats['completion_percentage']:.1f}%")
                col4.metric("Remaining", stats["remaining_images"])

                # Progress bar
                st.progress(stats["completion_percentage"] / 100)

            except Exception as e:
                st.error(f"Error loading annotation stats: {e}")
                st.metric("Total Images", 0)
                st.metric("Annotated", 0)

            return app_mode

    def render_rag_sidebar(self):
        """Render sidebar for RAG mode"""
        with st.sidebar:
            try:
                st.image("app/assets/icons/logo.jpg", width=200)
            except FileNotFoundError:
                st.title("Ocean Observation RAG")

            # Navigation
            app_mode = st.radio(
                "Navigation",
                ["Query Interface", "History Viewer", "Feedback Analysis"],
                index=0
            )

            # System status
            st.divider()
            st.subheader("System Status")

            # Initialize RAG engine for metrics if needed
            if 'rag_engine' not in st.session_state:
                st.session_state.rag_engine = GenerationManager()

            # Get metrics
            try:
                metrics = st.session_state.rag_engine.get_quality_metrics(last_n=100)
                human_metrics = metrics['human_metrics']

                col1, col2 = st.columns(2)
                col1.metric("Avg Accuracy", f"{human_metrics['accuracy']:.1f}/5")
                col2.metric("Hallucination", f"{human_metrics['hallucination_rate'] * 100:.1f}%")

                # Quick stats
                st.divider()
                st.subheader("Recent Activity")

                records = st.session_state.rag_engine.get_recent_records(limit=10)
                st.metric("Recent Queries", len(records))
                st.metric("Feedback Received", human_metrics['feedback_count'])
            except Exception as e:
                st.error(f"Error loading metrics: {e}")
                st.metric("Recent Queries", 0)
                st.metric("Feedback Received", 0)

            return app_mode

    def render_sidebar(self):
        with st.sidebar:
            try:
                st.image("app/assets/icons/logo.jpg", width=200)
            except FileNotFoundError:
                st.title("Ocean Observation RAG")

            # Navigation
            app_mode = st.radio(
                "Navigation",
                ["Query Interface", "History Viewer", "Feedback Analysis",
                 "OCR Annotation"
                 ],
                index=0
            )

            # System status
            st.divider()
            st.subheader("System Status")

            # Initialize RAG engine for metrics if needed
            if 'rag_engine' not in st.session_state:
                st.session_state.rag_engine = GenerationManager()

            # Get metrics
            try:
                metrics = st.session_state.rag_engine.get_quality_metrics(last_n=100)
                human_metrics = metrics['human_metrics']

                col1, col2 = st.columns(2)
                col1.metric("Avg Accuracy", f"{human_metrics['accuracy']:.1f}/5")
                col2.metric("Hallucination", f"{human_metrics['hallucination_rate'] * 100:.1f}%")

                # Quick stats
                st.divider()
                st.subheader("Recent Activity")

                records = st.session_state.rag_engine.get_recent_records(limit=10)
                st.metric("Recent Queries", len(records))
                st.metric("Feedback Received", human_metrics['feedback_count'])
            except Exception as e:
                st.error(f"Error loading metrics: {e}")
                st.metric("Recent Queries", 0)
                st.metric("Feedback Received", 0)

            return app_mode

    def run(self):
        # Initialize session state
        if 'query_history' not in st.session_state:
            st.session_state.query_history = []

        # Display mode indicator
        st.sidebar.markdown(f"**Mode:** {self.mode}")

        # Render appropriate sidebar and content based on config mode
        if self.mode == "RAG":
            app_mode = self.render_rag_sidebar()

            # Render main content based on selection
            if app_mode == "Query Interface":
                QueryInterface().render()
            elif app_mode == "History Viewer":
                HistoryViewer().render()
            elif app_mode == "Feedback Analysis":
                FeedbackPanel().render()

        elif self.mode == "ANNOTATION":  # Annotation mode
            app_mode = self.render_annotation_sidebar()
            AnnotationInterface().render()
        else:
            app_mode = self.render_sidebar()
            # Render main content based on selection

            if app_mode == "Query Interface":
                QueryInterface().render()
            elif app_mode == "History Viewer":
                HistoryViewer().render()
            elif app_mode == "Feedback Analysis":
                FeedbackPanel().render()
            elif app_mode == "OCR Annotation":
                AnnotationInterface().render()


if __name__ == "__main__":
    app = OceanRAGApp()
    app.run()
