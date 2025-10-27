import streamlit as st
import yaml
from pathlib import Path

class ConfigEditor:
    def __init__(self):
        self.config_path = "config/ui_settings.yaml"

    def render(self):
        st.header("System Configuration")

        # Warning for configuration changes
        st.warning("""
        Changing these settings may affect system performance and results. 
        Only modify if you understand the implications.
        """)

        # Load current configuration
        config = self.load_config()

        # Configuration tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "Retrieval Settings",
            "Generation Settings",
            "UI Preferences",
            "Advanced"
        ])

        with tab1:
            config = self.render_retrieval_settings(config)

        with tab2:
            config = self.render_generation_settings(config)

        with tab3:
            config = self.render_ui_settings(config)

        with tab4:
            config = self.render_advanced_settings(config)

        # Save button
        if st.button("Save Configuration", type="primary"):
            self.save_config(config)
            st.success("Configuration saved successfully!")

    def load_config(self):
        with open(self.config_path) as f:
            return yaml.safe_load(f)

    def save_config(self, config):
        with open(self.config_path, 'w') as f:
            yaml.dump(config, f)

    def render_retrieval_settings(self, config):
        st.subheader("Retrieval Configuration")

        config["retrieval"]["top_k"] = st.slider(
            "Number of documents to retrieve",
            min_value=1,
            max_value=20,
            value=config["retrieval"]["top_k"],
            help="How many source documents to retrieve for each query"
        )

        config["retrieval"]["similarity_threshold"] = st.slider(
            "Similarity threshold",
            min_value=0.0,
            max_value=1.0,
            value=config["retrieval"]["similarity_threshold"],
            step=0.05,
            help="Minimum similarity score for documents to be included"
        )

        config["retrieval"]["diversity"] = st.checkbox(
            "Enable result diversity",
            value=config["retrieval"]["diversity"],
            help="Ensure retrieved documents cover different aspects of the query"
        )

        return config

    def render_generation_settings(self, config):
        st.subheader("Generation Configuration")

        config["generation"]["model"] = st.selectbox(
            "LLM Model",
            ["mistral-7b-instruct", "llama2-13b-chat", "claude-instant", "gpt-3.5-turbo"],
            index=["mistral-7b-instruct", "llama2-13b-chat", "claude-instant", "gpt-3.5-turbo"].index(
                config["generation"]["model"]
            )
        )

        config["generation"]["temperature"] = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=config["generation"]["temperature"],
            step=0.1,
            help="Higher values make output more creative, lower values more deterministic"
        )

        config["generation"]["max_length"] = st.slider(
            "Max response length",
            min_value=100,
            max_value=1000,
            value=config["generation"]["max_length"],
            step=50,
            help="Maximum length of generated responses in tokens"
        )

        return config

    def render_ui_settings(self, config):
        st.subheader("UI Preferences")

        config["ui"]["theme"] = st.selectbox(
            "Color theme",
            ["Light", "Dark", "Ocean Blue"],
            index=["Light", "Dark", "Ocean Blue"].index(config["ui"]["theme"])
        )

        config["ui"]["results_per_page"] = st.slider(
            "History items per page",
            min_value=5,
            max_value=50,
            value=config["ui"]["results_per_page"],
            step=5
        )

        config["ui"]["show_timestamps"] = st.checkbox(
            "Show timestamps",
            value=config["ui"]["show_timestamps"]
        )

        return config

    def render_advanced_settings(self, config):
        st.subheader("Advanced Settings")

        # Raw YAML editor
        st.warning("Edit the raw YAML configuration. Incorrect formatting may break the application.")
        edited_config = st.text_area(
            "Raw configuration",
            value=yaml.dump(config),
            height=400
        )

        if st.button("Parse YAML"):
            try:
                return yaml.safe_load(edited_config)
            except yaml.YAMLError as e:
                st.error(f"Invalid YAML: {e}")

        return config
    