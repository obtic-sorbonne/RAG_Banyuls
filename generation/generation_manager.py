import os
import yaml
from typing import Dict, List, Optional

from .prompt_engineers import FrenchPrompts, DomainPrompts
from .llm_gateways import OpenRouterGateway, LocalLLMGateway, FallbackStrategy
from .augmentation import ContextAugmenter
from .feedback_integration import FeedbackManager
from .prompt_engineers.prompt_optimizer import PromptOptimizer


class GenerationManager:
    def __init__(self):
        with open("config/generation_settings.yaml", "r") as f:
            self.config = yaml.safe_load(f)
        self.prompts = FrenchPrompts()
        # self.domain = DomainPrompts()
        self.gateway = self._init_gateway()
        self.augmenter = ContextAugmenter()
        self.feedback = FeedbackManager(db_path=self.config.get("feedback_db_path", "./data/feedback/feedback.db"))
        self.optimizer = PromptOptimizer()

    def _init_gateway(self):
        primary = OpenRouterGateway(api_key=self.config.get("openrouter_key") or os.getenv("OPENROUTER_API_KEY"),
                                    model=self.config.get("openrouter_model", "qwen/qwen-2.5-72b-instruct:free"))
        fallback = LocalLLMGateway(model_name=self.config.get("local_model", "mistralai/Mistral-7B-Instruct-v0.2"))
        return FallbackStrategy(primary, fallback)

    def generate_response(self, query: str, retrieval_results: List[Dict],
                          query_metadata: dict = None, query_type: str = None, record_id: int = None) -> dict:
        """Generate a response using augmented context and optimized prompts"""
        # Augment context with temporal and entity information
        # augmented_context = self.augmenter.augment(context, query, query_metadata)
        # Augment context with metadata
        augmented_context = self.augmenter.augment_with_metadata(retrieval_results)

        # Get base prompt
        prompt_text = self.prompts.rag_prompt(augmented_context, query)

        # Apply domain-specific optimizations
        # if "température" in query.lower():
        #     prompt_text += "\n\n" + self.domain.temperature_analysis
        # elif "résumé" in query.lower():
        #     prompt_text += "\n\n" + self.domain.voyage_summary

        # Generate response
        messages = self.gateway.format_messages(
            self.prompts.system_prompt,
            prompt_text
        )

        # response = self.gateway.generate(messages)
        # content = response["choices"][0]["message"]["content"]

        response = self.gateway.generate(messages)
        # Record feedback record
        if record_id is None:
            record_id = self.feedback.record_record(query=query, response=response, augmented_context=augmented_context,
                                                    used_sources=retrieval_results,
                                                    prompt=prompt_text)

        return {
            "response": response,
            "augmented_context": augmented_context,
            "prompt": prompt_text,
            "query_type": query_type,
            "retrieved_docs": retrieval_results,
            "record_id": record_id
        }


    def record_feedback(self, record_id: str, accuracy: int,
                        completeness: int, relevance: int,
                        hallucination: bool, comments: str = "") -> None:
        """Record human feedback for a generation record"""
        self.feedback.record_human_feedback(
            record_id, accuracy, completeness, relevance,
            hallucination, comments
        )

    def get_recent_records(self, limit=10) -> List[Dict]:
        """Get recent generation records"""
        return self.feedback.get_recent_records(limit)

    def get_quality_metrics(self, last_n=100) -> Dict:
        """Get quality metrics from feedback"""
        return self.feedback.get_quality_metrics(last_n)

    def get_record(self, record_id: str) -> Optional[Dict]:
        """Get a specific record by ID"""
        return self.feedback.get_record(record_id)

    def process_feedback(self, record_id: str, accuracy: int,
                         completeness: int, relevance: int,
                         hallucination: bool, comments: str = "") -> None:
        """Process human feedback and optimize prompts"""
        # Record feedback
        self.feedback.record_human_feedback(
            record_id, accuracy, completeness, relevance,
            hallucination, comments
        )

        # Get the original generation data
        rcd = self.feedback.get_record(record_id)
        if rcd:
            # Optimize prompt for future similar queries
            optimized_prompt = self.optimizer.optimize(
                rcd["prompt"], rcd["query"], rcd["response"],
                {"accuracy": accuracy, "completeness": completeness,
                 "relevance": relevance, "hallucination": hallucination}
            )
            # Store optimized prompt for future use
            self._store_optimized_prompt(rcd["query"], optimized_prompt)

    def get_records_with_feedback(self, limit=50):
        """Get records with their feedback (if available)"""
        return self.feedback.get_records_with_feedback(limit)

    def get_feedback_trends(self, days=30):
        """Get feedback trends over time"""
        return self.feedback.get_feedback_trends(days)

    # TODO Slot for Human in the loop
    def _store_optimized_prompt(self, query_pattern: str, prompt: str) -> None:
        """Store optimized prompts for similar future queries"""
        # This would typically be stored in a database
        print(f"Storing optimized prompt for queries like: {query_pattern}")
        # Implementation would save to a prompt database
