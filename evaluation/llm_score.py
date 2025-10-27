# evaluation/llm_judge.py
import os
import yaml
from typing import Dict, List
from generation.llm_gateways import OpenRouterGateway, LocalLLMGateway, FallbackStrategy

class LLMJudge:
    def __init__(self):
        with open("config/evaluation_settings.yaml", "r") as f:
            self.config = yaml.safe_load(f)

        self.gateway = self._init_gateway()
        self.evaluation_prompts = self._load_evaluation_prompts()

    def _init_gateway(self):
        primary = OpenRouterGateway(
            api_key=self.config.get("openrouter_api_key") or os.getenv("OPENROUTER_API_KEY"),
            model=self.config.get("judge_model", "anthropic/claude-3-sonnet")
        )
        fallback = LocalLLMGateway(
            model_name=self.config.get("judge_local_model", "mistralai/Mistral-7B-Instruct")
        )
        return FallbackStrategy(primary, fallback)

    def _load_evaluation_prompts(self) -> Dict:
        return {
            "faithfulness": """
            Evaluate if the generated answer is faithful to the provided context.
            Score from 0 to 1, where 1 means completely faithful and 0 means completely unfaithful.
            Consider if the answer contains information not present in the context (hallucinations).
            
            Context: {context}
            
            Answer: {response}
            
            Provide your evaluation in JSON format with keys: score, explanation, and hallucinations (list of any hallucinations found).
            """,

            "relevance": """
            Evaluate if the generated answer is relevant to the original query.
            Score from 0 to 1, where 1 means completely relevant and 0 means completely irrelevant.
            
            Query: {query}
            
            Answer: {response}
            
            Provide your evaluation in JSON format with keys: score and explanation.
            """,

            "completeness": """
            Evaluate if the generated answer completely addresses the query.
            Score from 0 to 1, where 1 means completely addressed and 0 means not addressed at all.
            
            Query: {query}
            
            Answer: {response}
            
            Provide your evaluation in JSON format with keys: score and explanation.
            """,

            "citation_quality": """
            Evaluate the quality of citations in the generated answer.
            Score from 0 to 1, where 1 means perfect citations and 0 means no or incorrect citations.
            Check if sources are properly referenced and match the context.
            
            Context: {context}
            
            Answer: {response}
            
            Provide your evaluation in JSON format with keys: score, explanation, and missing_citations (list of any uncited information).
            """
        }

    def evaluate(self, query: str, context: str, response: str,
                 evaluation_types: List[str] = None) -> Dict:
        """Evaluate a response using LLM judge"""
        if evaluation_types is None:
            evaluation_types = ["faithfulness", "relevance", "completeness", "citation_quality"]

        results = {}

        for eval_type in evaluation_types:
            if eval_type not in self.evaluation_prompts:
                continue

            prompt = self.evaluation_prompts[eval_type].format(
                query=query, context=context, response=response
            )

            messages = self.gateway.format_messages(
                "You are an expert evaluator of AI-generated content. Provide accurate, objective evaluations.",
                prompt
            )

            evaluation_result = self.gateway.generate(messages)

            try:
                # Parse JSON response
                parsed_result = json.loads(evaluation_result)
                results[eval_type] = parsed_result
            except json.JSONDecodeError:
                # Fallback if LLM doesn't return proper JSON
                results[eval_type] = {
                    "score": 0.5,
                    "explanation": "Failed to parse evaluation result",
                    "error": evaluation_result
                }

        return results