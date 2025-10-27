class PromptOptimizer:
    def __init__(self, feedback_db=None):
        self.feedback_db = feedback_db
    
    def optimize(self, prompt: str, query: str, response: str, feedback: dict) -> str:
        """Refine prompts based on human feedback"""
        # Apply feedback-based modifications
        if feedback.get("accuracy") < 3:
            prompt = self._enhance_accuracy_constraints(prompt)
        if feedback.get("completeness") < 3:
            prompt = self._add_completeness_requirements(prompt)
        if feedback.get("hallucination"):
            prompt = self._strengthen_context_constraints(prompt)
        
        return prompt
    
    def _enhance_accuracy_constraints(self, prompt: str) -> str:
        """Add precision requirements to prompt"""
        if "exact values" not in prompt:
            return prompt + "\n- When reporting measurements, always include exact values with units."
        return prompt
    
    def _add_completeness_requirements(self, prompt: str) -> str:
        """Ensure all aspects of query are addressed"""
        if "all parts" not in prompt:
            return prompt + "\n- Ensure your response addresses all aspects of the query."
        return prompt
    
    def _strengthen_context_constraints(self, prompt: str) -> str:
        """Prevent hallucination by stricter context usage"""
        constraints = [
            "Only use information explicitly stated in the context.",
            "If the context doesn't contain an answer, state 'No relevant information found'."
        ]
        for constraint in constraints:
            if constraint not in prompt:
                return prompt + "\n- " + constraint
        return prompt
