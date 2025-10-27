from transformers import pipeline

class FactualConsistency:
    def __init__(self, model_name="google/t5_xxl_true_nli_mixture"):
        self.nli_pipeline = pipeline(
            "text2text-generation",
            model=model_name,
            device=0 if torch.cuda.is_available() else -1
        )
    
    def calculate(self, response: str, context: str) -> float:
        """Calculate factual consistency score between response and context"""
        input_text = f"premise: {context} hypothesis: {response}"
        result = self.nli_pipeline(input_text, max_length=512)
        
        # Parse output - models typically return "entailment", "neutral", "contradiction"
        output = result[0]['generated_text'].lower()
        
        if "entailment" in output:
            return 1.0
        elif "contradiction" in output:
            return 0.0
        else:  # neutral
            return 0.5
