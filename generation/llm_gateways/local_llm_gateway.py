from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch
from typing import Dict, Any

class LocalLLMGateway:
    def __init__(self, model_name: str = "mistralai/Mistral-7B-Instruct-v0.2"):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.generator = None
    
    def load_model(self):
        """Load the model and tokenizer"""
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map="auto",
            torch_dtype=torch.float16
        )
        self.generator = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if torch.cuda.is_available() else -1
        )
    
    def generate(self, prompt: str, max_length: int = 1000, 
                 temperature: float = 0.3) -> Dict[str, Any]:
        """Generate text locally"""
        if not self.generator:
            self.load_model()
        
        try:
            response = self.generator(
                prompt,
                max_length=max_length,
                temperature=temperature,
                num_return_sequences=1,
                do_sample=True
            )
            # return {"choices": [{"message": {"content": response[0]["generated_text"]}}]}
            return response[0]["generated_text"]
        except Exception as e:
            raise Exception(f"Local generation error: {str(e)}")
