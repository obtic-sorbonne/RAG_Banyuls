from transformers import pipeline

class FluencyEvaluator:
    def __init__(self, model_name="textattack/roberta-base-CoLA"):
        self.classifier = pipeline(
            "text-classification",
            model=model_name,
            device=0 if torch.cuda.is_available() else -1
        )
    
    def calculate(self, text: str) -> float:
        """Calculate grammatical fluency score"""
        result = self.classifier(text)
        # Model returns LABEL_0 for unacceptable, LABEL_1 for acceptable
        if result[0]['label'] == 'LABEL_1':
            return result[0]['score']
        return 1 - result[0]['score']
