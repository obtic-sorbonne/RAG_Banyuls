from .base_prompts import BasePrompts

class FrenchPrompts(BasePrompts):
    @property
    def system_prompt(self):
        return """
        Vous êtes un expert en analyse de données historiques maritimes. 
        Vous répondez aux questions basées sur des documents historiques numérisés.
        Vous êtes précis, factuel, et fournissez toujours des citations claires.
        """
    
    def rag_prompt(self, context: str, query: str) -> str:
        base_prompt = super().rag_prompt(context, query)

        enhanced_prompt = f"""
        Instructions supplémentaires:
        1. Répondez en français avec un style formel et académique
        2. Citez vos sources en mentionnant le livre et la page pour chaque information
        3. Si les données sont incomplètes, mentionnez-le explicitement
        4. Structurez votre réponse de manière logique
        5. Incluez les dates spécifiques lorsque disponibles

        Contexte:
        {context}

        Question: {query}

        Réponse:
        """

        return enhanced_prompt