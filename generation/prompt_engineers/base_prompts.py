class BasePrompts:
    ## TODO change it fix type of strategy like add sql resaut
    @property
    def system_prompt(self):
        return (
            "You are an expert oceanographer analyzing historical ocean observation data. "
            "Your responses should be precise, factual, and based strictly on the provided context. "
            "When numerical data is available, include exact values with units. For temporal references, "
            "use ISO 8601 format (YYYY-MM-DD). Always cite your sources using the document metadata."
        )
    
    def rag_prompt(self, context: str, query: str) -> str:
        return f"""
        ### Context:
        {context}
        
        ### Instruction:
        Using ONLY the information from the context above, answer the following query.
        If the context doesn't contain the answer, state that you couldn't find relevant information.
        
        ### Query:
        {query}
        
        ### Response:
        """
    
    def analytical_prompt(self, data: str, question: str) -> str:
        return f"""
        ### Data:
        {data}
        
        ### Task:
        Perform detailed analysis on this data to answer: {question}
        Include trends, comparisons, and statistical insights where applicable.
        """
    
    def summary_prompt(self, context: str, focus: str = None) -> str:
        focus_clause = f" focusing on {focus}" if focus else ""
        return f"""
        ### Context:
        {context}
        
        ### Task:
        Create a comprehensive summary{focus_clause} organized by key themes. 
        Include important dates, measurements, and observations. 
        Preserve all numerical precision and units.
        """
