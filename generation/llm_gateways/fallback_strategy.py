class FallbackStrategy:
    def __init__(self, primary_gateway, fallback_gateway):
        self.primary = primary_gateway
        self.fallback = fallback_gateway
    
    def generate(self, messages: list, **kwargs) -> str:
        """Try primary gateway, fallback to secondary if fails"""
        try:
            return self.primary.generate(messages, **kwargs)
        except Exception as e:
            print(f"Primary gateway failed: {str(e)}. Using fallback.")
            try:
                # Convert messages to single prompt
                prompt = self._messages_to_prompt(messages)
                return self.fallback.generate(prompt, **kwargs)
            except Exception as fallback_e:
                raise Exception(f"Both gateways failed: {str(e)} | {str(fallback_e)}")
    
    def _messages_to_prompt(self, messages: list) -> str:
        """Convert message history to single prompt string"""
        prompt = ""
        for msg in messages:
            prompt += f"{msg['role'].capitalize()}: {msg['content']}\n"
        return prompt

    def format_messages(self, system_prompt: str, user_prompt: str) -> list:
        """Format messages for the API"""
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
