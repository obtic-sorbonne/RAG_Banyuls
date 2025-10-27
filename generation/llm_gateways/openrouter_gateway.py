import os
import time
from typing import Dict, List, Any
from openai import OpenAI


class OpenRouterGateway:
    def __init__(self, api_key: str, model: str = "qwen/mistral-7b-instruct", max_retries: int = 3):
        self.api_key = api_key
        self.model = model
        self.max_retries = max_retries
        # self.max_retries = config.get("max_retries", 3)
        # self.http_referer = config.get("http_referer")
        # self.x_title = config.get("x_title")
        self.client = None

    def _get_client(self):
        if self.client is None:
            if not self.api_key:
                raise ValueError("OpenRouter API key is required")
            self.client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=self.api_key
            )
        return self.client

    def generate(self, messages: List[Dict[str, Any]],
                 temperature: float = 0.3, max_tokens: int = 1000) -> str:
        """Generate text using OpenRouter API with retry logic"""
        client = self._get_client()
        extra_headers = {}
        # if self.http_referer:
        #     extra_headers["HTTP-Referer"] = self.http_referer
        # if self.x_title:
        #     extra_headers["X-Title"] = self.x_title

        for attempt in range(self.max_retries):
            try:
                response = client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **({"extra_headers": extra_headers} if extra_headers else {})
                )
                return response.choices[0].message.content
            except Exception as e:
                if hasattr(e, 'status_code') and e.status_code == 429:
                    time.sleep(2 ** attempt)  # exponential backoff
                else:
                    raise Exception(f"OpenRouter API error: {str(e)}")
        return ""

    def format_messages(self, system_prompt: str, user_prompt: str) -> List[Dict[str, str]]:
        """Format messages for the API"""
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
