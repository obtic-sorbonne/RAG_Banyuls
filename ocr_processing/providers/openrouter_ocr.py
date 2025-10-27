import os
import time
from openai import OpenAI
from tqdm import tqdm
from typing import Dict, List
from ocr_processing.utils import image_processing

class OpenRouterOCRProvider:
    def __init__(self, config: Dict):
        self.api_key = config.get('openrouter_api_key') or os.getenv("OPENROUTER_API_KEY")
        self.model = config.get('openrouter_model', "qwen/qwen2.5-vl-32b-instruct:free")
        self.max_retries = config.get('max_retries', 3)
        self.http_referer = config.get('http_referer')
        self.x_title = config.get('x_title')
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

    def process_image(self, image_url: str, output_path: str, prompt_text: str) -> bool:
        client = self._get_client()
        extra_headers = {}
        if self.http_referer:
            extra_headers["HTTP-Referer"] = self.http_referer
        if self.x_title:
            extra_headers["X-Title"] = self.x_title

        for attempt in range(self.max_retries):
            try:
                response = client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt_text},
                                {"type": "image_url", "image_url": {"url": image_url}}
                            ]
                        }
                    ],
                    max_tokens=4096,
                    **({'extra_headers': extra_headers} if extra_headers else {})
                )
                ocr_text = response.choices[0].message.content
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(ocr_text)
                return True
            except Exception as e:
                if hasattr(e, 'status_code') and e.status_code == 429:
                    time.sleep(2 ** attempt)
                else:
                    print(f"OpenRouter API error: {str(e)}")
        return False

    def process_book(self, book_path: str, output_root: str, prompt_text: str) -> int:
        _, file_map = image_processing.create_image_urls(book_path, output_root)
        if not file_map:
            print("No valid images found")
            return 0
        
        processed = 0
        for custom_id, file_info in tqdm(file_map.items(), desc="Processing pages"):
            success = self.process_image(
                file_info["image_url"],
                file_info["output_path"],
                prompt_text
            )
            if success:
                processed += 1
        return processed

