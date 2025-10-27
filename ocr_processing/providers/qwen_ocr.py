import os
import time
import requests
import json
from tqdm import tqdm
from typing import Dict, List
from ocr_processing.utils import image_processing

class QwenOCRProvider:
    def __init__(self, config: Dict):
        self.api_key = config.get('qwen_api_key') or os.getenv("QWEN_API_KEY")
        self.model = config.get('qwen_model', "qwen2.5-vl-72b-instruct")
        self.max_retries = config.get('max_retries', 3)
        self.base_url = "https://dashscope-intl.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation"
        self.timeout = config.get('timeout', 60)
        self.min_pixels = config.get('min_pixels', 3136)
        self.max_pixels = config.get('max_pixels', 6422528)

    def _make_request(self, image_url: str, prompt_text: str) -> str:
        """Make API request to Qwen using the correct DashScope API format"""
        if not self.api_key:
            raise ValueError("Qwen API key (DASHSCOPE_API_KEY) is required")

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        # Qwen API uses a different structure than OpenAI
        payload = {
            "model": self.model,
            "input": {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "image": image_url,
                                "min_pixels": self.min_pixels,
                                "max_pixels": self.max_pixels
                            },
                            {
                                "text": prompt_text
                            }
                        ]
                    }
                ]
            },
            "parameters": {
                "result_format": "message",
                "max_tokens": 4096,
                "temperature": 0.1,
                "top_p": 0.001,  # Recommended for VL models
                "repetition_penalty": 1.05
            }
        }

        response = requests.post(
            self.base_url,
            headers=headers,
            json=payload,
            timeout=self.timeout
        )

        if response.status_code == 200:
            result = response.json()
            # Extract content from Qwen's response format
            output = result.get('output', {})
            choices = output.get('choices', [])
            if choices:
                message = choices[0].get('message', {})
                content = message.get('content', [])
                if content and isinstance(content, list):
                    # Find text content in the response
                    for item in content:
                        if isinstance(item, dict) and 'text' in item:
                            return item['text']
                elif isinstance(content, str):
                    return content
            raise ValueError("Unexpected response format from Qwen API")
        elif response.status_code == 429:
            # Rate limit exceeded
            raise requests.exceptions.HTTPError(f"Rate limit exceeded: {response.status_code}")
        else:
            try:
                error_detail = response.json()
                error_msg = error_detail.get('message', f"HTTP {response.status_code}")
            except:
                error_msg = f"HTTP {response.status_code}"
            raise requests.exceptions.HTTPError(f"Qwen API error: {error_msg}")

    def process_image(self, image_url: str, output_path: str, prompt_text: str) -> bool:
        """Process a single image and save OCR results"""
        for attempt in range(self.max_retries):
            try:
                ocr_text = self._make_request(image_url, prompt_text)

                # Ensure output directory exists
                os.makedirs(os.path.dirname(output_path), exist_ok=True)

                # Save OCR text to file
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(ocr_text)

                return True

            except requests.exceptions.HTTPError as e:
                if "429" in str(e):  # Rate limit
                    wait_time = 2 ** attempt
                    print(f"Rate limit hit, waiting {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    print(f"Qwen API HTTP error: {str(e)}")
                    if attempt == self.max_retries - 1:
                        return False
            except requests.exceptions.RequestException as e:
                print(f"Qwen API request error: {str(e)}")
                if attempt == self.max_retries - 1:
                    return False
            except json.JSONDecodeError as e:
                print(f"JSON decode error: {str(e)}")
                if attempt == self.max_retries - 1:
                    return False
            except Exception as e:
                print(f"Unexpected error: {str(e)}")
                if attempt == self.max_retries - 1:
                    return False

            # Wait before retry (except for rate limits which have their own wait)
            if attempt < self.max_retries - 1:
                time.sleep(1)

        return False

    def process_book(self, book_path: str, output_root: str, prompt_text: str) -> int:
        """Process all images in a book directory"""
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
            else:
                print(f"Failed to process {custom_id}")

        return processed

    def test_connection(self) -> bool:
        """Test if the API connection is working"""
        try:
            # Simple test with a minimal request using a test image
            test_payload = {
                "model": self.model,
                "input": {
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "image": "https://dashscope.oss-cn-beijing.aliyuncs.com/images/dog_and_girl.jpeg",
                                    "min_pixels": self.min_pixels,
                                    "max_pixels": self.max_pixels
                                },
                                {
                                    "text": "What do you see in this image?"
                                }
                            ]
                        }
                    ]
                },
                "parameters": {
                    "result_format": "message",
                    "max_tokens": 50
                }
            }

            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }

            response = requests.post(
                self.base_url,
                headers=headers,
                json=test_payload,
                timeout=10
            )

            if response.status_code == 200:
                print("✓ Qwen API connection successful!")
                return True
            else:
                print(f"✗ Connection test failed: HTTP {response.status_code}")
                return False

        except Exception as e:
            print(f"✗ Connection test failed: {str(e)}")
            return False

    def get_available_models(self) -> List[str]:
        """Return list of available Qwen VL models"""
        return [
            "qwen2.5-vl-72b-instruct",
            "qwen2.5-vl-32b-instruct",
            "qwen2.5-vl-7b-instruct",
            "qwen2.5-vl-3b-instruct",
            "qwen-vl-max",
            "qwen-vl-plus",
            "qwen-vl-ocr"  # Specialized OCR model
        ]

    def set_model(self, model_name: str):
        """Set the model to use"""
        if model_name in self.get_available_models():
            self.model = model_name
            print(f"Model set to: {model_name}")
        else:
            print(f"Warning: {model_name} might not be a valid model name")
            print(f"Available models: {self.get_available_models()}")
            self.model = model_name  # Still allow setting in case of new models