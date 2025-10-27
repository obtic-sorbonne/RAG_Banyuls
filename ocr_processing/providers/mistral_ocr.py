import os
import time
import mistralai
from mistralai import Mistral
from tqdm import tqdm
from typing import Dict, List
from ocr_processing.utils import image_processing

class MistralOCRProvider:
    def __init__(self, config: Dict):
        self.api_key = config.get('mistral_api_key') or os.getenv("MISTRAL_API_KEY")
        self.batch_size = config.get('batch_size', 3)
        self.max_retries = config.get('max_retries', 3)
        self.rate_limit_delay = config.get('rate_limit_delay', 1.0)
        self.ocr_model = config.get('model', "mistral-ocr-latest")
        self.client = None

    def _get_client(self):
        if self.client is None:
            if not self.api_key:
                raise ValueError("Mistral API key is required")
            self.client = Mistral(api_key=self.api_key)
        return self.client

    def process_image(self, image_url: str, output_path: str, prompt_text: str) -> bool:
        client = self._get_client()
        for attempt in range(self.max_retries):
            try:
                response = client.ocr.process(
                    model=self.ocr_model,
                    document={"type": "image_url", "image_url": image_url}
                )
                if response.pages:
                    ocr_text = response.pages[0].markdown
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)
                    with open(output_path, "w", encoding="utf-8") as f:
                        f.write(ocr_text)
                    return True
            except mistralai.models.sdkerror.SDKError as e:
                if e.status_code == 429:
                    time.sleep(2 ** attempt)
                else:
                    print(f"Mistral API error: {str(e)}")
            except Exception as e:
                print(f"Error processing image: {str(e)}")
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
            time.sleep(self.rate_limit_delay)
        return processed

