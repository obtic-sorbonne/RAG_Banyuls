import os
import time
import torch
from PIL import Image
from io import BytesIO
import base64
from typing import Dict, List

class LocalOCRProvider:
    def __init__(self, config: Dict):
        self.model_name = config.get('hf_model', "Qwen/Qwen2.5-32B-Instruct")
        self.device = config.get('device', "cuda" if torch.cuda.is_available() else "cpu")
        self.torch_dtype = config.get('torch_dtype', torch.float16)
        self.model = None
        self.tokenizer = None
        self.processor = None

    def _load_model(self):
        if self.model is None:
            print(f"Loading local model: {self.model_name} on {self.device}...")
            try:
                from transformers import AutoTokenizer, AutoProcessor, AutoModelForCausalLM
                
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name,
                    trust_remote_code=True
                )
                self.processor = AutoProcessor.from_pretrained(
                    self.model_name
                )
                
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=self.torch_dtype,
                    device_map="auto"
                ).eval()
            except Exception as e:
                print(f"Error loading model: {str(e)}")
                raise

    def process_image(self, image_url: str, output_path: str, prompt_text: str) -> bool:
        self._load_model()
        
        if image_url.startswith("data:image/jpeg;base64,"):
            base64_data = image_url.split(",")[1]
            image_data = base64.b64decode(base64_data)
            pil_img = Image.open(BytesIO(image_data))
        else:
            print("Invalid image format for local processing")
            return False

        try:
            inputs = self.processor(
                text=prompt_text,
                images=pil_img,
                return_tensors="pt"
            ).to(self.device)
            
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=1024,
                do_sample=False
            )
            
            ocr_text = self.processor.batch_decode(
                generated_ids,
                skip_special_tokens=True
            )[0]
            
            # Extract only the assistant's response
            ocr_text = ocr_text.split("<|im_start|>assistant\n")[-1].strip()
            
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(ocr_text)
            return True
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                print("CUDA OOM error - try reducing image size or using CPU")
            else:
                print(f"Runtime error: {str(e)}")
            return False
        except Exception as e:
            print(f"Local processing error: {str(e)}")
            return False

    def process_book(self, book_path: str, output_root: str, prompt_text: str) -> int:
        from ocr_processing.utils import image_processing
        _, file_map = image_processing.create_image_urls(book_path, output_root)
        processed = 0
        for custom_id, file_info in file_map.items():
            success = self.process_image(
                file_info["image_url"],
                file_info["output_path"],
                prompt_text
            )
            if success:
                processed += 1
        return processed

