import os
import re
import base64
import glymur
import numpy as np
from PIL import Image
from io import BytesIO
from tqdm import tqdm

def load_jp2(image_path: str, downsample_level: int = 1) -> np.ndarray:
    """Load and downsample JPEG2000 image"""
    try:
        jp2 = glymur.Jp2k(image_path)
        return jp2[::2**downsample_level, ::2**downsample_level]
    except Exception as e:
        print(f"Error loading {image_path}: {str(e)}")
        return None

def convert_to_base64(img_array: np.ndarray) -> str:
    """Convert image array to base64-encoded JPEG"""
    try:
        if img_array is None:
            return None
        pil_img = Image.fromarray(img_array)

        # Convert to grayscale for 66% channel reduction
        if pil_img.mode != 'L':
            pil_img = pil_img.convert('L')

        buffered = BytesIO()
        pil_img.save(buffered,
                     format="JPEG",
                     quality=85,
                     optimize=True)  # Enable Huffman optimization
        img_bytes = buffered.getvalue()
        return f"data:image/jpeg;base64,{base64.b64encode(img_bytes).decode('utf-8')}"
    except Exception as e:
        print(f"Image conversion error: {str(e)}")
        return None

def create_image_urls(book_path: str, output_root: str) -> tuple:
    """Create image URLs for all pages in a book"""
    image_urls = []
    file_map = {}
    
    pages = [f for f in os.listdir(book_path) if f.lower().endswith(".jp2")]
    pages.sort(key=lambda x: int(re.search(r"(\d+)\.jp2", x).group(1)))
    
    for i, page in enumerate(tqdm(pages, desc="Preparing images")):
        img_path = os.path.join(book_path, page)
        img_array = load_jp2(img_path)
        if img_array is None:
            continue
            
        image_url = convert_to_base64(img_array)
        if not image_url:
            continue
            
        output_path = os.path.join(
            output_root,
            os.path.basename(book_path),
            f"{os.path.splitext(page)[0]}.txt"
        )
        
        image_urls.append(image_url)
        file_map[str(i)] = {
            "path": img_path,
            "output_path": output_path,
            "image_url": image_url
        }
        
    return image_urls, file_map
