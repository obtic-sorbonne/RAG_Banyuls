# ocr_processing/providers/kraken_ocr.py
import os
import re
import subprocess
import tempfile
import json
from typing import Dict, List
from tqdm import tqdm
import logging

class KrakenOCRProvider:
    def __init__(self, config: Dict):
        self.config = config
        self.model_path = config.get('kraken_model_path')
        self.segment_path = config.get('kraken_segment_path')
        self.binarize = config.get('kraken_binarize', True)
        self.segment = config.get('kraken_segment', True)
        self.timeout = config.get('kraken_timeout', 300)
        self.max_retries = config.get('max_retries', 3)

        # Setup logging
        self.logger = logging.getLogger("KrakenOCR")

        # Validate Kraken installation
        # self._validate_installation()

    def _validate_installation(self):
        """Check if Kraken is properly installed"""
        try:
            result = subprocess.run(['kraken', '--version'],
                                    capture_output=True, text=True, timeout=10)
            if result.returncode != 0:
                raise ImportError("Kraken OCR not found. Please install it from: https://github.com/mittagessen/kraken")
        except (subprocess.TimeoutExpired, FileNotFoundError):
            raise ImportError("Kraken OCR not found. Please install it from: https://github.com/mittagessen/kraken")

    def _run_kraken_command(self, image_path: str, output_path: str) -> bool:
        """Execute Kraken OCR command on a single image"""
        cmd = ['kraken', '-i', image_path, output_path]

        if self.binarize:
            cmd.extend(['binarize'])
        if self.segment:
            cmd.extend(['segment', '-bl', '-i', self.segment_path])
        if self.model_path:
            cmd.extend(['ocr', '-m', self.model_path])
        else:
            cmd.extend(['ocr'])

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=self.timeout)
            if result.returncode != 0:
                self.logger.error(f"Kraken OCR failed: {result.stderr}")
                return False
            return True
        except subprocess.TimeoutExpired:
            self.logger.error(f"Kraken OCR timed out for {image_path}")
            return False
        except Exception as e:
            self.logger.error(f"Kraken OCR error: {str(e)}")
            return False

    def process_image(self, image_url: str, output_path: str, prompt_text: str = None) -> bool:
        """
        Process a single image using Kraken OCR

        Args:
            image_url: Can be local file path or base64 encoded image
            output_path: Path to save OCR results
            prompt_text: Not used for Kraken, maintained for interface consistency
        """
        # Kraken works with local files, so we need to handle different input types
        if image_url.startswith('data:image'):
            # Handle base64 images - save to temporary file
            import base64
            from ocr_processing.utils import image_processing

            # Extract base64 data and save to temp file
            header, data = image_url.split(',', 1)
            img_format = header.split('/')[1].split(';')[0]

            with tempfile.NamedTemporaryFile(suffix=f'.{img_format}', delete=False) as temp_file:
                temp_file.write(base64.b64decode(data))
                temp_image_path = temp_file.name

            try:
                success = self._run_kraken_command(temp_image_path, output_path)
                os.unlink(temp_image_path)  # Clean up temp file
                return success
            except:
                os.unlink(temp_image_path)
                return False
        else:
            # Assume it's a local file path
            return self._run_kraken_command(image_url, output_path)

    def process_book(self, book_path: str, output_root: str, prompt_text: str = None) -> int:
        """
        Process all images in a book directory using Kraken OCR

        Args:
            book_path: Path to directory containing images
            output_root: Root directory for output files
            prompt_text: Not used for Kraken, maintained for interface consistency
        """
        # Get image files from the book directory
        image_extensions = ['jp2', '.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp']
        image_files = []

        for file in os.listdir(book_path):
            if any(file.lower().endswith(ext) for ext in image_extensions):
                image_files.append(file)

        if not image_files:
            self.logger.warning(f"No image files found in {book_path}")
            return 0

        # Sort files naturally
        image_files.sort(key=lambda x: [int(c) if c.isdigit() else c for c in re.split(r'(\d+)', x)])

        processed = 0
        for image_file in tqdm(image_files, desc="Processing pages with Kraken"):
            image_path = os.path.join(book_path, image_file)

            # Create output path
            base_name = os.path.splitext(image_file)[0]
            output_path = os.path.join(output_root, os.path.basename(book_path), f"{base_name}.txt")
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            # Process image
            success = self.process_image(image_path, output_path)
            if success:
                processed += 1
            else:
                self.logger.error(f"Failed to process {image_file}")

        return processed

    def test_connection(self) -> bool:
        """Test if Kraken OCR is working properly"""
        try:
            # Create a simple test image
            from PIL import Image, ImageDraw, ImageFont
            import tempfile

            # Create a test image with text
            img = Image.new('RGB', (200, 50), color='white')
            d = ImageDraw.Draw(img)

            try:
                font = ImageFont.truetype("Arial", 15)
            except:
                font = ImageFont.load_default()

            d.text((10, 10), "Kraken OCR Test", fill='black', font=font)

            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_img:
                img.save(temp_img.name)
                temp_img_path = temp_img.name

            with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as temp_output:
                temp_output_path = temp_output.name

            # Test OCR
            success = self._run_kraken_command(temp_img_path, temp_output_path)

            # Clean up
            os.unlink(temp_img_path)
            if os.path.exists(temp_output_path):
                with open(temp_output_path, 'r') as f:
                    result = f.read().strip()
                os.unlink(temp_output_path)

                if "Kraken OCR Test" in result:
                    self.logger.info("✓ Kraken OCR connection successful!")
                    return True

            self.logger.error("✗ Kraken OCR test failed")
            return False

        except Exception as e:
            self.logger.error(f"✗ Kraken OCR test failed: {str(e)}")
            return False

    def get_available_models(self) -> List[str]:
        """Return list of available Kraken models"""
        # This would typically check a model directory
        models_dir = self.config.get('kraken_models_dir', 'data/kraken/models')
        models_dir = os.path.expanduser(models_dir)

        if not os.path.exists(models_dir):
            return []

        models = []
        for file in os.listdir(models_dir):
            if file.endswith('.mlmodel'):
                models.append(file)

        return models

    def set_model(self, model_name: str):
        """Set the model to use"""
        models_dir = self.config.get('kraken_models_dir', 'data/kraken/models')
        models_dir = os.path.expanduser(models_dir)
        model_path = os.path.join(models_dir, model_name)

        if os.path.exists(model_path):
            self.model_path = model_path
            self.logger.info(f"Model set to: {model_name}")
        else:
            available_models = self.get_available_models()
            self.logger.error(f"Model {model_name} not found. Available models: {available_models}")
            raise ValueError(f"Model {model_name} not found")