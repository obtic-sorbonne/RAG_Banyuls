import os
import re
import json
import yaml
from .chunking_strategies import PageChunker, RecordChunker
from .vector_store_manager import VectorStoreManager

class CorpusProcessor:
    def __init__(self, config_path="config/corpus_settings.yaml"):
        self.config = self.load_config(config_path)
        self.vector_store = VectorStoreManager(self.config.get('vector'))
    
    def load_config(self, path):
        with open(path, 'r') as f:
            return yaml.safe_load(f)
    
    def process_directory(self, input_dir, output_dir):
        """Process OCR results from Mistral API output"""
        os.makedirs(output_dir, exist_ok=True)
        
        for book_name in os.listdir(input_dir):
            book_path = os.path.join(input_dir, book_name)
            if os.path.isdir(book_path):
                self.process_book(book_path, output_dir)
    
    def process_book(self, book_path, output_dir):
        """Process all pages in a book"""
        book_name = os.path.basename(book_path)
        metadata = self.load_or_extract_metadata(book_path, book_name)

        processed_data = {
            "book_name": book_name,
            "pages": [],
            "metadata": metadata
        }
        
        page_files = sorted(
            [f for f in os.listdir(book_path) if f.endswith('.txt')],
            key=lambda x: int(x.split('_')[-1].split('.')[0])
        )
        for page_file in page_files:
            page_path = os.path.join(book_path, page_file)
            page_data = self.process_page(page_path)
            processed_data["pages"].append(page_data)
        
        # Extract years from entities if metadata not found
        if not processed_data["metadata"]["years"]:
            processed_data["metadata"]["years"] = self.extract_years_from_entities(
                processed_data["pages"]
            )

        # Save processed book data
        output_path = os.path.join(output_dir, f"{book_name}.json")
        with open(output_path, 'w') as f:
            json.dump(processed_data, f, indent=2)
        
        # Index to vector store
        self.vector_store.index_book(processed_data)
    
    def extract_years_from_entities(self, pages):
        """Extract unique years from page entities"""
        years = set()
        for page in pages:
            for date_str in page["entities"]["dates"]:
                # Extract 4-digit years using more robust pattern
                year_matches = re.findall(r'\b(1[6-9]\d{2}|20[0-2]\d)\b', date_str)
                years.update(year_matches)
        return sorted(list(years))
    
    def load_or_extract_metadata(self, book_path, book_name):
        """Extract book-level metadata support metadata load"""
        metadata = {"years": [], "book_name": book_name}
        
        # 1. Check for metadata.json
        metadata_path = os.path.join(book_path, "metadata.json")
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'r') as f:
                    external_meta = json.load(f)
                if "years" in external_meta:
                    metadata["years"] = [str(y) for y in external_meta["years"]]
                if "book_name" in external_meta:
                    metadata["book_name"] = external_meta["book_name"]

            except json.JSONDecodeError:
                print(f"Invalid JSON in {metadata_path}")
        
        return metadata
    
    def extract_metadata(self, book_path):
        """Extract book-level metadata"""
        metadata = {
            "year": None,
            "vessel": None,
            "captain": None
        }
        
        # Extract from first 3 pages
        for i in range(1, 4):
            page_path = os.path.join(book_path, f"{os.path.basename(book_path)}_{i:03d}.txt")
            if os.path.exists(page_path):
                with open(page_path, 'r') as f:
                    content = f.read()
                    if not metadata["year"]:
                        year_match = re.search(r"\b(18\d{2}|19\d{2})\b", content)
                        if year_match: metadata["year"] = year_match.group(0)
                    if not metadata["vessel"]:
                        vessel_match = re.search(r"Navire:\s*(.*)|Vessel:\s*(.*)", content)
                        if vessel_match: metadata["vessel"] = vessel_match.group(1) or vessel_match.group(2)
                    if not metadata["captain"]:
                        captain_match = re.search(r"Signature:\s*(.*)|Capitaine:\s*(.*)", content)
                        if captain_match: metadata["captain"] = captain_match.group(1) or captain_match.group(2)
        return metadata
    
    def process_page(self, page_path):
        """Process individual page"""
        with open(page_path, 'r') as f:
            content = f.read()
        
        return {
            "page_number": os.path.basename(page_path).split('_')[-1].split('.')[0],
            "raw_content": content,
            "cleaned_content": self.clean_content(content),
            "entities": self.extract_entities(content)
        }
    
    def clean_content(self, text):
        """Basic cleaning for OCR output"""
        # Remove header/footer artifacts
        text = re.sub(r'Page\s*\d+', '', text)
        text = re.sub(r'- \d+ -', '', text)
        # Fix common OCR errors
        text = re.sub(r'([a-z])\s+([a-z])', r'\1\2', text)  # Join split words
        return text.strip()
    
    def extract_entities(self, text):
        """Basic entity extraction (placeholder for Team 3)"""
        return {
            "dates": re.findall(r'\d{1,2}\s+[a-zA-Z]+\s+\d{4}', text),
            # "measurements": re.findall(r'\d+\.\d+\s*°?[CF]|\d+\s*knots', text),
			# "locations": re.findall(r'Lat:\s*[\d°\'"]+\s*[NS],?\s*Lon:\s*[\d°\'"]+\s*[WE]', text)
        }
