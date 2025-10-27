from ocr_processing.providers import MistralOCRProvider, OpenRouterOCRProvider, LocalOCRProvider, QwenOCRProvider, KrakenOCRProvider
from ocr_processing.utils import image_processing
from typing import Dict
import os


class OCRManager:
    def __init__(self, config: Dict):
        """
        Initialize OCR processing system with configuration
        
        Args:
            config: Dictionary with OCR settings
        """
        self.config = config
        self.provider = self._init_provider(config)
        self.book_type = "log"
        self.prompt_version = "v2"
    
    def _init_provider(self, config: Dict):
        provider = config.get('api_provider', 'mistral')
        if provider == "mistral":
            return MistralOCRProvider(config)
        elif provider == "openrouter":
            return OpenRouterOCRProvider(config)
        elif provider == "local":
            return LocalOCRProvider(config)
        elif provider == "qwen":
            return QwenOCRProvider(config)
        elif provider == "kraken":
            return KrakenOCRProvider(config)
        raise ValueError(f"Unsupported provider: {provider}")
    
    def detect_book_type(self, book_path: str) -> str:
        """Detect book type based on folder name patterns"""
        book_name = os.path.basename(book_path).lower()
        meteo_keywords = ["meteo", "table", "weather", "climat", "registre", "tableau"]
        log_keywords = ["journal", "log", "bord", "voyage", "ship"]
        
        if any(kw in book_name for kw in meteo_keywords):
            return "meteo"
        if any(kw in book_name for kw in log_keywords):
            return "log"
        return "log"

    def set_prompt_strategy(self, book_type: str = "log", prompt_version: str = "v2"):
        """Set prompt strategy based on document type"""
        self.book_type = book_type
        self.prompt_version = prompt_version

    def get_prompt(self) -> str:
        """Get specialized prompt based on document type and version"""
        prompts = {
            "log": {
                "v1": (
                    "Extrayez TOUT le texte de ce journal de bord historique EXACTEMENT comme il apparaît. "
                    "Préservez la ponctuation, les chiffres et la mise en forme originale. "
                    "Ne corrigez pas le texte. Le document est en français avec des termes maritimes historiques. "
                    "Incluez toutes les notes marginales et annotations. "
                    "Ne retournez que le contenu, sans commentaire ni explication. "
                    "Si le document est vide ou ne contient pas de texte, retournez une chaîne vide \"\"."
                ),
                "v2": (
                    "Transcrivez fidèlement le contenu de ce journal de navigation historique. "
                    "Conservez la disposition originale, y compris les sauts de ligne et les alinéas. "
                    "Normalisez les formats de date ou d'heure. Les termes techniques maritimes doivent "
                    "être transcrits tels quels. Le texte est intégralement en français ancien. "
                    "Renvoyez uniquement le contenu, sans commentaire ni explication. "
                    "Si le document est vide ou ne contient pas de texte, retournez \"\"."
                )
            },
            "meteo": {
                "v1": (
                    "Extrayez TOUT le texte de ce registre météorologique historique EXACTEMENT comme il apparaît. "
                    "Préservez la disposition tabulaire, les en-têtes de colonnes et les valeurs numériques. "
                    "Ne modifiez aucun symbole ou unité de mesure. Le document est en français avec des "
                    "notations scientifiques historiques. Incluez les en-têtes et pieds de page. "
                    "Ne retournez que le contenu, sans commentaire ni explication. "
                    "Si le document est vide ou ne contient pas de texte, retournez \"\"."
                ),
                "v2": (
                    "Transcrivez méticuleusement ce registre d'observations météorologiques historiques. "
                    "Maintenez la structure tabulaire avec les colonnes et lignes visibles. "
                    "Conservez toutes les valeurs numériques, unités de mesure et symboles. "
                    "Normalisez les formats de date ou d'heure. Le texte est en français technique ancien. "
                    "Renvoyez uniquement le contenu, sans commentaire ni explication. "
                    "Si le document est vide ou ne contient pas de texte, retournez \"\"."
                )
            }
        }
        return prompts.get(self.book_type, {}).get(self.prompt_version, prompts["log"]["v1"])

    def process_image(self, image_path: str, output_path: str) -> bool:
        """
        Process a single image file
        
        Args:
            image_path: Path to input image (JP2 format)
            output_path: Path to save OCR results
            
        Returns:
            True if successful, False otherwise
        """
        img_array = image_processing.load_jp2(image_path)
        if img_array is None:
            return False
        image_url = image_processing.convert_to_base64(img_array)
        if not image_url:
            return False
        return self.provider.process_image(
            image_url, 
            output_path, 
            self.get_prompt()
        )

    def process_book(self, book_path: str, output_root: str) -> int:
        """
        Process all images in a book directory
        
        Args:
            book_path: Path to book directory with JP2 images
            output_root: Root directory to save OCR results
            
        Returns:
            Number of successfully processed pages
        """
        # Auto-detect book type
        book_type = self.detect_book_type(book_path)
        self.set_prompt_strategy(book_type, self.config.get('prompt_version', 'v2'))
        
        return self.provider.process_book(
            book_path, 
            output_root, 
            self.get_prompt()
        )

