import os
import re
import json
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple
from difflib import SequenceMatcher
from rapidfuzz import process, fuzz
from tqdm import tqdm
import logging
import html

class WordLevelMoECleaner:
    def __init__(self, config: Dict):
        self.config = config
        self.base_dir = config["base_dir"]
        self.models = config["models"]
        self.output_dir = config["output_dir"]
        self.trust_weights = config.get("trust_weights", {})
        self.min_agreement = config.get("min_agreement", 0.7)
        self.context_window = config.get("context_window", 2)
        self.kraken_model = config.get("kraken_model", "kraken")
        self.preserve_formatting = config.get("preserve_formatting", True)

        # Setup logging
        logging.basicConfig(
            filename=os.path.join(self.output_dir, "word_level_moe_cleaner.log"),
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s"
        )
        self.logger = logging.getLogger("WordLevelMoECleaner")

        # Set default weights if not provided
        if not self.trust_weights:
            self.trust_weights = {model: 1.0 for model in self.models}
            self.logger.info("Using equal weights for all models")

        # Markdown/HTML pattern detection
        self.markdown_patterns = [
            r'<sup>(.*?)</sup>',  # Superscript
            r'<sub>(.*?)</sub>',  # Subscript
            r'<b>(.*?)</b>',      # Bold
            r'<i>(.*?)</i>',      # Italic
            r'`(.*?)`',           # Code/monospace
            r'\*\*(.*?)\*\*',     # Markdown bold
            r'\*(.*?)\*',         # Markdown italic
        ]

    def _clean_markdown_artifacts(self, text: str) -> str:
        """
        Clean markdown/HTML artifacts from text while preserving meaningful content
        """
        # First, decode HTML entities
        text = html.unescape(text)

        # Handle specific markdown patterns
        for pattern in self.markdown_patterns:
            # Extract content inside markdown but keep it as plain text
            text = re.sub(pattern, r'\1', text)

        # Remove any remaining HTML tags
        text = re.sub(r'<[^>]+>', '', text)

        # Fix common OCR artifacts that might look like markdown
        text = re.sub(r'\\[a-zA-Z]+\{.*?\}', '', text)  # LaTeX-like commands
        text = re.sub(r'&[a-zA-Z]+;', '', text)         # HTML entities

        return text.strip()

    def _extract_formatting_elements(self, text: str) -> List[Dict]:
        """
        Extract formatting elements from text before cleaning
        Returns a list of formatting elements with their positions
        """
        formatting_elements = []

        # Find all markdown/HTML patterns
        for pattern in self.markdown_patterns:
            for match in re.finditer(pattern, text):
                element = {
                    'type': pattern,
                    'content': match.group(1),
                    'start': match.start(),
                    'end': match.end(),
                    'full_match': match.group(0)
                }
                formatting_elements.append(element)

        return formatting_elements
    def _apply_contextual_correction(self, word: str, context: List[str], book_type: str) -> str:
        """Apply contextual correction based on surrounding words and book type"""
        if book_type == "log":
            # Maritime log specific corrections
            maritime_corrections = {
                "bâbord": "bâbord", "tribord": "tribord", "nord": "nord",
                "sud": "sud", "est": "est", "ouest": "ouest", "navire": "navire",
                "cap": "cap", "vent": "vent", "mer": "mer", "voile": "voile"
            }

            if word.lower() in maritime_corrections:
                return maritime_corrections[word.lower()]

        elif book_type == "meteo":
            # Meteorological specific corrections
            meteo_corrections = {
                "température": "température", "baromètre": "baromètre",
                "hygromètre": "hygromètre", "pluie": "pluie", "neige": "neige",
                "vent": "vent", "ciel": "ciel", "nuage": "nuage", "soleil": "soleil"
            }

            if word.lower() in meteo_corrections:
                return meteo_corrections[word.lower()]

        # Use Kraken's output for character-level accuracy if available
        # kraken_word = None
        # for model in self.models:
        #     if self.kraken_model in model:
        #         # In a real implementation, we'd have access to Kraken's output
        #         # For now, we'll simulate this by preferring Kraken for character accuracy
        #         kraken_word = word  # Placeholder
        #         break
        #
        # if kraken_word and len(kraken_word) > 3:
        #     # If Kraken has a suggestion and it's a substantial word, consider it
        #     return kraken_word

        return word
    def _restore_formatting(self, clean_text: str, original_text: str, formatting_elements: List[Dict]) -> str:
        """
        Restore meaningful formatting to cleaned text based on original formatting elements
        """
        if not self.preserve_formatting or not formatting_elements:
            return clean_text

        # For now, we'll handle superscripts which are common in historical documents
        # We'll represent them with Unicode superscript characters where possible
        superscript_map = {
            '0': '⁰', '1': '¹', '2': '²', '3': '³', '4': '⁴',
            '5': '⁵', '6': '⁶', '7': '⁷', '8': '⁸', '9': '⁹',
            'a': 'ᵃ', 'b': 'ᵇ', 'c': 'ᶜ', 'd': 'ᵈ', 'e': 'ᵉ', 'f': 'ᶠ',
            'g': 'ᵍ', 'h': 'ʰ', 'i': 'ⁱ', 'j': 'ʲ', 'k': 'ᵏ', 'l': 'ˡ',
            'm': 'ᵐ', 'n': 'ⁿ', 'o': 'ᵒ', 'p': 'ᵖ', 'r': 'ʳ', 's': 'ˢ',
            't': 'ᵗ', 'u': 'ᵘ', 'v': 'ᵛ', 'w': 'ʷ', 'x': 'ˣ', 'y': 'ʸ', 'z': 'ᶻ',
            '+': '⁺', '-': '⁻', '=': '⁼', '(': '⁽', ')': '⁾'
        }

        # Process each formatting element
        for element in formatting_elements:
            if '<sup>' in element['type']:
                # Handle superscripts
                superscript_content = element['content']
                # Convert to Unicode superscript if possible
                superscript_unicode = ''.join(superscript_map.get(c, c) for c in superscript_content)

                # Try to find the position in clean text
                # This is a simplified approach - in a real implementation, we'd need
                # to track positions more carefully during the cleaning process
                clean_pos = clean_text.find(superscript_content)
                if clean_pos != -1:
                    # Replace with superscript version
                    clean_text = clean_text[:clean_pos] + superscript_unicode + clean_text[clean_pos + len(superscript_content):]

        return clean_text

    def _load_page(self, book_id: str, page_id: str) -> Dict[str, Tuple[str, List[Dict]]]:
        """
        Load OCR results for a page from all models, along with formatting information
        Returns a dict with both cleaned text and formatting elements
        """
        page_texts = {}
        for model in self.models:
            page_path = os.path.join(self.base_dir, model, book_id, f"{page_id}.txt")
            try:
                with open(page_path, "r", encoding="utf-8") as f:
                    original_text = f.read().strip()

                    # Extract formatting elements before cleaning
                    formatting_elements = self._extract_formatting_elements(original_text)

                    # Clean markdown artifacts
                    cleaned_text = self._clean_markdown_artifacts(original_text)

                    page_texts[model] = (cleaned_text, formatting_elements)
            except FileNotFoundError:
                self.logger.warning(f"Missing page {page_id} for model {model} in book {book_id}")
                page_texts[model] = ("", [])
        return page_texts

    def _tokenize_text(self, text: str) -> List[Dict]:
        """Tokenize text into words with position information"""
        # Use regex to split while keeping punctuation with words
        tokens = []
        pattern = re.compile(r'(\w+|\S)')

        for match in pattern.finditer(text):
            token = match.group()
            tokens.append({
                "text": token,
                "start": match.start(),
                "end": match.end(),
                "is_word": bool(re.match(r'\w', token))
            })

        return tokens

    def _align_words_across_models(self, model_texts: Dict[str, Tuple[str, List[Dict]]]) -> List[Dict]:
        """Align words across different model outputs using multiple sequence alignment"""
        # Extract just the cleaned text for alignment
        cleaned_texts = {model: text for model, (text, _) in model_texts.items()}

        # Tokenize all model outputs
        tokenized_models = {}
        for model, (text, _) in model_texts.items():
            tokenized_models[model] = self._tokenize_text(text)

        # Use the model with highest trust weight as reference
        ref_model = max(self.trust_weights, key=self.trust_weights.get)
        ref_tokens = tokenized_models[ref_model]

        # Create alignment matrix
        aligned_words = []

        # For each token in reference model, find corresponding tokens in other models
        for ref_idx, ref_token in enumerate(ref_tokens):
            if not ref_token["is_word"]:
                # Skip non-word tokens for alignment
                continue

            word_info = {
                "position": ref_idx,
                "ref_text": ref_token["text"],
                "models": {},
                "scores": {},
                "formatting": {}  # Store formatting information for each model
            }

            # Find matches in other models
            for model, (text, formatting_elements) in model_texts.items():
                if model == ref_model:
                    word_info["models"][model] = ref_token["text"]
                    word_info["scores"][model] = 1.0
                    # Check if this word had formatting in the original
                    word_start = ref_token["start"]
                    word_end = ref_token["end"]
                    for fmt in formatting_elements:
                        if fmt["start"] <= word_start and fmt["end"] >= word_end:
                            word_info["formatting"][model] = fmt
                    continue

                # Find best match using fuzzy matching
                tokens = tokenized_models[model]
                best_match = None
                best_score = 0
                best_idx = -1

                # Search in a window around the reference position
                search_start = max(0, ref_idx - 5)
                search_end = min(len(tokens), ref_idx + 5)

                for i in range(search_start, search_end):
                    if i < len(tokens) and tokens[i]["is_word"]:
                        score = fuzz.ratio(ref_token["text"].lower(), tokens[i]["text"].lower()) / 100
                        if score > best_score and score > 0.6:  # Minimum similarity threshold
                            best_score = score
                            best_match = tokens[i]["text"]
                            best_idx = i

                if best_match:
                    word_info["models"][model] = best_match
                    word_info["scores"][model] = best_score

                    # Check if this word had formatting in the original
                    if best_idx != -1:
                        word_start = tokens[best_idx]["start"]
                        word_end = tokens[best_idx]["end"]
                        for fmt in model_texts[model][1]:  # formatting_elements
                            if fmt["start"] <= word_start and fmt["end"] >= word_end:
                                word_info["formatting"][model] = fmt

            aligned_words.append(word_info)

        return aligned_words

    def _calculate_word_confidence(self, word_info: Dict) -> Tuple[str, float, Dict]:
        """Calculate confidence for a word based on model agreements"""
        model_votes = defaultdict(float)
        formatting_votes = defaultdict(int)

        for model, word in word_info["models"].items():
            # Weight by model trust and similarity score
            weight = self.trust_weights.get(model, 1.0) * word_info["scores"].get(model, 0.5)
            model_votes[word] += weight

            # Count formatting votes
            if model in word_info["formatting"]:
                fmt_type = word_info["formatting"][model]["type"]
                formatting_votes[fmt_type] += 1

        if not model_votes:
            return word_info["ref_text"], 0.0, {}

        # Find the word with highest weighted votes
        best_word = max(model_votes.items(), key=lambda x: x[1])[0]
        total_weight = sum(model_votes.values())
        confidence = model_votes[best_word] / total_weight if total_weight > 0 else 0

        # Determine the most common formatting
        formatting_decision = {}
        if formatting_votes:
            most_common_fmt = max(formatting_votes.items(), key=lambda x: x[1])
            if most_common_fmt[1] > len(self.models) / 2:  # Majority vote
                formatting_decision = {"type": most_common_fmt[0]}

        return best_word, confidence, formatting_decision

    def _apply_formatting(self, word: str, formatting: Dict) -> str:
        """Apply formatting to a word based on formatting decision"""
        if not formatting:
            return word

        fmt_type = formatting.get("type", "")

        # Handle different formatting types
        if "sup" in fmt_type:
            # Convert to Unicode superscript
            superscript_map = {
                '0': '⁰', '1': '¹', '2': '²', '3': '³', '4': '⁴',
                '5': '⁵', '6': '⁶', '7': '⁷', '8': '⁸', '9': '⁹',
                'a': 'ᵃ', 'b': 'ᵇ', 'c': 'ᶜ', 'd': 'ᵈ', 'e': 'ᵉ', 'f': 'ᶠ',
                'g': 'ᵍ', 'h': 'ʰ', 'i': 'ⁱ', 'j': 'ʲ', 'k': 'ᵏ', 'l': 'ˡ',
                'm': 'ᵐ', 'n': 'ⁿ', 'o': 'ᵒ', 'p': 'ᵖ', 'r': 'ʳ', 's': 'ˢ',
                't': 'ᵗ', 'u': 'ᵘ', 'v': 'ᵛ', 'w': 'ʷ', 'x': 'ˣ', 'y': 'ʸ', 'z': 'ᶻ',
                '+': '⁺', '-': '⁻', '=': '⁼', '(': '⁽', ')': '⁾'
            }
            return ''.join(superscript_map.get(c, c) for c in word)

        elif "sub" in fmt_type:
            # Convert to Unicode subscript
            subscript_map = {
                '0': '₀', '1': '₁', '2': '₂', '3': '₃', '4': '₄',
                '5': '₅', '6': '₆', '7': '₇', '8': '₈', '9': '₉',
                'a': 'ₐ', 'e': 'ₑ', 'h': 'ₕ', 'i': 'ᵢ', 'k': 'ₖ', 'l': 'ₗ',
                'm': 'ₘ', 'n': 'ₙ', 'o': 'ₒ', 'p': 'ₚ', 'r': 'ᵣ', 's': 'ₛ',
                't': 'ₜ', 'u': 'ᵤ', 'v': 'ᵥ', 'x': 'ₓ'
            }
            return ''.join(subscript_map.get(c, c) for c in word)

        # For other formatting types, we might use markdown in the output
        elif "b" in fmt_type or "strong" in fmt_type:
            return f"**{word}**"
        elif "i" in fmt_type or "em" in fmt_type:
            return f"*{word}*"

        return word

    def _combine_at_word_level(self, model_texts: Dict[str, Tuple[str, List[Dict]]],
                               prev_pages: List[str], next_pages: List[str],
                               book_type: str) -> str:
        """Combine OCR results at word level with contextual awareness"""
        # Align words across models
        aligned_words = self._align_words_across_models(model_texts)

        # Build context from previous and next pages
        context = " ".join(prev_pages[-self.context_window:] + next_pages[:self.context_window])
        context_words = re.findall(r'\b\w+\b', context.lower())

        # Process each aligned word
        final_words = []
        confidence_scores = []

        for word_info in aligned_words:
            # Calculate the best word based on model agreement
            best_word, confidence, formatting = self._calculate_word_confidence(word_info)

            # Apply contextual correction if confidence is low
            if confidence < self.min_agreement:
                best_word = self._apply_contextual_correction(best_word, context_words, book_type)

            # Apply formatting if needed
            if formatting and self.preserve_formatting:
                best_word = self._apply_formatting(best_word, formatting)

            final_words.append(best_word)
            confidence_scores.append(confidence)

        # Reconstruct the text with proper spacing
        # For simplicity, we'll just join with spaces
        # In a more advanced implementation, we'd preserve original spacing
        return " ".join(final_words)

    def process_book(self, book_id: str):
        """Process all pages in a book with word-level combination"""
        book_type = self._detect_book_type(book_id)
        self.logger.info(f"Processing book: {book_id} | Type: {book_type}")

        # Get page list from first model
        model_dir = os.path.join(self.base_dir, self.models[0], book_id)
        if not os.path.exists(model_dir):
            self.logger.error(f"Book directory not found: {model_dir}")
            return 0

        # Get sorted page list
        page_files = sorted(
            [f for f in os.listdir(model_dir) if f.endswith(".txt")],
            key=lambda x: self._extract_page_number(x)
        )

        output_book_dir = os.path.join(self.output_dir, book_id)
        os.makedirs(output_book_dir, exist_ok=True)

        processed = 0

        for idx, page_file in enumerate(tqdm(page_files, desc=f"Processing {book_id}")):
            page_id = os.path.splitext(page_file)[0]

            # Load context pages (2 before and 2 after)
            prev_pages, next_pages = self._load_context_pages(book_id, page_files, idx)

            # Load candidate OCR results with formatting information
            candidates = self._load_page(book_id, page_id)
            if not any(text for text, _ in candidates.values()):
                self.logger.warning(f"Skipping empty page: {page_id}")
                continue

            # Combine at word level
            combined_text = self._combine_at_word_level(candidates, prev_pages, next_pages, book_type)

            # Save combined result
            output_path = os.path.join(output_book_dir, f"{page_id}.txt")
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(combined_text)

            processed += 1

        return processed
    def _extract_page_number(self, filename: str) -> int:
        """Extract page number from filename"""
        match = re.search(r'(\d+)\.txt$', filename)
        if match:
            return int(match.group(1))

        numbers = re.findall(r'\d+', filename)
        if numbers:
            return int(numbers[-1])

        return 0

    def _detect_book_type(self, book_id: str) -> str:
        """Detect book type from its ID"""
        book_id_lower = book_id.lower()
        if any(kw in book_id_lower for kw in ["journal", "log", "bord", "voyage", "ship"]):
            return "log"
        elif any(kw in book_id_lower for kw in ["meteo", "table", "weather", "climat", "registre"]):
            return "meteo"
        return "log"  # Default to log

    def _load_context_pages(self, book_id: str, page_files: List[str], current_idx: int) -> Tuple[List[str], List[str]]:
        """Load context pages (2 before and 2 after current page)"""
        prev_pages = []
        next_pages = []

        # Load previous pages (up to 2)
        for i in range(max(0, current_idx - 2), current_idx):
            if i < len(page_files):
                page_id = os.path.splitext(page_files[i])[0]
                # Use the first model's version for context
                context_path = os.path.join(self.base_dir, self.models[0], book_id, f"{page_id}.txt")
                try:
                    with open(context_path, "r", encoding="utf-8") as f:
                        prev_pages.append(f.read().strip())
                except FileNotFoundError:
                    self.logger.warning(f"Missing context page: {context_path}")

        # Load next pages (up to 2)
        for i in range(current_idx + 1, min(current_idx + 3, len(page_files))):
            if i < len(page_files):
                page_id = os.path.splitext(page_files[i])[0]
                # Use the first model's version for context
                context_path = os.path.join(self.base_dir, self.models[0], book_id, f"{page_id}.txt")
                try:
                    with open(context_path, "r", encoding="utf-8") as f:
                        next_pages.append(f.read().strip())
                except FileNotFoundError:
                    self.logger.warning(f"Missing context page: {context_path}")

        return prev_pages, next_pages

    def process_library(self, book_ids: List[str] = None):
        """Process multiple books in the library"""
        if book_ids is None:
            model_dir = os.path.join(self.base_dir, self.models[0])
            book_ids = [d for d in os.listdir(model_dir)
                        if os.path.isdir(os.path.join(model_dir, d))]

        total = 0
        for book_id in book_ids:
            page_count = self.process_book(book_id)
            total += page_count
            self.logger.info(f"Processed {book_id} with {page_count} pages")

        self.logger.info(f"Finished! Total pages processed: {total}")
        return total