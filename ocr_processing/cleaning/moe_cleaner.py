# ocr_processing/cleaning/moe_cleaner.py
import os
import json
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple
from difflib import SequenceMatcher
from tqdm import tqdm
import logging


class MoECleaner:
    def __init__(self, config: Dict):
        self.config = config
        self.base_dir = config["base_dir"]
        self.models = config["models"]
        self.output_dir = config["output_dir"]
        self.trust_weights = config.get("trust_weights", {})
        self.min_agreement = config.get("min_agreement", 0.7)
        self.context_window = config.get("context_window", 3)  # Previous pages to consider

        # Setup logging
        self.logger = logging.getLogger("MoECleaner")

        # Set default weights if not provided
        if not self.trust_weights:
            self.trust_weights = {model: 1.0 for model in self.models}
            self.logger.info("Using equal weights for all models")

    def _load_page(self, book_id: str, page_id: str) -> Dict[str, str]:
        """Load OCR results for a page from all models"""
        page_texts = {}
        for model in self.models:
            page_path = os.path.join(self.base_dir, model, book_id, f"{page_id}.txt")
            try:
                with open(page_path, "r", encoding="utf-8") as f:
                    page_texts[model] = f.read().strip()
            except FileNotFoundError:
                self.logger.warning(f"Missing page {page_id} for model {model} in book {book_id}")
                page_texts[model] = ""
        return page_texts

    def _temporal_consistency_check(self, current_text: str, previous_texts: List[str]) -> float:
        """Calculate consistency score with previous pages"""
        if not previous_texts:
            return 1.0  # No previous context

        similarities = []
        current_lines = current_text.split('\n')[-self.context_window:]  # Last few lines

        for prev_text in previous_texts:
            prev_lines = prev_text.split('\n')[-self.context_window:]  # Last few lines
            for curr_line in current_lines:
                if not curr_line.strip():
                    continue
                max_sim = max(
                    SequenceMatcher(None, curr_line, prev_line).ratio()
                    for prev_line in prev_lines
                )
                similarities.append(max_sim)

        return np.mean(similarities) if similarities else 1.0

    def _combine_texts(self, page_texts: Dict[str, str], previous_texts: List[str]) -> str:
        """Combine OCR results using MoE strategy with temporal consistency"""
        # Split texts into lines
        all_lines = []
        for model, text in page_texts.items():
            lines = [line.strip() for line in text.split('\n') if line.strip()]
            all_lines.append((model, lines))

        # Create line clusters
        line_clusters = defaultdict(list)
        cluster_map = {}

        # First pass: group identical lines
        for model, lines in all_lines:
            for i, line in enumerate(lines):
                if line not in cluster_map:
                    cluster_map[line] = f"cluster_{len(cluster_map)}"
                line_clusters[cluster_map[line]].append((model, line, i))

        # Second pass: group similar lines (fuzzy matching)
        threshold = 0.85  # Similarity threshold
        clusters = list(line_clusters.keys())
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                cluster_i = next(iter(line_clusters[clusters[i]]))[1]
                cluster_j = next(iter(line_clusters[clusters[j]]))[1]
                if SequenceMatcher(None, cluster_i, cluster_j).ratio() >= threshold:
                    # Merge clusters
                    line_clusters[clusters[i]].extend(line_clusters[clusters[j]])
                    del line_clusters[clusters[j]]
                    break

        # Calculate line scores
        line_scores = {}
        for cluster_id, entries in line_clusters.items():
            model_scores = defaultdict(float)
            for model, line, pos in entries:
                # Base score = model trust weight + position bonus
                score = self.trust_weights.get(model, 1.0) + (1 / (pos + 1))
                model_scores[line] += score

            # Select best line in cluster
            best_line, best_score = "", 0.0
            for line, score in model_scores.items():
                if score > best_score:
                    best_line = line
                    best_score = score
            line_scores[best_line] = best_score

        # Temporal consistency adjustment
        consistency_score = self._temporal_consistency_check(
            "\n".join(line_scores.keys()),
            previous_texts
        )

        # Apply temporal consistency weighting
        for line in line_scores:
            line_scores[line] *= consistency_score

        # Order lines by original positions
        ordered_lines = []
        max_lines = max(len(lines) for _, lines in all_lines)
        for pos in range(max_lines):
            candidates = []
            for model, lines in all_lines:
                if pos < len(lines):
                    line = lines[pos]
                    if line in line_scores:
                        candidates.append((line, line_scores[line]))

            if candidates:
                # Select candidate with highest score
                best_candidate = max(candidates, key=lambda x: x[1])
                ordered_lines.append(best_candidate[0])

        return "\n".join(ordered_lines)

    def process_book(self, book_id: str):
        """Process all pages in a book with temporal consistency"""
        book_path = os.path.join(self.base_dir, self.models[0], book_id)
        if not os.path.exists(book_path):
            self.logger.error(f"Book {book_id} not found in base directory")
            return 0

        # Get sorted page list
        page_files = sorted(
            [f for f in os.listdir(book_path) if f.endswith(".txt")],
            key=lambda x: x.split('.')[0]
        )

        output_book_dir = os.path.join(self.output_dir, book_id)
        os.makedirs(output_book_dir, exist_ok=True)

        previous_texts = []  # Maintain context window
        processed = 0

        for page_file in tqdm(page_files, desc=f"Processing {book_id}"):
            page_id = os.path.splitext(page_file)[0]
            page_texts = self._load_page(book_id, page_id)

            combined_text = self._combine_texts(page_texts, previous_texts)

            # Save combined result
            output_path = os.path.join(output_book_dir, f"{page_id}.txt")
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(combined_text)

            # Update context window
            previous_texts.append(combined_text)
            if len(previous_texts) > self.context_window:
                previous_texts.pop(0)

            processed += 1

        return processed

    def process_library(self, book_ids: List[str] = None):
        """Process multiple books in the library"""
        if book_ids is None:
            # Get all books from first model directory
            book_ids = [
                d for d in os.listdir(os.path.join(self.base_dir, self.models[0]))
                if os.path.isdir(os.path.join(self.base_dir, self.models[0], d))
            ]

        total_pages = 0
        for book_id in book_ids:
            page_count = self.process_book(book_id)
            total_pages += page_count
            self.logger.info(f"Processed {page_count} pages in {book_id}")

        self.logger.info(f"Completed! Total pages processed: {total_pages}")
        return total_pages
