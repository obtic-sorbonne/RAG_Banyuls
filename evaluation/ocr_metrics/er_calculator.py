import numpy as np
from typing import Dict, List, Tuple, Optional
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class OCRMetrics:
    cer: float
    wer: float
    substitutions: int
    deletions: int
    insertions: int
    ref_length: int
    hyp_length: int
    ref_text: str
    hyp_text: str
    file_path: str = ""
    book_name: str = ""
    page_name: str = ""


class CERWERCalculator:
    """
    Calculator for Character Error Rate (CER) and Word Error Rate (WER)
    based on Levenshtein distance algorithm with proper text cleaning
    """

    def __init__(self):
        pass

    def clean_text(self, text: str) -> str:
        """
        Clean text by removing markdown, extra whitespace, and normalizing

        Args:
            text: Input text to clean

        Returns:
            Cleaned text
        """
        if not text:
            return ""

        # Remove markdown code blocks
        text = re.sub(r'```[a-zA-Z]*\s*', '', text)

        # Remove extra whitespace (newlines, tabs, multiple spaces)
        text = re.sub(r'\s+', ' ', text)

        # Strip leading/trailing whitespace
        text = text.strip()

        return text

    def _levenshtein_distance(self, ref: List, hyp: List) -> Tuple[int, int, int, int]:
        """
        Calculate Levenshtein distance and operation counts between two sequences
        with proper operation counting using operation matrix

        Args:
            ref: Reference sequence (characters or words)
            hyp: Hypothesis sequence (OCR output)

        Returns:
            Tuple of (substitutions, deletions, insertions, distance)
        """
        m = len(ref)
        n = len(hyp)

        # Initialize distance matrix
        dp = np.zeros((m + 1, n + 1), dtype=int)

        # Operation tracking matrix: 0=match, 1=substitution, 2=deletion, 3=insertion
        ops = np.zeros((m + 1, n + 1), dtype=int)

        # Initialize first column (all deletions)
        for i in range(m + 1):
            dp[i][0] = i
            ops[i][0] = 2  # deletion

        # Initialize first row (all insertions)
        for j in range(n + 1):
            dp[0][j] = j
            ops[0][j] = 3  # insertion

        ops[0][0] = 0  # start

        # Fill distance and operation matrices
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if ref[i - 1] == hyp[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                    ops[i][j] = 0  # match
                else:
                    substitution_cost = dp[i - 1][j - 1] + 1
                    deletion_cost = dp[i - 1][j] + 1
                    insertion_cost = dp[i][j - 1] + 1

                    min_cost = min(substitution_cost, deletion_cost, insertion_cost)
                    dp[i][j] = min_cost

                    if min_cost == substitution_cost:
                        ops[i][j] = 1  # substitution
                    elif min_cost == deletion_cost:
                        ops[i][j] = 2  # deletion
                    else:
                        ops[i][j] = 3  # insertion

        # Backtrack to count operations
        i, j = m, n
        substitutions = deletions = insertions = 0

        while i > 0 or j > 0:
            if ops[i][j] == 0:  # match
                i -= 1
                j -= 1
            elif ops[i][j] == 1:  # substitution
                substitutions += 1
                i -= 1
                j -= 1
            elif ops[i][j] == 2:  # deletion
                deletions += 1
                i -= 1
            else:  # insertion
                insertions += 1
                j -= 1

        total_distance = substitutions + deletions + insertions

        # Verify the distance matches
        assert total_distance == dp[m][n], f"Operation counting mismatch: {total_distance} vs {dp[m][n]}"

        return substitutions, deletions, insertions, total_distance

    def calculate_cer(self, ref_text: str, hyp_text: str) -> OCRMetrics:
        """
        Calculate Character Error Rate (CER) between reference and hypothesis texts

        Args:
            ref_text: Ground truth reference text
            hyp_text: OCR output text

        Returns:
            OCRMetrics object with CER and detailed statistics
        """
        # Clean texts
        ref_text_clean = self.clean_text(ref_text)
        hyp_text_clean = self.clean_text(hyp_text)

        # Convert to character sequences
        ref_chars = list(ref_text_clean)
        hyp_chars = list(hyp_text_clean)

        # Handle edge cases
        if len(ref_chars) == 0 and len(hyp_chars) == 0:
            # Both empty - perfect match
            return OCRMetrics(
                cer=0.0,
                wer=0.0,
                substitutions=0,
                deletions=0,
                insertions=0,
                ref_length=0,
                hyp_length=0,
                ref_text=ref_text,
                hyp_text=hyp_text
            )
        elif len(ref_chars) == 0:
            # Reference is empty but hypothesis isn't - 100% error
            return OCRMetrics(
                cer=1.0,
                wer=1.0,
                substitutions=0,
                deletions=0,
                insertions=len(hyp_chars),
                ref_length=0,
                hyp_length=len(hyp_chars),
                ref_text=ref_text,
                hyp_text=hyp_text
            )

        # Calculate Levenshtein distance at character level
        substitutions, deletions, insertions, distance = self._levenshtein_distance(ref_chars, hyp_chars)

        # Calculate CER (always between 0-1)
        total_errors = substitutions + deletions + insertions
        cer = min(total_errors / len(ref_chars), 1.0)  # Cap at 100%

        return OCRMetrics(
            cer=cer,
            wer=0.0,  # Will be calculated separately
            substitutions=substitutions,
            deletions=deletions,
            insertions=insertions,
            ref_length=len(ref_chars),
            hyp_length=len(hyp_chars),
            ref_text=ref_text_clean,
            hyp_text=hyp_text_clean
        )

    def calculate_wer(self, ref_text: str, hyp_text: str) -> OCRMetrics:
        """
        Calculate Word Error Rate (WER) between reference and hypothesis texts

        Args:
            ref_text: Ground truth reference text
            hyp_text: OCR output text

        Returns:
            OCRMetrics object with WER and detailed statistics
        """
        # Clean texts
        ref_text_clean = self.clean_text(ref_text)
        hyp_text_clean = self.clean_text(hyp_text)

        # Tokenize into words (split by whitespace)
        ref_words = ref_text_clean.split()
        hyp_words = hyp_text_clean.split()

        # Handle edge cases
        if len(ref_words) == 0 and len(hyp_words) == 0:
            # Both empty - perfect match
            return OCRMetrics(
                cer=0.0,
                wer=0.0,
                substitutions=0,
                deletions=0,
                insertions=0,
                ref_length=0,
                hyp_length=0,
                ref_text=ref_text,
                hyp_text=hyp_text
            )
        elif len(ref_words) == 0:
            # Reference is empty but hypothesis isn't - 100% error
            return OCRMetrics(
                cer=1.0,
                wer=1.0,
                substitutions=0,
                deletions=0,
                insertions=len(hyp_words),
                ref_length=0,
                hyp_length=len(hyp_words),
                ref_text=ref_text,
                hyp_text=hyp_text
            )

        # Calculate Levenshtein distance at word level
        substitutions, deletions, insertions, distance = self._levenshtein_distance(ref_words, hyp_words)

        # Calculate WER (always between 0-1)
        total_errors = substitutions + deletions + insertions
        wer = min(total_errors / len(ref_words), 1.0)  # Cap at 100%

        return OCRMetrics(
            cer=0.0,  # Will be calculated separately
            wer=wer,
            substitutions=substitutions,
            deletions=deletions,
            insertions=insertions,
            ref_length=len(ref_words),
            hyp_length=len(hyp_words),
            ref_text=ref_text_clean,
            hyp_text=hyp_text_clean
        )

    def calculate_all_metrics(self, ref_text: str, hyp_text: str) -> OCRMetrics:
        """
        Calculate both CER and WER for given texts with proper text cleaning

        Args:
            ref_text: Ground truth reference text
            hyp_text: OCR output text

        Returns:
            OCRMetrics object with both CER and WER
        """
        # Clean texts first
        ref_text_clean = self.clean_text(ref_text)
        hyp_text_clean = self.clean_text(hyp_text)

        cer_metrics = self.calculate_cer(ref_text_clean, hyp_text_clean)
        wer_metrics = self.calculate_wer(ref_text_clean, hyp_text_clean)

        return OCRMetrics(
            cer=cer_metrics.cer,
            wer=wer_metrics.wer,
            substitutions=cer_metrics.substitutions,  # Character-level substitutions
            deletions=cer_metrics.deletions,  # Character-level deletions
            insertions=cer_metrics.insertions,  # Character-level insertions
            ref_length=cer_metrics.ref_length,
            hyp_length=cer_metrics.hyp_length,
            ref_text=ref_text_clean,
            hyp_text=hyp_text_clean
        )