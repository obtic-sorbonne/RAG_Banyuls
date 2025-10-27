import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime

from .er_calculator import CERWERCalculator, OCRMetrics

class OCREvaluationManager:
    """
    Manager for OCR evaluation tasks including batch processing and result analysis
    """

    def __init__(self, ground_truth_dir: str = "data/ground_truth",
                 ocr_results_dir: str = "data/ocr_results"):
        self.ground_truth_dir = Path(ground_truth_dir)
        self.ocr_results_dir = Path(ocr_results_dir)
        self.calculator = CERWERCalculator()

        # Ensure directories exist
        self.ground_truth_dir.mkdir(parents=True, exist_ok=True)
        self.ocr_results_dir.mkdir(parents=True, exist_ok=True)

    def find_matching_pairs(self) -> List[Tuple[Path, Path]]:
        """
        Find matching ground truth and OCR result files

        Returns:
            List of tuples (ground_truth_path, ocr_result_path)
        """
        pairs = []

        # Iterate through book directories
        for book_dir in self.ground_truth_dir.iterdir():
            if book_dir.is_dir():
                ocr_book_dir = self.ocr_results_dir / book_dir.name

                if ocr_book_dir.exists():
                    # Find matching text files
                    for gt_file in book_dir.glob("*.txt"):
                        ocr_file = ocr_book_dir / gt_file.name

                        if ocr_file.exists():
                            pairs.append((gt_file, ocr_file))
                        else:
                            print(f"Warning: OCR file not found for {gt_file}")

        # Also check for files in root directories
        for gt_file in self.ground_truth_dir.glob("*.txt"):
            ocr_file = self.ocr_results_dir / gt_file.name
            if ocr_file.exists() and (gt_file, ocr_file) not in pairs:
                pairs.append((gt_file, ocr_file))

        print(f"Found {len(pairs)} matching file pairs")
        return pairs

    def evaluate_single_pair(self, gt_path: Path, ocr_path: Path) -> Optional[OCRMetrics]:
        """
        Evaluate a single ground truth - OCR result pair

        Args:
            gt_path: Path to ground truth file
            ocr_path: Path to OCR result file

        Returns:
            OCRMetrics object or None if evaluation fails
        """
        try:
            # Read files with error handling
            with open(gt_path, 'r', encoding='utf-8', errors='replace') as f:
                ref_text = f.read()

            with open(ocr_path, 'r', encoding='utf-8', errors='replace') as f:
                hyp_text = f.read()

            # Calculate metrics
            metrics = self.calculator.calculate_all_metrics(ref_text, hyp_text)

            # Add file metadata
            metrics.file_path = str(gt_path.relative_to(self.ground_truth_dir))
            metrics.book_name = gt_path.parent.name
            metrics.page_name = gt_path.stem

            print(f"Processed: {metrics.file_path} - CER: {metrics.cer:.4f}, WER: {metrics.wer:.4f}")

            return metrics

        except Exception as e:
            print(f"Error evaluating {gt_path}: {e}")
            return None

    def evaluate_batch(self) -> List[OCRMetrics]:
        """
        Evaluate all matching ground truth - OCR result pairs

        Returns:
            List of OCRMetrics objects for all evaluated pairs
        """
        pairs = self.find_matching_pairs()
        results = []

        print(f"Found {len(pairs)} file pairs to evaluate")

        for i, (gt_path, ocr_path) in enumerate(pairs):
            print(f"Processing {i+1}/{len(pairs)}: {gt_path.name}")
            metrics = self.evaluate_single_pair(gt_path, ocr_path)
            if metrics:
                results.append(metrics)

        print(f"Successfully evaluated {len(results)} out of {len(pairs)} pairs")
        return results

    def _convert_to_serializable(self, obj):
        """
        Convert numpy/pandas types to native Python types for JSON serialization
        """
        if isinstance(obj, (np.integer, pd.Int64Dtype)):
            return int(obj)
        elif isinstance(obj, (np.floating, pd.Float64Dtype)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.Series):
            return obj.to_dict()
        elif isinstance(obj, dict):
            return {k: self._convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_serializable(item) for item in obj]
        elif pd.isna(obj):
            return None
        else:
            return obj

    def calculate_weighted_metrics(self, results: List[OCRMetrics]) -> Dict:
        """
        Calculate weighted CER and WER based on text length

        Args:
            results: List of OCRMetrics objects

        Returns:
            Dictionary with weighted metrics
        """
        if not results:
            return {}

        total_char_errors = 0
        total_ref_chars = 0
        total_word_errors = 0
        total_ref_words = 0

        for metrics in results:
            # Character level: sum errors and reference characters
            char_errors = metrics.substitutions + metrics.deletions + metrics.insertions
            total_char_errors += char_errors
            total_ref_chars += metrics.ref_length

            # Word level: estimate word errors from WER and reference word count
            ref_words = len(metrics.ref_text.split())
            word_errors = metrics.wer * ref_words
            total_word_errors += word_errors
            total_ref_words += ref_words

        # Calculate weighted metrics
        weighted_cer = total_char_errors / total_ref_chars if total_ref_chars > 0 else 1.0
        weighted_wer = total_word_errors / total_ref_words if total_ref_words > 0 else 1.0

        return {
            'weighted_cer': weighted_cer,
            'weighted_wer': weighted_wer,
            'weighted_char_accuracy': 1 - weighted_cer,
            'weighted_word_accuracy': 1 - weighted_wer,
            'total_char_errors': total_char_errors,
            'total_ref_chars': total_ref_chars,
            'total_word_errors': total_word_errors,
            'total_ref_words': total_ref_words
        }

    def generate_report(self, results: List[OCRMetrics],
                        output_dir: str = "evaluation/reports") -> Dict:
        """
        Generate comprehensive evaluation report

        Args:
            results: List of OCRMetrics objects
            output_dir: Directory to save report files

        Returns:
            Dictionary with summary statistics
        """
        if not results:
            print("No results to generate report")
            return {}

        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Convert to DataFrame for analysis
        data = []
        for metrics in results:
            ref_word_count = len(metrics.ref_text.split())
            hyp_word_count = len(metrics.hyp_text.split())
            char_errors = metrics.substitutions + metrics.deletions + metrics.insertions
            word_errors = metrics.wer * ref_word_count  # Estimate from WER

            data.append({
                'book': metrics.book_name,
                'page': metrics.page_name,
                'file_path': metrics.file_path,
                'cer': float(metrics.cer),
                'wer': float(metrics.wer),
                'substitutions': int(metrics.substitutions),
                'deletions': int(metrics.deletions),
                'insertions': int(metrics.insertions),
                'ref_length': int(metrics.ref_length),
                'hyp_length': int(metrics.hyp_length),
                'ref_word_count': ref_word_count,
                'hyp_word_count': hyp_word_count,
                'char_errors': char_errors,
                'word_errors': word_errors,
                'char_accuracy': float(1 - metrics.cer),
                'word_accuracy': float(1 - metrics.wer),
                'ref_text_length': len(metrics.ref_text),
                'hyp_text_length': len(metrics.hyp_text)
            })

        df = pd.DataFrame(data)

        # Calculate weighted metrics (most important for overall quality)
        weighted_metrics = self.calculate_weighted_metrics(results)

        # Calculate simple page-level statistics (for reference)
        page_level_stats = {
            'avg_cer': float(df['cer'].mean()),
            'avg_wer': float(df['wer'].mean()),
            'median_cer': float(df['cer'].median()),
            'median_wer': float(df['wer'].median()),
            'std_cer': float(df['cer'].std()),
            'std_wer': float(df['wer'].std()),
            'avg_char_accuracy': float(df['char_accuracy'].mean()),
            'avg_word_accuracy': float(df['word_accuracy'].mean()),
        }

        # Calculate summary statistics
        summary = {
            'total_files': int(len(df)),
            'total_ref_characters': int(df['ref_length'].sum()),
            'total_hyp_characters': int(df['hyp_length'].sum()),
            'total_ref_words': int(df['ref_word_count'].sum()),
            'total_hyp_words': int(df['hyp_word_count'].sum()),
            'total_char_errors': int(df['char_errors'].sum()),
            'total_word_errors': float(df['word_errors'].sum()),
            'total_substitutions': int(df['substitutions'].sum()),
            'total_deletions': int(df['deletions'].sum()),
            'total_insertions': int(df['insertions'].sum()),
            'evaluation_date': datetime.now().isoformat(),

            # Page-level statistics (for reference)
            'page_level': page_level_stats,

            # Weighted metrics (most important for overall quality)
            'weighted_metrics': weighted_metrics,

            # Primary metrics for corpus quality - use weighted averages
            'primary_metrics': {
                'cer': weighted_metrics['weighted_cer'],
                'wer': weighted_metrics['weighted_wer'],
                'char_accuracy': weighted_metrics['weighted_char_accuracy'],
                'word_accuracy': weighted_metrics['weighted_word_accuracy']
            }
        }

        # Add book-level statistics with weighted averages
        if len(df['book'].unique()) > 1:
            book_stats = []
            for book_name in df['book'].unique():
                book_df = df[df['book'] == book_name]

                # Calculate weighted metrics for each book
                book_char_errors = book_df['char_errors'].sum()
                book_ref_chars = book_df['ref_length'].sum()
                book_word_errors = book_df['word_errors'].sum()
                book_ref_words = book_df['ref_word_count'].sum()

                book_weighted_cer = book_char_errors / book_ref_chars if book_ref_chars > 0 else 1.0
                book_weighted_wer = book_word_errors / book_ref_words if book_ref_words > 0 else 1.0

                book_stats.append({
                    'book': book_name,
                    'weighted_cer': float(book_weighted_cer),
                    'weighted_wer': float(book_weighted_wer),
                    'avg_cer': float(book_df['cer'].mean()),
                    'avg_wer': float(book_df['wer'].mean()),
                    'page_count': int(len(book_df)),
                    'total_ref_chars': int(book_ref_chars),
                    'total_ref_words': int(book_ref_words)
                })

            summary['book_statistics'] = book_stats
        else:
            summary['book_statistics'] = []

        # Save detailed results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        detailed_path = output_path / f"ocr_evaluation_detailed_{timestamp}.csv"
        df.to_csv(detailed_path, index=False)
        print(f"Detailed results saved to: {detailed_path}")

        # Save summary report - ensure all values are serializable
        serializable_summary = self._convert_to_serializable(summary)
        summary_path = output_path / f"ocr_evaluation_summary_{timestamp}.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_summary, f, indent=2, ensure_ascii=False)
        print(f"Summary report saved to: {summary_path}")

        # Generate visualization data
        self._generate_visualization_data(df, output_path, weighted_metrics)

        print(f"\n=== EVALUATION SUMMARY ===")
        print(f"Processed {summary['total_files']} files")
        print(f"Total reference characters: {summary['total_ref_characters']:,}")
        print(f"Total reference words: {summary['total_ref_words']:,}")

        print(f"\n--- WEIGHTED METRICS (Recommended for overall quality) ---")
        print(f"Weighted CER: {summary['primary_metrics']['cer']:.4f} ({summary['primary_metrics']['char_accuracy']:.2%} accuracy)")
        print(f"Weighted WER: {summary['primary_metrics']['wer']:.4f} ({summary['primary_metrics']['word_accuracy']:.2%} accuracy)")

        print(f"\n--- PAGE-LEVEL AVERAGES (For reference) ---")
        print(f"Average page CER: {summary['page_level']['avg_cer']:.4f}")
        print(f"Average page WER: {summary['page_level']['avg_wer']:.4f}")

        print(f"\n--- ERROR BREAKDOWN ---")
        print(f"Total character errors: {summary['total_char_errors']:,}")
        print(f"  - Substitutions: {summary['total_substitutions']:,}")
        print(f"  - Deletions: {summary['total_deletions']:,}")
        print(f"  - Insertions: {summary['total_insertions']:,}")
        print(f"Total word errors: {int(summary['total_word_errors']):,}")

        return serializable_summary

    def _generate_visualization_data(self, df: pd.DataFrame, output_path: Path, weighted_metrics: Dict):
        """
        Generate data for visualizations

        Args:
            df: DataFrame with evaluation results
            output_path: Path to save visualization data
            weighted_metrics: Weighted metrics for the corpus
        """
        # Error distribution by type
        error_data = {
            'substitutions': int(df['substitutions'].sum()),
            'deletions': int(df['deletions'].sum()),
            'insertions': int(df['insertions'].sum())
        }

        # CER distribution bins
        cer_bins = [0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
        cer_labels = ['<1%', '1-2%', '2-5%', '5-10%', '10-20%', '20-50%', '50-100%']

        df['cer_bin'] = pd.cut(df['cer'], bins=cer_bins, labels=cer_labels, right=False)
        cer_distribution = df['cer_bin'].value_counts().sort_index().to_dict()

        # Convert to native types
        cer_distribution_native = {str(k): int(v) for k, v in cer_distribution.items()}

        # Save visualization data
        viz_data = {
            'error_types': error_data,
            'cer_distribution': cer_distribution_native,
            'books': list(df['book'].unique()),
            'weighted_metrics': self._convert_to_serializable(weighted_metrics),
            'avg_metrics_by_book': self._convert_to_serializable(
                df.groupby('book')[['cer', 'wer']].mean().to_dict('index')
            )
        }

        viz_path = output_path / "visualization_data.json"
        with open(viz_path, 'w', encoding='utf-8') as f:
            json.dump(viz_data, f, indent=2, ensure_ascii=False)