from evaluation.evaluation_manager import EvaluationManager

# Initialize evaluation manager
# evaluation/example_usage.py
"""
Example usage of the OCR evaluation module
"""

from evaluation.ocr_metrics import OCREvaluationManager, CERWERCalculator

def example_single_evaluation():
    """Example of evaluating a single text pair"""
    calculator = CERWERCalculator()

    # Example texts
    ref_text = "The quick brown fox jumps over the lazy dog."
    hyp_text = "The quik brown fox jump over the lazy dog."

    # Calculate metrics
    metrics = calculator.calculate_all_metrics(ref_text, hyp_text)

    print(f"Reference: {ref_text}")
    print(f"OCR Output: {hyp_text}")
    print(f"CER: {metrics.cer:.4f} ({metrics.cer * 100:.2f}%)")
    print(f"WER: {metrics.wer:.4f} ({metrics.wer * 100:.2f}%)")
    print(f"Substitutions: {metrics.substitutions}")
    print(f"Deletions: {metrics.deletions}")
    print(f"Insertions: {metrics.insertions}")

def example_batch_evaluation():
    """Example of batch evaluation of all ground truth - OCR pairs"""
    evaluator = OCREvaluationManager(
        ground_truth_dir="data/ground_truth",
        ocr_results_dir="data/raw-ocr/judged/context-32b-candidate-3"
        # ocr_results_dir="data/raw-ocr/MoE/Qwen-VL-72b-instruct"

    )

    # Evaluate all matching pairs
    results = evaluator.evaluate_batch()

    # Generate comprehensive report
    summary = evaluator.generate_report(results, output_dir="data/evaluation/wer-reports")

    return summary

if __name__ == "__main__":
    print("=== Single Text Evaluation Example ===")
    example_single_evaluation()

    print("\n=== Batch Evaluation Example ===")
    summary = example_batch_evaluation()
    print(f"Batch evaluation completed. Processed {summary['total_files']} files.")