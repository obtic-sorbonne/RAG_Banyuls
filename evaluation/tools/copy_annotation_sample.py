import os
import shutil
from pathlib import Path

def copy_text_files(sample_list_file, text_input_dir, text_output_dir, text_ext=".txt"):
    """
    Copy text files corresponding to sampled JP2 files into a new folder.

    Args:
        sample_list_file (str or Path): Path to the text file containing sampled JP2 file paths.
        text_input_dir (str or Path): Root directory where original text files are stored.
        text_output_dir (str or Path): Directory to copy sampled text files to.
        text_ext (str): Extension of the text files (e.g., '.txt', '.xml').
    """
    text_input_dir = Path(text_input_dir)
    text_output_dir = Path(text_output_dir)
    text_output_dir.mkdir(parents=True, exist_ok=True)

    # Read sampled JP2 list
    with open(sample_list_file, 'r') as f:
        lines = f.readlines()

    # Extract valid JP2 file paths
    jp2_files = [Path(line.strip()) for line in lines if line.strip().endswith(".jp2")]

    copied = 0
    missing = 0

    for jp2_file in jp2_files:
        book_name = jp2_file.parent.name
        text_name = jp2_file.stem + text_ext

        source_text_path = text_input_dir / book_name / text_name
        target_book_dir = text_output_dir / book_name
        target_book_dir.mkdir(parents=True, exist_ok=True)
        target_text_path = target_book_dir / text_name

        if source_text_path.exists():
            shutil.copy2(source_text_path, target_text_path)
            copied += 1
        else:
            print(f"⚠️ Missing text file: {source_text_path}")
            missing += 1

    print(f"\nText file copy complete!")
    print(f"Copied: {copied}")
    print(f"Missing: {missing}")
    print(f"Output folder: {text_output_dir}")


def main():
    # Adjust paths as needed
    sample_list_file = "data/images/sample/sampled_files.txt"  # from your first script
    text_input_dir = "data/raw-ocr/judged/context-32b-candidate-3"
    text_output_dir = "data/ocr/sample"
    text_ext = ".txt"

    copy_text_files(sample_list_file, text_input_dir, text_output_dir, text_ext)


if __name__ == "__main__":
    main()
