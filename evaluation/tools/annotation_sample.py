import os
import random
from pathlib import Path
from PIL import Image
import argparse

def find_jp2_files(root_dir):
    """Find all JP2 files in the directory structure."""
    jp2_files = []
    root_path = Path(root_dir)

    for book_dir in root_path.iterdir():
        if book_dir.is_dir():
            for file_path in book_dir.glob("*.jp2"):
                jp2_files.append(file_path)

    return jp2_files

def sample_files(file_list, sample_percentage=10):
    """Sample a percentage of files from the list."""
    sample_size = max(1, int(len(file_list) * sample_percentage / 100))
    return random.sample(file_list, sample_size)

def convert_jp2_to_png(jp2_path, output_dir, max_dimension=1024, jpeg_output=True):
    """Convert a single JP2 file to PNG/JPEG format with compression."""
    try:
        # Create output directory structure
        book_name = jp2_path.parent.name
        output_book_dir = Path(output_dir) / book_name
        output_book_dir.mkdir(parents=True, exist_ok=True)

        # Generate output filename
        if jpeg_output:
            output_filename = jp2_path.stem + ".jpg"
        else:
            output_filename = jp2_path.stem + ".png"

        output_path = output_book_dir / output_filename

        # Convert image
        with Image.open(jp2_path) as img:
            original_size = img.size
            original_mb = jp2_path.stat().st_size / (1024 * 1024)

            # Convert to RGB if necessary
            if img.mode != 'RGB':
                img = img.convert('RGB')

            # Resize if image is too large
            if max(img.size) > max_dimension:
                # Calculate new size maintaining aspect ratio
                if img.width > img.height:
                    new_width = max_dimension
                    new_height = int(img.height * max_dimension / img.width)
                else:
                    new_height = max_dimension
                    new_width = int(img.width * max_dimension / img.height)

                img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                print(f"Resized: {original_size} -> {img.size}")

            # Save with appropriate compression
            if jpeg_output:
                img.save(output_path, 'JPEG', quality=85, optimize=True)
            else:
                img.save(output_path, 'PNG', optimize=True, compress_level=6)

        # Check output file size
        output_size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"Converted: {jp2_path.name} ({original_mb:.1f} MB) -> {output_path.name} ({output_size_mb:.1f} MB)")
        return output_path

    except Exception as e:
        print(f"Error converting {jp2_path}: {str(e)}")
        return None

def main():
    input_dir = "data/images/original"
    output_dir = "data/images/sample"
    percentage = 10.0
    seed = 202509041614

    # Compression settings - adjust these to control file size
    max_dimension = 1024  # Maximum width/height in pixels (smaller = smaller files)
    jpeg_output = True    # True for JPEG (smaller), False for PNG (higher quality)

    # Set random seed for reproducible results
    random.seed(seed)

    # Find all JP2 files
    print(f"Searching for JP2 files in {input_dir}...")
    jp2_files = find_jp2_files(input_dir)
    print(f"Found {len(jp2_files)} JP2 files")

    if not jp2_files:
        print("No JP2 files found. Please check your input directory.")
        return

    # Sample files
    sampled_files = sample_files(jp2_files, percentage)
    print(f"Sampled {len(sampled_files)} files ({percentage}%)")

    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Convert files
    print(f"\nConverting files to {'JPEG' if jpeg_output else 'PNG'} with max dimension {max_dimension}px...")
    successful_conversions = 0
    failed_conversions = 0
    total_original_size = 0
    total_output_size = 0

    for jp2_file in sampled_files:
        original_size = jp2_file.stat().st_size / (1024 * 1024)
        total_original_size += original_size

        result = convert_jp2_to_png(jp2_file, output_dir, max_dimension, jpeg_output)
        if result:
            output_size = result.stat().st_size / (1024 * 1024)
            total_output_size += output_size
            successful_conversions += 1
        else:
            failed_conversions += 1

    print(f"\nConversion complete!")
    print(f"Successful: {successful_conversions}")
    print(f"Failed: {failed_conversions}")
    if total_original_size > 0:
        print(f"Total size reduction: {total_original_size:.1f} MB -> {total_output_size:.1f} MB")
        print(f"Compression ratio: {(total_output_size/total_original_size)*100:.1f}%")

    # Save sample list for reference
    sample_list_file = Path(output_dir) / "sampled_files.txt"
    with open(sample_list_file, 'w') as f:
        f.write("Sampled files for OCR annotation:\n")
        f.write(f"Sample percentage: {percentage}%\n")
        f.write(f"Random seed: {seed}\n")
        f.write(f"Max dimension: {max_dimension}px\n")
        f.write(f"Output format: {'JPEG' if jpeg_output else 'PNG'}\n")
        f.write(f"Total sampled: {len(sampled_files)}\n")
        f.write(f"Size reduction: {total_original_size:.1f} MB -> {total_output_size:.1f} MB\n\n")
        for file_path in sorted(sampled_files):
            f.write(f"{file_path}\n")

    print(f"Sample list saved to: {sample_list_file}")

if __name__ == "__main__":
    main()