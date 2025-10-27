from ocr_processing.ocr_manager import OCRManager
from ocr_processing.cleaning.moe_cleaner import MoECleaner
from ocr_processing.cleaning.word_level_moe_cleaner import WordLevelMoECleaner
from ocr_processing.cleaning.llm_judge import LLMJudgeCleaner
import yaml
import os

# Load configuration
with open("config/ocr_settings.yaml", "r") as f:
    file = yaml.safe_load(f)
    ocr_config = file.get("ocr")
    moe_config = file.get("moe")
    llm_judge_config = file.get("llm_judge")

# Initialize OCR system
ocr_processor = OCRManager(ocr_config)


def ocr_image():
    # Example 1: Process a single image
    ocr_processor.process_image(
        image_path="data/feedback/08_OOB_05_meteo/08_OOB_05_024.jp2",
        output_path="data/raw-ocr/08_OOB_05_024-qwen.txt"
    )


def ocr_book():
    # Example 2: Process an entire book
    processed_pages = ocr_processor.process_book(
        book_path="data/books/Log_1832_NEPTUNE",
        output_root="data/raw-ocr"
    )
    print(f"Processed {processed_pages} pages")


def ocr_lib():
    output_dir = ocr_config.get("output", "data/raw-ocr")
    input_dir = ocr_config.get("input", "data/images")
    provider = ocr_config.get("api_provider")

    os.makedirs(output_dir, exist_ok=True)
    books = [os.path.join(input_dir, d)
             for d in os.listdir(input_dir)
             if os.path.isdir(os.path.join(input_dir, d))]

    print(f"Found {len(books)} books to process using {provider.upper()}")
    total_pages = 0
    successful_pages = 0

    for book_path in books:
        processed_pages = ocr_processor.process_book(
            book_path=book_path,
            output_root=output_dir
        )
        book_name = os.path.basename(book_path)
        print(f"\nProcessing book: {book_name}")

    print(f"\nProcessing complete! Success rate: {successful_pages / total_pages:.1%}")


# Post-process MoE strategy
def moe_clean():
    # Post-process MoE strategy
    # Initialize cleaner
    cleaner = MoECleaner(moe_config)
    # cleaner = WordLevelMoECleaner(moe_config)

    ## Process specific books or entire library
    ## cleaner.process_book()
    cleaner.process_library()


def llm_judge():
    # vote clean
    judge = LLMJudgeCleaner(llm_judge_config)

    # Process specific books or entire library
    # judge.process_book("05_OOB_01_log")
    # judge.process_book("05_OOB_02_log")
    # judge.process_book("05_OOB_03_log")
    judge.process_book("08_OOB_04_meteo")
    # judge.process_library()


if __name__ == '__main__':

    # moe_clean()
    llm_judge()
    # ocr_lib()
