from data_process.corpus_processor import CorpusProcessor

def main():
    # Process and index all OCR results
    processor = CorpusProcessor()
    processor.process_directory(
        input_dir="./data/raw-ocr/judged/context-32b-candidate-3",
        output_dir="./data/dump/context-32b-candidate-3"
    )
    
    # Get vector store statistics
    stats = processor.vector_store.get_collection_stats()
    print(f"Vector store contains {stats['count']} documents")

if __name__ == "__main__":
    main()
