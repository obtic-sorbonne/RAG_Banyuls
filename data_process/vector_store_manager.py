import os
import uuid
from typing import List, Dict

import chromadb
from chromadb.config import Settings

from .chunking_strategies import PageChunker, RecordChunker
from .embedding_generator import EmbeddingGenerator


class VectorStoreManager:
    def __init__(self, config: Dict, persist_dir="vector_store"):
        # self.client = chromadb.Client(Settings(
        #     chroma_db_impl="duckdb+parquet",
        #     persist_directory=persist_dir
        # ))
        self.config = config
        self.vector_store_type = config.get("vector_store_type", "chromadb")

        self.embedding_dim = config.get("embedding_dim", 768)

        if self.vector_store_type == "chromadb":
            chroma_persist_dir = config.get("chromadb_persist_dir", "./chromadb_data")
            os.makedirs(chroma_persist_dir, exist_ok=True)
            if config.get("chromadb_type", "cloud") == "cloud":
                chroma_api_key = config.get("chroma_api_key") or os.getenv("CHROMA_API_KEY")
                self.client = chromadb.CloudClient(
                    tenant=config.get("chroma_tenant", "default"),
                    api_key=chroma_api_key,
                    database=config.get("vector_db", "ocean_observations")
                )
            elif config.get("chromadb_type", "cloud") == "http":
                self.client = chromadb.HttpClient(
                    host=config.get("chromadb_host", 'localhost'),
                    port=config.get("chromadb_port", 8000)
                )
            else:
                self.client = chromadb.PersistentClient(
                    path=chroma_persist_dir,
                    settings=Settings(
                        anonymized_telemetry=False,
                        allow_reset=True
                    )
                )
            self.collection = self.client.get_or_create_collection(
                name=config.get("vector_db", "ocean_observations"),
                metadata={"hnsw:space": "cosine"}
            )

        elif self.vector_store_type == "deeplake":
            deeplake_api_key = config.get("deeplake_api_key") or os.getenv("DEEPLAKE_API_KEY")
            self.deeplake_path = config.get("deeplake_path", "al://wenjun/oob_rag_dev")

            try:
                # Try to open existing dataset
                self.ds = deeplake.open(self.deeplake_path, token=deeplake_api_key)
                print(f"Opened existing Activeloop dataset: {self.deeplake_path}")
            except Exception as e:
                print(f"Dataset doesn't exist or couldn't be opened: {e}")
                # Create new dataset
                try:
                    print(f"Attempting to create new dataset at: {self.deeplake_path}")
                    self.ds = deeplake.create(self.deeplake_path, token=deeplake_api_key)
                    embedding_dim = config.get("embedding_dim", 768)

                    # Add columns with DeepLake v4 API - correct syntax
                    self.ds.add_column('text', deeplake.types.Text())
                    self.ds.add_column('embedding', deeplake.types.Embedding(embedding_dim))
                    self.ds.add_column('metadata', deeplake.types.Dict())

                    # Commit the schema
                    self.ds.commit()
                    print(f"Created new Activeloop dataset: {self.deeplake_path}")
                except Exception as create_error:
                    print(f"Failed to create dataset: {create_error}")
                    print(f"Error type: {type(create_error).__name__}")
                    print(f"Consider creating the dataset manually via Activeloop dashboard first")
                    raise

            self.embedding_model = EmbeddingGenerator(
                config.get("embedding_model", "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"),
                config.get("embedding_dim", 768)
            )

    def index_book(self, book_data: Dict):
        """Index a processed book into the vector store"""
        chunks = self._prepare_chunks(book_data)

        years_str = ",".join(str(year) for year in book_data["metadata"]["years"]) if book_data["metadata"][
            "years"] else ""
        # primary_year = book_data["metadata"]["years"][0] if book_data["metadata"]["years"] else None

        if self.vector_store_type == "chromadb":
            embeddings = [chunk["embedding"] for chunk in chunks]
            documents = [chunk["content"] for chunk in chunks]
            metadatas = [{
                "book": book_data["book_name"],
                "years": years_str,
                "page": chunk["metadata"]["page_number"],
                "chunk_type": chunk["chunk_type"],
                "entities": str(chunk["metadata"]["entities"])
            } for chunk in chunks]

            ids = [str(uuid.uuid4()) for _ in chunks]

            self.collection.add(
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )

        elif self.vector_store_type == "deeplake":
            # Process chunks in batches for better performance
            for chunk in chunks:
                text = chunk["content"]
                embedding = chunk["embedding"]
                metadata = {
                    "book": book_data["book_name"],
                    "years": years_str,
                    "page": chunk["metadata"]["page_number"],
                    "chunk_type": chunk["chunk_type"],
                    "entities": str(chunk["metadata"]["entities"])
                }

                # Ensure embedding is properly formatted as a list
                if hasattr(embedding, 'tolist'):
                    embedding = embedding.tolist()
                elif not isinstance(embedding, list):
                    embedding = list(embedding)

                # Append individual records (not batches)
                self.ds.append({
                    'text': text,  # Single text string
                    'embedding': embedding,  # Single embedding list
                    'metadata': metadata  # Single metadata dict
                })

            self.ds.commit_async()
            print(f"Successfully indexed {len(chunks)} chunks from book: {book_data['book_name']}")

    def _prepare_chunks(self, book_data: Dict) -> List[Dict]:
        """Apply chunking strategy to book content"""
        # Select chunking strategy based on book type
        if "page" in self.config['chunk_strategy']:
            chunker = PageChunker()
        else:
            chunker = RecordChunker()

        chunks = []
        for page in book_data["pages"]:
            page_metadata = {
                "book": book_data["book_name"],
                "years": book_data["metadata"]["years"],
                "page_number": page["page_number"],
                "entities": page["entities"]
            }
            chunks.extend(chunker.chunk(page["cleaned_content"], page_metadata))

        # Generate embeddings
        embedding_gen = EmbeddingGenerator(
            self.config.get("embedding_model", "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"),
            self.config.get("embedding_dim", 768))
        return embedding_gen.batch_generate(chunks)

    def get_collection_stats(self):
        """Get statistics about the vector store"""
        if self.vector_store_type == "chromadb":
            return {
                "count": self.collection.count(),
                "dimension": self.collection.metadata.get("dimension", "unknown"),
                "space": self.collection.metadata.get("hnsw:space", "cosine")
            }
        elif self.vector_store_type == "deeplake":
            try:
                return {
                    "count": len(self.ds),
                    "dimension": len(self.ds.embedding[0]) if len(self.ds) > 0 else self.config.get("embedding_dim",
                                                                                                    768)
                }
            except Exception:
                return {
                    "count": len(self.ds) if hasattr(self.ds, '__len__') else 0,
                    "dimension": self.config.get("embedding_dim", 768)
                }
