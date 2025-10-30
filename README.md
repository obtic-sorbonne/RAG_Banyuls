# Historical Oceanographic Document Valorization

An end-to-end pipeline for transforming historical oceanographic documents into an intelligent, queryable knowledge base using advanced OCR correction and hybrid RAG systems.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Streamlit](https://img.shields.io/badge/Web%20UI-Streamlit-FF4B4B)
![RAG](https://img.shields.io/badge/Architecture-RAG%20Pipeline-green)
![OCR](https://img.shields.io/badge/OCR-Multi--Model-orange)

##  Overview

This project transforms ~2,900 pages of 18th-20th century oceanographic documents from the Banyuls-sur-Mer Oceanological Observatory (OOB) into an intelligent question-answering system. The pipeline combines:

- **Multi-model OCR** with context-aware post-correction (67.02% word accuracy)
- **Hybrid RAG** with vector + keyword retrieval and re-ranking
- **Web interface** for both research queries and expert annotation

##  Key Features

### OCR Pipeline
- **Multi-engine text recognition**: Kraken, Mistral OCR, Qwen-VL
- **Context-aware correction**: Uses temporal document context for improved accuracy
- **Expert annotation tool**: Web interface for ground truth creation

### RAG System
- **Hybrid retrieval**: ChromaDB (vector) + Whoosh (BM25F)
- **Intelligent re-ranking**: Similarity-based result optimization
- **Citation generation**: Source verification with book/page references
- **Query classification**: Automatic strategy selection

### User Interfaces
- **Research interface**: Natural language querying for historical data
- **Annotation tool**: Expert-friendly ground truth creation
- **Feedback system**: Quality assessment and continuous improvement

## 📊 Results

| Method | WER | CER | Word Accuracy |
|--------|-----|-----|---------------|
| Kraken | 96.12% | 78.63% | 3.88% |
| Mistral OCR | 77.65% | 53.29% | 22.35% |
| Qwen-VL-32B | 49.89% | 42.48% | 50.11% |
| **Our Method** | **32.98%** | **38.36%** | **67.02%** |

*17% relative improvement over best single model*



## ⚡ Quick Start

### Prerequisites

- Python 3.9+
- 16GB+ RAM (for embedding models)

### Installation

1. **Clone repository**
```bash
git clone https://github.com/obtic-sorbonne/RAG_Banyuls.git
cd RAG_Banyuls

pip install -r requirements.txt
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Configure dependencies**
   
Edit configuration with your API keys and paths

4. **Process sample data**

Edit and run
``` bash
python ocr.py
python collect.py
```
5. **Launch web interface**

```bash
streamlit run app/app.py
```


# Citation
```bibtex
@article{wenjun2025oceanographic,
  title={Valorizing Historical Oceanographic Documents: An End-to-End RAG Pipeline with Advanced OCR Correction},
  author={Li Wenjun and Alrahabi Motasem and Castellon Clément},
  journal={arXiv preprint},
  year={2025}
}
```




  


