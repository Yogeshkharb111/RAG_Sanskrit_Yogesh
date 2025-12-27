# Sanskrit RAG — CPU-Only Retrieval-Augmented Generation

## Overview

This project implements an **end-to-end Retrieval-Augmented Generation (RAG) system for Sanskrit documents**, running entirely on **CPU (no GPU required)**.

The system:

- Ingests Sanskrit documents (`.pdf` / `.txt`)
- Retrieves relevant context using **vector search**
- Generates **grounded answers** using a **local LLM**
- Works **offline after one-time model download**

---

## Why CPU-Only?

- GPU usage is **not permitted by the assignment**
- Ensures **portability and local execution**
- Uses **CPU-friendly open-source models**
- **No external APIs** (OpenAI, cloud services) required

CPU-only execution is enforced by:

- FAISS **CPU** index
- Hugging Face models loaded with **CPU settings**
- No CUDA or `device_map` usage

---

## Repository Structure

```
RAG_Sanskrit_Yogesh/
│
├── code/
│ ├── main.py         # Entry point (run this)
│ │
│ ├── config.py       # Model names, paths, parameters
│ │
│ ├── data_loader.py  # Load .txt / .pdf Sanskrit documents
│ │
│ ├── preprocessor.py # Sanskrit text cleaning & normalization
│ │
│ ├── chunker.py      # Text chunking logic
│ │
│ ├── embeddings.py   # Load embedding model (CPU, local)
│ │
│ ├── indexer.py      # Create & save FAISS index
│ │
│ ├── retriever.py    # Retrieve top-k relevant chunks
│ │
│ ├── generator.py    # CPU-based LLM response generation
│ │
│ └── rag_pipeline.py # Connect retriever + generator
│
├── data/
│ ├── raw/
│ │ └── rag-docs.pdf
│ │
│ └── processed/
│ ├── cleaned_text.txt
│ └── chunks.json
│
├── models/
│ ├── embeddings/
│ │ └── paraphrase-multilingual-MiniLM-L12-v2/
│ │
│ └── llm/
│ └── flan-t5-base/
│
├── vectorstore/
│ ├── faiss.index # FAISS CPU index
│ └── metadata.pkl # Chunk metadata
│
├── report/
│ └── Sanskrit_RAG_Report.pdf
├── images/
│ └── System_Architecture.png
│
├── demo/
│ └── demo_video.mp4 # Optional demo video
│
├── requirements.txt
└── README.md
```
## System Architecture
![RAG Sanskrit System Architecture](https://github.com/user-attachments/assets/ad1abdc9-d269-40f3-96ec-a22930d1582d)


---

## Requirements

- Python **3.9+**
- Windows / Linux / macOS
- Internet connection (**first run only**, for model download)
- Virtual environment is **optional**
- Project can be run using **system Python**

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Place Sanskrit Documents

Put your Sanskrit `.pdf` or `.txt` files inside, e.g.:

```
data/raw/Rag-docs.pdf
```

---

## One-Time Model Download (IMPORTANT)

Models are downloaded **once** and cached locally. After this step, the system works **completely offline**.

---

### Download Embedding Model

**Model:**

```
sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
```

Create a temporary file in the project root: `download_embeddings.py`

```python
from sentence_transformers import SentenceTransformer

SentenceTransformer(
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    cache_dir="models/embeddings"
)
```

Run:

```bash
python download_embeddings.py
```

You may delete the file after download.

### Download Generator Model (LLM)

**Model:** `google/flan-t5-base`

Create a temporary file: `download_llm.py`

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

AutoTokenizer.from_pretrained(
    "google/flan-t5-base",
    cache_dir="models/llm"
)
AutoModelForSeq2SeqLM.from_pretrained(
    "google/flan-t5-base",
    cache_dir="models/llm"
)
```

Run:

```bash
python download_llm.py
```

---

## Build the Index

This step performs:

- PDF text extraction
- Text cleaning and normalization
- Chunking
- Embedding generation
- FAISS index construction

Run:

```bash
python code/main.py --build
```

Generated artifacts:

- `vectorstore/faiss.index`
- `vectorstore/metadata.pkl`
- `data/processed/cleaned_text.txt`
- `data/processed/chunks.json`

---

## Query the System

### Example Queries

```bash
python code/main.py --query "Who is Kalidasa?"
python code/main.py --query "What lesson is taught in the story of the devotee?"
python code/main.py --query "कालीदासः कः?"
```

---

## Configuration

All configurable parameters are defined in:

```
code/config.py
```

You can modify:

- Embedding model
- Generator (LLM) model
- Chunk size and overlap
- Top-K retrieval count

---

## Known Limitations

- Sanskrit text may be OCR-corrupted
- Limited document corpus
- CPU-based LLM inference is slow
- Abstract reasoning capability is limited

These constraints are expected and documented.

---

## Troubleshooting

**Answer could not be generated**

- Use entity-based questions
- Prefer simple Sanskrit or English
- Ensure index build completed successfully

**Slow performance**

- First run includes model downloads
- CPU inference is slower than GPU
- Reduce `TOP_K` in `code/config.py`

---

## Future Improvements

- Improved Sanskrit OCR cleanup
- Roman-to-Devanagari transliteration
- Quantized CPU LLMs (GGUF / GPTQ)
- Larger Sanskrit document corpus
- Retrieval quality evaluation metrics

---

## Conclusion

This project demonstrates a fully offline, CPU-only Retrieval-Augmented Generation system for Sanskrit documents, following standard RAG architecture and best practices. Despite OCR noise and limited data, it successfully retrieves relevant Sanskrit context and generates grounded answers.

---

## License

Open-source, for academic and learning purposes.
