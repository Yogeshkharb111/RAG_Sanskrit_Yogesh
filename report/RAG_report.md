# RAG Sanskrit — Technical Report

## 1. System Architecture

### System Architecture Diagram

![RAG Sanskrit System Architecture](<img width="1024" height="1024" alt="Gemini_Generated_Image_9wwoap9wwoap9wwo" src="https://github.com/user-attachments/assets/ad1abdc9-d269-40f3-96ec-a22930d1582d" />
)

**Figure 1:** High-level architecture of the CPU-only Sanskrit Retrieval-Augmented
Generation (RAG) system. The pipeline consists of document ingestion,
preprocessing, chunking, embedding generation using sentence-transformers,
vector indexing with FAISS, and answer generation using a Hugging Face
transformer model running entirely on CPU.

### Architecture Description

- **Retriever:** Dense vector retrieval using `sentence-transformers`
  embeddings combined with FAISS (`IndexFlatIP`) over L2-normalized vectors
  for efficient nearest-neighbor search.
- **Generator:** A Hugging Face transformer model running on CPU via the
  `transformers` pipeline. The default generator used in this project is
  `google/flan-t5-base`.
- **Pipeline Design:** Modular separation between data loading, preprocessing,
  chunking, embedding generation, indexing, retrieval, and generation.
  The main entry points are `code/main.py` and `code/rag_pipeline.py`.

---

## 2. Documents Used

- Source documents are placed in the `data/raw/` directory.
- For the current run, the system processed:
  - `Rag-docs.pdf` (user-provided Sanskrit document)

---

## 3. Preprocessing

### 3.1 PDF Extraction
- Raw text is extracted from PDF documents using the `PyPDF2` library.
- Each page is read sequentially and concatenated into a single text stream.

### 3.2 Text Cleaning
- Unicode normalization and whitespace collapsing are applied to remove noise.
- Cleaning logic is implemented in `code/preprocessor.py`.
- Output file:
  - `data/processed/cleaned_text.txt`

### 3.3 Chunking
- Cleaned text is split using a sliding-window chunking strategy.
- Parameters:
  - `chunk_size = 600`
  - `overlap = 100`
- Chunk metadata and content are stored in:
  - `data/processed/chunks.json`

---

## 4. Retrieval

### 4.1 Embedding Model
- Embedding model used:
  - `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- This multilingual model supports Sanskrit and related Indic scripts.
- Model configuration is defined in `code/config.py`.

### 4.2 Vector Index
- FAISS index type:
  - `IndexFlatIP` (Inner Product similarity)
- All embeddings are L2-normalized to enable cosine similarity via inner product.
- Index file:
  - `vectorstore/faiss.index`

### 4.3 Metadata Storage
- Chunk texts and identifiers are stored separately to map FAISS indices
  back to their corresponding text.
- Metadata file:
  - `vectorstore/metadata.pkl`

---

## 5. Generation

### 5.1 Generator Model
- Default generator:
  - `google/flan-t5-base`
- Loaded using Hugging Face `text2text-generation` pipeline.
- CPU-only execution enforced by:
  - `device = -1`

### 5.2 Prompt Construction
- Retrieved top-k chunks are concatenated to form context.
- Final prompt format:


- The generator produces a short, grounded answer based on retrieved context.

---
```
Context:
<retrieved chunks>

Question:
<user query>
```

---

## 6. Run & Observed Status

### 6.1 Indexing Run
- Indexing completed successfully without errors.
- Generated artifacts:
- `vectorstore/faiss.index`
- `vectorstore/metadata.pkl`
- `data/processed/cleaned_text.txt`
- `data/processed/chunks.json`

### 6.2 Model Downloads
- On the first run, required transformer and embedding model weights are
downloaded and cached locally by Hugging Face.
- Download size and time depend on selected models and network speed.

---

## 7. Performance Notes & Measurement Plan

### 7.1 Execution Constraints
- The system runs entirely on CPU (no GPU or CUDA usage).
- Performance depends on:
- CPU architecture
- Available RAM
- Model size

### 7.2 Suggested Metrics for Final Evaluation
- Model download size and time (first run only)
- Total indexing time
- Average query latency:
- Embedding time
- Retrieval time
- Generation time
- Peak memory usage during indexing and querying

> These measurements should be recorded on the target machine and added
> before final submission.

---

## 8. Improvements & Future Work

- Replace the generator with a quantized LLM (GPTQ / GGUF via llama.cpp)
for faster CPU inference.
- Add Sanskrit transliteration support (Romanized input → Devanagari).
- Use Sanskrit-specific or Indic-focused embedding and generation models.
- Implement retrieval evaluation metrics such as precision@k and recall@k.
- Add automated and human-based evaluation for answer quality.

---

## 9. How to Reproduce

### 9.1 Install Dependencies
```powershell
pip install -r requirements.txt
