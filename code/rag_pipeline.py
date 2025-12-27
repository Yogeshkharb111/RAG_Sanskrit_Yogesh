import os
import json

from config import *
from data_loader import load_all
from preprocessor import simple_sanskrit_cleanup
from chunker import chunk_text
from indexer import build_faiss_index, load_metadata
from retriever import Retriever
from generator import Generator

# --------------------------------------------------
# Global generator (lazy-loaded)
# --------------------------------------------------
_generator = None


# --------------------------------------------------
# BUILD INDEX
# --------------------------------------------------
def build_pipeline_index():
    text = load_all(RAW_DIR)
    cleaned = simple_sanskrit_cleanup(text)

    os.makedirs(os.path.dirname(CLEANED_TEXT_PATH), exist_ok=True)

    with open(CLEANED_TEXT_PATH, "w", encoding="utf-8") as f:
        f.write(cleaned)

    chunks = chunk_text(
        cleaned,
        chunk_size=CHUNK_SIZE,
        overlap=CHUNK_OVERLAP
    )

    with open(CHUNKS_JSON, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    build_faiss_index(
        chunks,
        EMBEDDING_MODEL,
        FAISS_INDEX_PATH,
        METADATA_PATH
    )

    print("✅ Index built successfully")
    print("📦 Total chunks:", len(chunks))


# --------------------------------------------------
# QUERY PIPELINE
# --------------------------------------------------
def answer_query(query: str) -> str:
    global _generator

    # Safety check
    if not os.path.exists(FAISS_INDEX_PATH):
        return "FAISS index not found. Run with --build first."

    metadata = load_metadata(METADATA_PATH)

    retriever = Retriever(
        FAISS_INDEX_PATH,
        metadata,
        EMBEDDING_MODEL
    )

    # Handle English queries on Sanskrit corpus
    retrieval_query = query
    if query.isascii():
        retrieval_query = "शंखनाद"  # safe Sanskrit anchor keyword

    contexts = retriever.query(retrieval_query, TOP_K)

    if not contexts:
        return "⚠️ No relevant context found in the documents."

    # Lazy-load generator
    if _generator is None:
        _generator = Generator(GENERATION_MODEL)

    answer = _generator.generate(query, contexts)

    if not answer.strip():
        return (
            "The system retrieved relevant Sanskrit context, "
            "but the language model could not generate a clear answer "
            "due to OCR noise and limited corpus size."
        )

    return answer
