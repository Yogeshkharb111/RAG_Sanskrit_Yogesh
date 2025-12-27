# test/test_debug.py

from config import (
    FAISS_INDEX_PATH,
    METADATA_PATH,
    EMBEDDING_MODEL,
    GENERATION_MODEL,
    TOP_K,
)

from indexer import load_metadata
from retriever import Retriever
from generator import Generator


def main():
    print("Loading metadata...")
    meta = load_metadata(METADATA_PATH)
    print("Number of chunks:", len(meta.get("texts", [])))

    query = "Who is Kalidasa?"

    print("\n Query:", query)

    retriever = Retriever(FAISS_INDEX_PATH, meta, EMBEDDING_MODEL)
    contexts = retriever.query(query, top_k=TOP_K)

    print("\n Retrieved contexts:")
    for i, c in enumerate(contexts):
        print(f"\n--- Context {i} ---")
        print(c[:500])

    gen = Generator(GENERATION_MODEL)
    print("\n Generating answer...")
    answer = gen.generate(query, contexts, max_length=128)

    print("\n Answer:")
    print(answer)


if __name__ == "__main__":
    main()
