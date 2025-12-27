from typing import List


def chunk_text(text: str, chunk_size: int = 600, overlap: int = 100) -> List[str]:
    tokens = text.split()
    if not tokens:
        return []
    chunks = []
    i = 0
    while i < len(tokens):
        chunk = tokens[i:i+chunk_size]
        chunks.append(' '.join(chunk))
        i += chunk_size - overlap
    return chunks
