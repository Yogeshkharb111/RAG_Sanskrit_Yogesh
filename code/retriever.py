from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from embeddings import EmbeddingModel


class Retriever:
    def __init__(self, index_path: str, meta: dict, embed_model: str):
        self.index = faiss.read_index(index_path)
        self.meta = meta
        self.embed_model = EmbeddingModel(embed_model)

    def query(self, q: str, top_k: int = 5) -> List[str]:
        q_emb = self.embed_model.embed([q])

        # normalize
        q_emb = q_emb / (np.linalg.norm(q_emb, axis=1, keepdims=True) + 1e-10)
        D, I = self.index.search(q_emb.astype('float32'), top_k)
        results = []
        for idx in I[0]:
            results.append(self.meta['texts'][idx])
        return results
