from sentence_transformers import SentenceTransformer
import numpy as np


class EmbeddingModel:
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name, device="cpu")

    def embed(self, texts, show_progress_bar=False):
        emb = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=show_progress_bar)
        # normalize for cosine-sim
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        norms[norms==0] = 1.0
        emb = emb / norms
        return emb

