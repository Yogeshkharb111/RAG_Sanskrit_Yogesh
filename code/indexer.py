import os
import pickle
from typing import List
import faiss
import numpy as np

from embeddings import EmbeddingModel


def build_faiss_index(texts: List[str], model_name: str, index_path: str, meta_path: str):
    emb_model = EmbeddingModel(model_name)
    embeddings = emb_model.embed(texts, show_progress_bar=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings.astype('float32'))
    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    faiss.write_index(index, index_path)
    # save metadata (texts)
    with open(meta_path, 'wb') as f:
        pickle.dump({'texts': texts}, f)
    print(f'Saved index -> {index_path}, metadata -> {meta_path}')


def load_index(index_path: str):
    return faiss.read_index(index_path)


def load_metadata(meta_path: str):
    with open(meta_path, 'rb') as f:
        return pickle.load(f)
