from sentence_transformers import SentenceTransformer

model = SentenceTransformer(
    "paraphrase-multilingual-MiniLM-L12-v2",
    device="cpu",
    cache_folder="./models/embeddings"
)

print("Model downloaded and loaded successfully")
