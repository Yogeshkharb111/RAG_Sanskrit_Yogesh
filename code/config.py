import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DIR = os.path.join(DATA_DIR, 'processed')
VSTORE_DIR = os.path.join(BASE_DIR, 'vectorstore')

# Models (change as needed)
EMBEDDING_MODEL = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
GENERATION_MODEL = 'google/flan-t5-base'
GENERATION_PIPE_TASK = 'text2text-generation'

# Index / chunk settings
CHUNK_SIZE = 600
CHUNK_OVERLAP = 100
TOP_K = 4

# Files
FAISS_INDEX_PATH = os.path.join(VSTORE_DIR, 'faiss.index')
METADATA_PATH = os.path.join(VSTORE_DIR, 'metadata.pkl')
CLEANED_TEXT_PATH = os.path.join(PROCESSED_DIR, 'cleaned_text.txt')
CHUNKS_JSON = os.path.join(PROCESSED_DIR, 'chunks.json')

# Ensure directories exist at runtime
for p in (RAW_DIR, PROCESSED_DIR, VSTORE_DIR):
    os.makedirs(p, exist_ok=True)
