from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODEL_NAME = "google/flan-t5-base"

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    cache_dir="./models/llm"
)

model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_NAME,
    device_map="cpu",
    cache_dir="./models/llm"
)

print("FLAN-T5-base downloaded and loaded successfully")
