import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


class Generator:
    def __init__(self, model_name: str, cache_dir: str = "./models/llm"):
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=cache_dir
        )

        # Load model (CPU only, NO device_map to avoid accelerate issues)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            dtype=torch.float32
        )


        self.model.eval()  # inference mode

    def generate(self, query, contexts, max_new_tokens: int = 128, max_length: int = None):
        # 🔹 Limit number of contexts (VERY important)
        contexts = contexts[:3]

        # Build prompt
        prompt = (
            "Answer ONLY using the given context. "
            "If the answer is present, extract it. "
            "If unsure, give a short factual answer.\n\n"
            "Context:\n" + "\n---\n".join(contexts) +
            f"\n\nQuestion: {query}\nAnswer:"
        )



        # Tokenize with truncation (CRITICAL FIX)
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )

        # Accept legacy `max_length` parameter from callers
        if max_length is not None:
            max_new_tokens = int(max_length)

        # Generate (CPU-safe)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=1
            )
        text = self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

        # Fallback: try a sampling/beam strategy if nothing was produced
        if not text:
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    top_p=0.9,
                    temperature=0.7,
                    num_beams=4,
                    early_stopping=True
                )
            text = self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

        return text
