import unicodedata
import re


def normalize_text(text: str) -> str:
    # Unicode normalization
    t = unicodedata.normalize('NFC', text)
    # collapse whitespace
    t = re.sub(r"\s+", ' ', t)
    # trim
    t = t.strip()
    return t


def simple_sanskrit_cleanup(text: str) -> str:
    # Placeholder for Sanskrit-specific normalization steps.
    # Could add: remove page numbers, headers, common OCR errors, transliteration rules etc.
    t = normalize_text(text)
    # remove repeated punctuation
    t = re.sub(r'[-]{2,}', '-', t)
    return t
