import os
import glob
from typing import List
from PyPDF2 import PdfReader
import pdfplumber


def load_pdf_text(path: str) -> str:
    text = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            t = page.extract_text()
            if t:
                text.append(t)
    return "\n".join(text)


def load_txt(path: str) -> str:
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()


def list_documents(data_dir: str) -> List[str]:
    paths = []
    for ext in ('*.pdf','*.txt'):
        paths.extend(glob.glob(os.path.join(data_dir, ext)))
    return paths


def load_all(data_dir: str) -> str:
    texts = []
    for p in list_documents(data_dir):
        if p.lower().endswith('.pdf'):
            texts.append(load_pdf_text(p))
        else:
            texts.append(load_txt(p))
    return "\n\n".join(texts)

