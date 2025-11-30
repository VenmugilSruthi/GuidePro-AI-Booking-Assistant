# rag.py

import numpy as np
import pdfplumber
import streamlit as st
from sentence_transformers import SentenceTransformer
from typing import List

# -------------------------------
# Local Embedding Model
# -------------------------------
@st.cache_resource
def load_local_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

local_model = load_local_model()


def get_embedding(text: str) -> np.ndarray:
    """Generate embeddings using local SentenceTransformer."""
    try:
        return local_model.encode(text)
    except Exception as e:
        st.error(f"Local embedding error: {e}")
        return np.zeros(384)


# -------------------------------
# Chunk Text
# -------------------------------
def chunk_text(text: str, chunk_size: int = 300) -> List[str]:
    words = text.split()
    return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]


# -------------------------------
# Extract PDF text
# -------------------------------
def extract_pdf_text(pdf_file) -> str:
    try:
        with pdfplumber.open(pdf_file) as pdf:
            text = ""
            for page in pdf.pages:
                t = page.extract_text()
                if t:
                    text += t + "\n"
        return text.strip()
    except Exception as e:
        st.error(f"PDF read error: {e}")
        return ""


# -------------------------------
# RAG STORE
# -------------------------------
class RAGStore:
    def __init__(self):
        self.chunks = []
        self.embeddings = []

    def add_pdf(self, pdf_file):
        """Load PDF → extract text → chunk → embed."""
        text = extract_pdf_text(pdf_file)
        if not text:
            st.error("PDF contains no readable text.")
            return

        chunks = chunk_text(text)
        self.chunks = chunks
        self.embeddings = []

        for chunk in chunks:
            emb = get_embedding(chunk)
            self.embeddings.append(emb)

        st.success(f"PDF processed — {len(chunks)} chunks added.")

    def query(self, question: str) -> str:
        """Return the most relevant PDF chunk."""
        if not self.embeddings:
            return "No PDF uploaded yet."

        q_emb = get_embedding(question)

        sims = [
            np.dot(q_emb, e) / (np.linalg.norm(q_emb) * np.linalg.norm(e))
            for e in self.embeddings
        ]

        best_idx = int(np.argmax(sims))
        return self.chunks[best_idx]
