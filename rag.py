# rag.py

import numpy as np
import pdfplumber
import streamlit as st
from typing import List
from groq import Groq

# Groq embedding model
EMB_MODEL = "nomic-embed-text-v1.5"


# Initialize Groq Client
groq_key = st.secrets.get("GROQ_API_KEY", None)
if not groq_key:
    st.error("❌ Missing GROQ_API_KEY in Streamlit Secrets.")

client = Groq(api_key=groq_key)


# -------------------------------
# Embedding Function (Groq Only)
# -------------------------------
def get_embedding(text: str) -> np.ndarray:
    try:
        res = client.embeddings.create(
            model=EMB_MODEL,
            input=text
        )
        return np.array(res.data[0].embedding)
    except Exception as e:
        st.error(f"Embedding error: {e}")
        return np.zeros(768)


# -------------------------------
# Chunk Text
# -------------------------------
def chunk_text(text: str, chunk_size: int = 300) -> List[str]:
    words = text.split()
    return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]


# -------------------------------
# Extract text from PDF
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
        if not self.embeddings:
            return "No PDF loaded yet."

        q_emb = get_embedding(question)

        sims = [
            np.dot(q_emb, e) / (np.linalg.norm(q_emb) * np.linalg.norm(e))
            for e in self.embeddings
        ]

        best_idx = int(np.argmax(sims))
        return self.chunks[best_idx]

