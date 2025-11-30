# rag.py
import numpy as np
import pdfplumber
import streamlit as st
from groq import Groq
from typing import List

# Embedding model
EMB_MODEL = "text-embedding-3-small"

# Load API key
client = Groq(api_key=st.secrets["GROQ_API_KEY"])


# -----------------------------------------------------
# SAFE embedding function (single text only)
# -----------------------------------------------------
def get_embedding_safe(text: str):
    text = text.strip()
    if not text:
        return None

    # HARD LIMIT - Groq fails if too large
    text = text[:1500]

    try:
        resp = client.embeddings.create(
            model=EMB_MODEL,
            input=text
        )
        emb = np.array(resp.data[0].embedding, dtype="float32")
        return emb

    except Exception as e:
        print("Embedding error:", e)
        return None


# -----------------------------------------------------
# Cosine similarity
# -----------------------------------------------------
def cosine_sim(a, b):
    if a is None or b is None:
        return 0.0
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


# -----------------------------------------------------
# RAG Store
# -----------------------------------------------------
class RAGStore:
    def __init__(self):
        self.embeddings: List[np.ndarray] = []
        self.text_chunks: List[str] = []

    def add_documents(self, uploaded_files):
        if uploaded_files is None:
            return

        if not isinstance(uploaded_files, list):
            uploaded_files = [uploaded_files]

        for f in uploaded_files:
            try:
                with pdfplumber.open(f) as pdf:
                    pages = []
                    for page in pdf.pages:
                        txt = page.extract_text()
                        if txt:
                            pages.append(txt)
                full_text = "\n".join(pages).strip()

            except Exception as e:
                print("PDF read error:", e)
                continue

            if not full_text:
                continue

            chunks = self.chunk_text(full_text)

            for ch in chunks:
                emb = get_embedding_safe(ch)
                if emb is not None:
                    self.embeddings.append(emb)
                    self.text_chunks.append(ch)

    def chunk_text(self, text: str):
        words = text.split()
        chunks = []

        size = 150  # small chunks

        for i in range(0, len(words), size):
            chunk = " ".join(words[i:i + size]).strip()
            if chunk:
                chunk = chunk[:1500]
                chunks.append(chunk)

        return chunks

    def query(self, question: str, top_k: int = 3):
        if not self.embeddings:
            return "No documents indexed yet. Upload a PDF to enable RAG."

        q_emb = get_embedding_safe(question)
        if q_emb is None:
            return "Unable to create embedding for your question."

        sims = [cosine_sim(q_emb, e) for e in self.embeddings]
        idxs = np.argsort(sims)[::-1][:top_k]

        results = [self.text_chunks[i] for i in idxs]

        return "\n\n---\n\n".join(results)
