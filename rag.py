# rag.py

import numpy as np
import pdfplumber
import streamlit as st
from typing import List

# Groq + fallback local embeddings
try:
    from groq import Groq
except:
    Groq = None

from sentence_transformers import SentenceTransformer


# -------------------------------
# CONFIG
# -------------------------------
GROQ_MODEL = "nomic-embed-text"     # Groq embedding model
LOCAL_MODEL_NAME = "all-MiniLM-L6-v2"  # Fallback model


# -------------------------------
# Initialize Clients
# -------------------------------
groq_key = st.secrets.get("GROQ_API_KEY", None)
client = None

if groq_key and Groq:
    client = Groq(api_key=groq_key)

local_model = SentenceTransformer(LOCAL_MODEL_NAME)


# -------------------------------
# Embedding Function
# -------------------------------
def get_embedding(text: str) -> np.ndarray:
    text = text.strip()

    # ---- Try Groq first ----
    if client:
        try:
            response = client.embeddings.create(
                model=GROQ_MODEL,
                input=text
            )
            return np.array(response.data[0].embedding)
        except Exception as e:
            st.warning(
                f"Groq embedding failed: {e} — falling back to local embeddings."
            )

    # ---- Fallback: Local Model ----
    try:
        return local_model.encode(text)
    except Exception as e:
        st.error(f"Local embedding failed: {e}")
        raise RuntimeError("No embedding method available.")


# -------------------------------
# Chunk PDF Text
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
                text += page.extract_text() + "\n"
        return text.strip()
    except Exception as e:
        st.error(f"PDF reading error: {e}")
        return ""


# -------------------------------
# RAG Store Class
# -------------------------------
class RAGStore:
    def __init__(self):
        self.chunks = []
        self.embeddings = []

    def add_pdf(self, pdf_file):
        text = extract_pdf_text(pdf_file)
        if not text:
            st.error("PDF has no readable text.")
            return

        chunks = chunk_text(text)
        self.chunks.extend(chunks)

        # Embed all chunks
        for chunk in chunks:
            emb = get_embedding(chunk)
            self.embeddings.append(emb)

        st.success(f"PDF processed successfully! Loaded {len(chunks)} text chunks.")

    def query(self, question: str) -> str:
        if not self.embeddings:
            return "No documents uploaded for RAG."

        q_emb = get_embedding(question)

        # compute cosine similarity
        sims = [np.dot(q_emb, e) / (np.linalg.norm(q_emb) * np.linalg.norm(e)) for e in self.embeddings]

        best_idx = int(np.argmax(sims))
        best_chunk = self.chunks[best_idx]

        return best_chunk
