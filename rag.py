# rag.py
import numpy as np
import pdfplumber
from groq import Groq
import streamlit as st
from typing import List


# Embedding model (Groq)
EMB_MODEL = "text-embedding-3-small"

# Load API key from Streamlit Secrets (correct way)
client = Groq(api_key=st.secrets["GROQ_API_KEY"])


# ---------------------------------------------------------
# Embeddings
# ---------------------------------------------------------
def get_batch_embeddings(texts: List[str]) -> List[np.ndarray]:
    """
    Creates embeddings for a batch of text chunks.
    Returns list of numpy arrays.
    Raises clean RuntimeError for Streamlit UI.
    """

    # Remove empty strings
    inputs = [t for t in texts if t and t.strip()]
    if not inputs:
        return []

    try:
        resp = client.embeddings.create(
            model=EMB_MODEL,
            input=inputs
        )
    except Exception as e:
        raise RuntimeError(
            f"Groq embedding failed. Check your GROQ_API_KEY and model name.\n\nError: {e}"
        ) from e

    out = []
    for item in resp.data:
        out.append(np.array(item.embedding, dtype="float32"))

    return out


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    if a is None or b is None:
        return 0.0
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


# ---------------------------------------------------------
# RAG Store
# ---------------------------------------------------------
class RAGStore:
    def __init__(self):
        self.embeddings: List[np.ndarray] = []
        self.text_chunks: List[str] = []

    # -------------------------------
    # ADD PDF DOCUMENTS
    # -------------------------------
    def add_documents(self, uploaded_files):
        if uploaded_files is None:
            return

        if not isinstance(uploaded_files, list):
            uploaded_files = [uploaded_files]

        all_chunks = []

        for f in uploaded_files:
            try:
                with pdfplumber.open(f) as pdf:
                    pages_text = []
                    for page in pdf.pages:
                        txt = page.extract_text()
                        if txt:
                            pages_text.append(txt)
                full_text = "\n".join(pages_text).strip()

            except Exception as e:
                raise RuntimeError(
                    f"Failed to read PDF file: {getattr(f, 'name', str(f))}\nError: {e}"
                ) from e

            if not full_text:
                # PDF has no extractable text (images only)
                continue

            # chunk the PDF text
            chunks = self.chunk_text(full_text, chunk_size=300)
            all_chunks.extend(chunks)

        if not all_chunks:
            return  # nothing to embed

        # Batch embed chunks to reduce calls
        batch_size = 16
        for i in range(0, len(all_chunks), batch_size):
            batch = all_chunks[i:i + batch_size]
            embeddings = get_batch_embeddings(batch)

            for emb, ch in zip(embeddings, batch):
                self.embeddings.append(emb)
                self.text_chunks.append(ch)

    # -------------------------------
    # TEXT CHUNKING
    # -------------------------------
    def chunk_text(self, text: str, chunk_size: int = 300) -> List[str]:
        words = text.split()
        out = []
        for i in range(0, len(words), chunk_size):
            chunk = " ".join(words[i:i + chunk_size]).strip()
            if chunk:
                out.append(chunk)
        return out

    # -------------------------------
    # QUERY
    # -------------------------------
    def query(self, question: str, top_k: int = 3) -> str:
        if not self.embeddings:
            return "No documents indexed yet. Upload a PDF in the sidebar to enable RAG."

        if not question.strip():
            return "Please type a question."

        # Embed the question
        try:
            q_emb = get_batch_embeddings([question])[0]
        except Exception as e:
            return f"Embedding error: {e}"

        # Calculate similarities
        sims = [cosine_sim(q_emb, emb) for emb in self.embeddings]
        idxs = np.argsort(sims)[::-1][:top_k]

        results = [self.text_chunks[i] for i in idxs]

        return "\n\n---\n\n".join(results)
