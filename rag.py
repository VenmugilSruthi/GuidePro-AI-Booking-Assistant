# rag.py
import os
import faiss
import numpy as np
from groq import Groq

# Hosted embedding model (works on Streamlit Cloud)
EMB_MODEL = "text-embedding-3-small"

client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def get_embedding(text: str):
    """Generate 1536-dim embedding using Groq API."""
    resp = client.embeddings.create(
        model=EMB_MODEL,
        input=text
    )
    return resp.data[0].embedding


class RAGStore:
    def __init__(self):
        # text-embedding-3-small → 1536 dim vector
        self.index = faiss.IndexFlatL2(1536)
        self.text_chunks = []

    # ----------------------------
    # SUPPORT BOTH old & new calls
    # ----------------------------
    def add_document(self, text: str):
        """Add a single text chunk."""
        emb = np.array(get_embedding(text)).astype("float32")
        self.index.add(np.array([emb]))
        self.text_chunks.append(text)

    def add_documents(self, docs):
        """App compatibility: supports uploading multiple PDFs."""
        for d in docs:
            if isinstance(d, str):
                self.add_document(d)
            else:
                # Ensure bytes or PDFs converted to text before this
                self.add_document(str(d))

    # ----------------------------
    # Search
    # ----------------------------
    def search(self, query: str, top_k: int = 3):
        """Retrieve most relevant chunks."""
        q_emb = np.array(get_embedding(query)).astype("float32").reshape(1, -1)
        D, I = self.index.search(q_emb, top_k)
        return [self.text_chunks[i] for i in I[0] if i < len(self.text_chunks)]
