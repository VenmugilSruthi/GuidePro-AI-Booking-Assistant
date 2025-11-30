import numpy as np
import faiss
import pdfplumber
from groq import Groq
import os

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
EMB_MODEL = "text-embedding-3-small"


# ---------- EMBEDDING FUNCTION ----------
def get_embedding(text):
    r = client.embeddings.create(
        model=EMB_MODEL,
        input=text
    )
    return np.array(r.data[0].embedding, dtype="float32")


# ---------- CHUNKING ----------
def chunk_text(text, max_len=350):
    words = text.split()
    chunks = []
    cur = []

    for w in words:
        cur.append(w)
        if len(cur) >= max_len:
            chunks.append(" ".join(cur))
            cur = []

    if cur:
        chunks.append(" ".join(cur))

    return chunks


# ---------- RAG STORE ----------
class RAGStore:
    def __init__(self):
        self.index = faiss.IndexFlatL2(1536)
        self.chunks = []

    def add_documents(self, uploaded_files):

        for f in uploaded_files:
            with pdfplumber.open(f) as pdf:
                full = ""
                for page in pdf.pages:
                    full += page.extract_text() + "\n"

            # Split into chunks
            parts = chunk_text(full)

            for p in parts:
                emb = get_embedding(p)
                self.index.add(np.array([emb]))
                self.chunks.append(p)

    def query(self, question, top_k=2):
        q_emb = get_embedding(question).reshape(1, -1)
        _, idx = self.index.search(q_emb, top_k)

        results = [self.chunks[i] for i in idx[0] if i < len(self.chunks)]

        if not results:
            return "No relevant information found."

        return "\n".join(results)
