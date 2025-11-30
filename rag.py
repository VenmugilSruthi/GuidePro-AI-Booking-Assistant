import os
import faiss
import numpy as np
import pdfplumber
from groq import Groq

EMB_MODEL = "text-embedding-3-small"
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# -----------------------------------------
# SAFE EMBEDDING FUNCTION
# -----------------------------------------
def get_embedding(text: str):
    text = text[:2000]   # prevent Groq NotFoundError
    resp = client.embeddings.create(
        model=EMB_MODEL,
        input=text
    )
    return resp.data[0].embedding


# -----------------------------------------
# PDF TEXT CLEANER & CHUNKER
# -----------------------------------------
def chunk_text(text, chunk_size=400):
    words = text.split()
    chunks = []
    cur = []

    for w in words:
        cur.append(w)
        if len(cur) >= chunk_size:
            chunks.append(" ".join(cur))
            cur = []

    if cur:
        chunks.append(" ".join(cur))

    return chunks


# -----------------------------------------
# RAG CLASS
# -----------------------------------------
class RAGStore:
    def __init__(self):
        self.index = faiss.IndexFlatL2(1536)
        self.text_chunks = []

    def add_documents(self, files):
        for f in files:
            with pdfplumber.open(f) as pdf:
                full_text = ""
                for page in pdf.pages:
                    full_text += page.extract_text() or ""

            # break into smaller chunks (prevents duplicate output)
            chunks = chunk_text(full_text, chunk_size=120)

            for ch in chunks:
                emb = np.array(get_embedding(ch)).astype("float32")
                self.index.add(np.array([emb]))
                self.text_chunks.append(ch)

    def query(self, question, top_k=3):
        q_emb = np.array(get_embedding(question)).astype("float32").reshape(1, -1)
        D, I = self.index.search(q_emb, top_k)

        results = [self.text_chunks[i] for i in I[0]]
        joined = "\n".join(results)

        return joined if joined.strip() else "No relevant information found."
