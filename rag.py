import numpy as np
import faiss
import pdfplumber
from groq import Groq
import os

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
EMB_MODEL = "text-embedding-3-small"

# ---------------------------------------------
# Get embedding from Groq
# ---------------------------------------------
def get_embedding(text):
    resp = client.embeddings.create(
        model=EMB_MODEL,
        input=text
    )
    return resp.data[0].embedding


# ---------------------------------------------
# Chunk text for better RAG accuracy
# ---------------------------------------------
def chunk_text(text, max_len=400):
    words = text.split()
    chunks = []
    current = []

    for w in words:
        current.append(w)
        if len(current) >= max_len:
            chunks.append(" ".join(current))
            current = []

    if current:
        chunks.append(" ".join(current))
    return chunks


# ---------------------------------------------
# RAG Store Class
# ---------------------------------------------
class RAGStore:

    def __init__(self):
        self.dimension = 1536
        self.index = faiss.IndexFlatL2(self.dimension)
        self.chunks = []

    # -----------------------------------------
    # Read & store uploaded PDFs
    # -----------------------------------------
    def add_documents(self, files):
        for f in files:
            with pdfplumber.open(f) as pdf:
                full_text = ""
                for page in pdf.pages:
                    full_text += page.extract_text() + "\n"

            # Create chunks
            split_chunks = chunk_text(full_text)

            # Add embeddings
            for c in split_chunks:
                emb = np.array(get_embedding(c)).astype("float32")
                self.index.add(np.array([emb]))
                self.chunks.append(c)

    # -----------------------------------------
    # Search for relevant text
    # -----------------------------------------
    def query(self, question, top_k=2):
        q_emb = np.array(get_embedding(question)).astype("float32").reshape(1, -1)
        D, I = self.index.search(q_emb, top_k)

        results = [self.chunks[i] for i in I[0] if i < len(self.chunks)]
        if not results:
            return "No relevant information found."

        return "\n\n".join(results)
