import numpy as np
import faiss
import pdfplumber
from sentence_transformers import SentenceTransformer

# Use MiniLM embedding model (runs locally)
EMB_MODEL = "all-MiniLM-L6-v2"
model = SentenceTransformer(EMB_MODEL)

def embed(text):
    return model.encode([text])[0]

class RAGStore:
    def __init__(self):
        self.index = faiss.IndexFlatL2(384)   # MiniLM = 384 dims
        self.text_chunks = []

    def add_documents(self, uploaded_files):
        for file in uploaded_files:
            with pdfplumber.open(file) as pdf:
                text = ""
                for page in pdf.pages:
                    text += page.extract_text() + "\n"

            chunks = self.chunk_text(text)
            for ch in chunks:
                vector = embed(ch).astype("float32")
                self.index.add(np.array([vector]))
                self.text_chunks.append(ch)

    def chunk_text(self, text, chunk_size=500):
        words = text.split()
        chunks = []
        for i in range(0, len(words), chunk_size):
            chunk = " ".join(words[i:i + chunk_size])
            chunks.append(chunk)
        return chunks

    def query(self, question, top_k=3):
        q_emb = embed(question).astype("float32").reshape(1, -1)
        D, I = self.index.search(q_emb, top_k)

        if len(I) == 0:
            return "No relevant information found."

        results = [self.text_chunks[i] for i in I[0]]
        return "\n\n".join(results)
