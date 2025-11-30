import numpy as np
import faiss
import pdfplumber
from groq import Groq
import os

# Groq embedding model
EMB_MODEL = "text-embedding-3-small"
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def embed(text):
    resp = client.embeddings.create(
        model=EMB_MODEL,
        input=text
    )
    return np.array(resp.data[0].embedding).astype("float32")

class RAGStore:
    def __init__(self):
        self.index = faiss.IndexFlatL2(1536)   # Groq embedding dimension
        self.text_chunks = []
        self.ready = False

    def add_documents(self, uploaded_files):
        for file in uploaded_files:
            with pdfplumber.open(file) as pdf:
                text = ""
                for page in pdf.pages:
                    t = page.extract_text()
                    if t:
                        text += t + "\n"

        chunks = self.chunk_text(text)

        for ch in chunks:
            vector = embed(ch)
            self.index.add(np.array([vector]))
            self.text_chunks.append(ch)

        self.ready = True

    def chunk_text(self, text, size=500):
        words = text.split()
        return [" ".join(words[i:i+size]) for i in range(0, len(words), size)]

    def query(self, question, top_k=3):
        if not self.ready:
            return "No documents indexed yet."

        q = embed(question).reshape(1, -1)
        D, I = self.index.search(q, top_k)
        results = [self.text_chunks[i] for i in I[0]]
        return "\n\n".join(results)
