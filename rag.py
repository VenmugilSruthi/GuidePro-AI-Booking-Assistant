import numpy as np
import faiss
from groq import Groq
import pdfplumber
import io
import os

# Groq embedding model
EMB_MODEL = "text-embedding-3-small"
client = Groq(api_key=os.getenv("GROQ_API_KEY"))


# -----------------------------
# Helper: Extract text from PDF
# -----------------------------
def extract_pdf_text(uploaded_file):
    text = ""
    with pdfplumber.open(io.BytesIO(uploaded_file.read())) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    uploaded_file.seek(0)   # Reset cursor for reuse
    return text.strip()


# -----------------------------
# Helper: Create embedding
# -----------------------------
def get_embedding(text):
    resp = client.embeddings.create(
        model=EMB_MODEL,
        input=text
    )
    return resp.data[0].embedding


# -----------------------------
# RAG STORE CLASS
# -----------------------------
class RAGStore:
    def __init__(self):
        self.emb_dim = 1536
        self.index = faiss.IndexFlatL2(self.emb_dim)
        self.text_chunks = []

    def add_document(self, text):
        emb = np.array(get_embedding(text)).astype("float32")
        self.index.add(np.array([emb]))
        self.text_chunks.append(text)

    def add_documents(self, uploaded_files):
        for f in uploaded_files:
            pdf_text = extract_pdf_text(f)

            if not pdf_text:
                continue

            # Split into smaller chunks for better recall
            chunks = [pdf_text[i:i+700] for i in range(0, len(pdf_text), 700)]

            for ch in chunks:
                self.add_document(ch)

    def query(self, question, top_k=3):
        q_emb = np.array(get_embedding(question)).astype("float32").reshape(1, -1)
        D, I = self.index.search(q_emb, top_k)

        results = [self.text_chunks[i] for i in I[0] if i < len(self.text_chunks)]

        if not results:
            return "No relevant information found in the documents."

        return "\n\n---\n\n".join(results)
