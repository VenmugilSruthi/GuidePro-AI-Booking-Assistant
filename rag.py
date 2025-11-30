import numpy as np
import faiss
import pdfplumber
from PyPDF2 import PdfReader

EMB_SIZE = 384  # dimensionality for MiniLM embeddings


def extract_text_from_pdf(file):
    """Extract PDF text safely using pdfplumber OR PyPDF2 fallback."""
    try:
        with pdfplumber.open(file) as pdf:
            pages = [page.extract_text() or "" for page in pdf.pages]
            return "\n".join(pages)
    except:
        reader = PdfReader(file)
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n".join(pages)


def simple_embedding(text: str):
    """
    Fake embedding (384D) so Streamlit never crashes.
    Replace later with Groq/OpenAI embeddings if needed.
    """
    arr = np.random.rand(EMB_SIZE).astype("float32")
    return arr


class RAGStore:
    def __init__(self):
        self.index = faiss.IndexFlatL2(EMB_SIZE)
        self.docs = []

    def add_documents(self, uploaded_files):
        """Accept list of UploadedFile objects → extract text → index."""
        for file in uploaded_files:
            text = extract_text_from_pdf(file)
            if not text.strip():
                continue

            emb = simple_embedding(text)
            self.index.add(np.array([emb]))
            self.docs.append(text)

    def query(self, query_text, top_k=1):
        """Return best‐matching document snippet."""
        if len(self.docs) == 0:
            return "No relevant information found in uploaded PDFs."

        q_emb = simple_embedding(query_text).reshape(1, -1)
        D, I = self.index.search(q_emb, top_k)
        best_idx = I[0][0]
        return self.docs[best_idx][:1000]  # return top snippet
