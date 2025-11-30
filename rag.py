# rag.py
import os
import numpy as np
import pdfplumber
from groq import Groq

# choose a hosted embedding model name that Groq supports
EMB_MODEL = "text-embedding-3-small"

# init client from env (Streamlit Secrets are loaded into environment automatically)
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def get_embedding(text: str):
    # call groq embeddings endpoint, return list/np array
    resp = client.embeddings.create(model=EMB_MODEL, input=text)
    return np.array(resp.data[0].embedding, dtype="float32")

def cosine_sim(a: np.ndarray, b: np.ndarray):
    # a and b are 1-D arrays
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0.0
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

class RAGStore:
    def __init__(self):
        # store embeddings and text chunks in memory (small app / small PDFs fine)
        self.embeddings = []   # list of np arrays
        self.text_chunks = []  # list of strings

    def add_documents(self, uploaded_files):
        """
        uploaded_files: list of streamlit UploadedFile or a single UploadedFile
        Each file is read with pdfplumber and chunked.
        """
        # support single file too
        if not isinstance(uploaded_files, list):
            uploaded_files = [uploaded_files]

        for f in uploaded_files:
            # streamlit UploadedFile is file-like, pdfplumber accepts a file object
            with pdfplumber.open(f) as pdf:
                text = []
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text.append(page_text)
                full_text = "\n".join(text)

            # chunk the text (split into pieces)
            chunks = self.chunk_text(full_text, chunk_size=300)  # ~300 words per chunk
            for ch in chunks:
                emb = get_embedding(ch)
                self.embeddings.append(emb)
                self.text_chunks.append(ch)

    def chunk_text(self, text, chunk_size=300):
        words = text.split()
        chunks = []
        for i in range(0, len(words), chunk_size):
            chunk = " ".join(words[i:i + chunk_size])
            if chunk.strip():
                chunks.append(chunk)
        return chunks

    def query(self, question: str, top_k: int = 3):
        """
        Return a concise response: join top 1 (or top_k) relevant chunks.
        If nothing indexed, return helpful message.
        """
        if len(self.embeddings) == 0:
            return "No documents indexed yet. Upload a PDF in the sidebar to enable RAG."

        q_emb = get_embedding(question)
        # compute similarities
        sims = [cosine_sim(q_emb, e) for e in self.embeddings]
        # get indices sorted by similarity desc
        idxs = np.argsort(sims)[::-1][:top_k]
        top_chunks = [self.text_chunks[i] for i in idxs]
        # return only top chunk to avoid repeating whole PDF; join if top_k>1
        return "\n\n---\n\n".join(top_chunks)
