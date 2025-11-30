import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

EMB_MODEL = "all-MiniLM-L6-v2"   # Local embedding model

model = SentenceTransformer(EMB_MODEL)


def get_embedding(text: str):
    """Return embedding as a numpy float32 array."""
    emb = model.encode([text])[0]
    return np.array(emb).astype("float32")


class RAGStore:
    def __init__(self):
        self.dim = 384   # all-MiniLM-L6-v2 output dimension
        self.index = faiss.IndexFlatL2(self.dim)
        self.text_chunks = []

    def add_document(self, text: str):
        emb = get_embedding(text).reshape(1, -1)
        self.index.add(emb)
        self.text_chunks.append(text)

    def add_documents(self, texts):
        for t in texts:
            self.add_document(str(t))

    def search(self, query, top_k=3):
        q_emb = get_embedding(query).reshape(1, -1)
        D, I = self.index.search(q_emb, top_k)
        return [self.text_chunks[i] for i in I[0] if i < len(self.text_chunks)]
