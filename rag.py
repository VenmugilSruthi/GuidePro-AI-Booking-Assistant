import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

EMB_MODEL = "all-MiniLM-L6-v2"   # local embedding model (384 dim)

class RAGStore:
    def __init__(self):
        self.model = SentenceTransformer(EMB_MODEL)
        self.index = faiss.IndexFlatL2(384)  # embedding dimension = 384
        self.text_chunks = []

    def get_embedding(self, text: str):
        emb = self.model.encode([text], convert_to_numpy=True)
        return emb[0]

    def add_document(self, text: str):
        emb = self.get_embedding(text).astype("float32")
        self.index.add(np.array([emb]))
        self.text_chunks.append(text)

    def add_documents(self, docs):
        for d in docs:
            self.add_document(str(d))

    def search(self, query, top_k=3):
        q_emb = self.get_embedding(query).astype("float32").reshape(1, -1)
        D, I = self.index.search(q_emb, top_k)
        return [self.text_chunks[i] for i in I[0]]
