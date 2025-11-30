import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

EMB_MODEL = "all-MiniLM-L6-v2"   # local embedding model

class RAGStore:
    def __init__(self):
        self.model = SentenceTransformer(EMB_MODEL)
        self.index = faiss.IndexFlatL2(384)   # MiniLM output dimension
        self.text_chunks = []

    def embed(self, text):
        emb = self.model.encode(text, convert_to_numpy=True)
        emb = emb.astype("float32")
        return emb

    def add_document(self, text):
        emb = self.embed(text)
        self.index.add(np.array([emb]))
        self.text_chunks.append(text)

    def add_documents(self, docs):
        for d in docs:
            self.add_document(str(d))

    def query(self, query, top_k=3):
        emb = self.embed(query).reshape(1, -1)
        D, I = self.index.search(emb, top_k)

        results = []
        for idx in I[0]:
            if idx < len(self.text_chunks):
                results.append(self.text_chunks[idx])

        if not results:
            return "No relevant information found."

        return "\n\n".join(results)
