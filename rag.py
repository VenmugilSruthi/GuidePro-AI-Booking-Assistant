import os
import faiss
import numpy as np
from groq import Groq

EMB_MODEL = "text-embedding-3-small"  # Use a hosted embedding model

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def get_embedding(text):
    resp = client.embeddings.create(
        model=EMB_MODEL,
        input=text
    )
    return resp.data[0].embedding


class RAGStore:
    def __init__(self):
        self.index = faiss.IndexFlatL2(1536)   # dimension of "text-embedding-3-small"
        self.text_chunks = []

    def add_document(self, text):
        emb = np.array(get_embedding(text)).astype("float32")
        self.index.add(np.array([emb]))
        self.text_chunks.append(text)

    def search(self, query, top_k=3):
        q_emb = np.array(get_embedding(query)).astype("float32").reshape(1, -1)
        D, I = self.index.search(q_emb, top_k)
        return [self.text_chunks[i] for i in I[0]]
