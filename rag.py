import os
import faiss
import numpy as np
from groq import Groq

# Groq embedding model
EMB_MODEL = "text-embedding-3-small"

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def get_embedding(text):
    """Returns embedding vector from Groq API"""
    resp = client.embeddings.create(
        model=EMB_MODEL,
        input=text
    )
    return resp.data[0].embedding


class RAGStore:
    def __init__(self):
        # Dimension of Groq embedding (1536)
        self.index = faiss.IndexFlatL2(1536)
        self.text_chunks = []

    def add_document(self, text):
        """Add a single text chunk"""
        emb = np.array(get_embedding(text)).astype("float32")
        self.index.add(np.array([emb]))
        self.text_chunks.append(text)

    def add_documents(self, list_of_texts):
        """Add list of text chunks"""
        for t in list_of_texts:
            self.add_document(str(t))

    def search(self, query, top_k=3):
        """Search relevant chunks"""
        q_emb = np.array(get_embedding(query)).astype("float32").reshape(1, -1)
        D, I = self.index.search(q_emb, top_k)
        return [self.text_chunks[i] for i in I[0]]
