import os
import faiss
import numpy as np
from groq import Groq
import pdfplumber

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

EMB_MODEL = "text-embedding-3-small"

def get_embedding(text):
    resp = client.embeddings.create(
        model=EMB_MODEL,
        input=text
    )
    return resp.data[0].embedding


class RAGStore:
    def __init__(self):
        self.index = faiss.IndexFlatL2(1536)
        self.text_chunks = []

    def extract_text(self, file):
        text = ""
        with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                text += page.extract_text() + "\n"
        return text

    def add_documents(self, pdf_files):
        for file in pdf_files:
            content = self.extract_text(file)
            chunks = content.split("\n")
            for c in chunks:
                if len(c.strip()) < 5:
                    continue
                emb = np.array(get_embedding(c)).astype("float32")
                self.index.add(np.array([emb]))
                self.text_chunks.append(c)

    def query(self, question):
        emb = np.array(get_embedding(question)).astype("float32").reshape(1, -1)
        D, I = self.index.search(emb, 3)
        results = [self.text_chunks[i] for i in I[0]]
        return "\n\n".join(results)
