import numpy as np
import faiss
import pdfplumber

# Very lightweight local embedding — NO Groq, NO HF
def embed(text):
    text = text.lower()
    vec = np.zeros(300, dtype="float32")
    for word in text.split():
        for i, ch in enumerate(word[:30]):
            vec[i % 300] += ord(ch)
    return vec / np.linalg.norm(vec)


class RAGStore:
    def __init__(self):
        self.index = faiss.IndexFlatL2(300)
        self.chunks = []

    def _extract_text_from_pdf(self, file):
        text = ""
        try:
            with pdfplumber.open(file) as pdf:
                for page in pdf.pages:
                    text += page.extract_text() + "\n"
        except:
            pass
        return text

    def _split_chunks(self, text, size=350):
        words = text.split()
        chunks = []
        for i in range(0, len(words), size):
            chunks.append(" ".join(words[i:i+size]))
        return chunks

    def add_documents(self, files):
        for f in files:
            pdf_text = self._extract_text_from_pdf(f)
            chunks = self._split_chunks(pdf_text)

            for c in chunks:
                emb = embed(c)
                self.index.add(np.array([emb]))
                self.chunks.append(c)

    def query(self, question, top_k=1):
        q_emb = embed(question).reshape(1, -1)
        D, I = self.index.search(q_emb, top_k)
        return self.chunks[I[0][0]]
