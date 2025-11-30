# rag.py
import os
import numpy as np
import pdfplumber
from groq import Groq
from typing import List

# choose a hosted embedding model name that Groq supports
EMB_MODEL = "text-embedding-3-small"

# init client from env (Streamlit Secrets are loaded into environment automatically)
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def get_batch_embeddings(texts: List[str]) -> List[np.ndarray]:
    """
    Request embeddings for a list of texts from Groq.
    Returns list of np.float32 arrays (one per text).
    Raises RuntimeError with helpful message if the API returns an error.
    """
    # filter out any empty strings
    inputs = [t for t in texts if t and t.strip()]
    if len(inputs) == 0:
        return []

    try:
        # Groq accepts a list input to create multiple embeddings in one request
        resp = client.embeddings.create(model=EMB_MODEL, input=inputs)
    except Exception as e:
        # Raise a clearer error so Streamlit shows something actionable in the app logs
        raise RuntimeError(
            f"Groq embeddings call failed. Check your GROQ_API_KEY, network and model name ({EMB_MODEL}).\n"
            f"Original error: {e}"
        ) from e

    # map response to numpy arrays
    out = []
    try:
        for item in resp.data:
            out.append(np.array(item.embedding, dtype="float32"))
    except Exception as e:
        raise RuntimeError("Unexpected Groq response shape when creating embeddings.") from e

    return out

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    # a and b are 1-D arrays
    if a is None or b is None:
        return 0.0
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))

class RAGStore:
    def __init__(self):
        # store embeddings and text chunks in memory (small app / small PDFs fine)
        self.embeddings: List[np.ndarray] = []   # list of np arrays
        self.text_chunks: List[str] = []         # list of strings

    def add_documents(self, uploaded_files):
        """
        uploaded_files: list of streamlit UploadedFile or a single UploadedFile
        Each file is read with pdfplumber and chunked.
        """
        # support single file too
        if uploaded_files is None:
            return

        if not isinstance(uploaded_files, list):
            uploaded_files = [uploaded_files]

        # gather all chunks to embed in a batch-friendly manner
        all_chunks = []
        for f in uploaded_files:
            try:
                # pdfplumber accepts a file-like object; ensure we rewind just in case
                with pdfplumber.open(f) as pdf:
                    pages_text = []
                    for page in pdf.pages:
                        page_text = page.extract_text()
                        if page_text:
                            pages_text.append(page_text)
                full_text = "\n".join(pages_text).strip()
            except Exception as e:
                # skip problematic file but log the issue for debugging
                raise RuntimeError(f"Failed to read uploaded PDF file: {getattr(f, 'name', str(f))}. Error: {e}") from e

            if not full_text:
                # skip files with no text
                continue

            chunks = self.chunk_text(full_text, chunk_size=300)  # chunk_size in words
            all_chunks.extend(chunks)

        if not all_chunks:
            # nothing to embed
            return

        # Batch embeddings to reduce number of API calls. Adjust batch_size if needed.
        batch_size = 16
        for i in range(0, len(all_chunks), batch_size):
            batch = all_chunks[i:i + batch_size]
            embeddings = get_batch_embeddings(batch)
            # append to store
            for emb, ch in zip(embeddings, batch):
                self.embeddings.append(emb)
                self.text_chunks.append(ch)

    def chunk_text(self, text: str, chunk_size: int = 300) -> List[str]:
        """
        Chunk by words. chunk_size is number of words per chunk.
        Returns list of chunks (strings).
        """
        words = text.split()
        if not words:
            return []
        chunks = []
        for i in range(0, len(words), chunk_size):
            chunk = " ".join(words[i:i + chunk_size]).strip()
            if chunk:
                chunks.append(chunk)
        return chunks

    def query(self, question: str, top_k: int = 3) -> str:
        """
        Return a concise response: join top_k relevant chunks.
        If nothing indexed, return helpful message.
        """
        if not self.embeddings:
            return "No documents indexed yet. Upload a PDF in the sidebar to enable RAG."

        if not question or not question.strip():
            return "Please enter a question."

        # embed the question
        try:
            q_embs = get_batch_embeddings([question])
        except RuntimeError as e:
            # show a friendly message to the user
            return f"Error creating embeddings for the question: {e}"

        if not q_embs:
            return "Failed to create embedding for your question. Make sure your API key and model are configured."

        q_emb = q_embs[0]

        # compute similarities
        sims = [cosine_sim(q_emb, e) for e in self.embeddings]
        # get indices sorted by similarity desc
        idxs = np.argsort(sims)[::-1][:top_k]
        top_chunks = [self.text_chunks[int(i)] for i in idxs if i < len(self.text_chunks)]
        return "\n\n---\n\n".join(top_chunks)
