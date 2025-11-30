# rag.py
import os
import pickle
import math
import pdfplumber
import numpy as np
import streamlit as st

# try faiss import (faiss-cpu)
try:
    import faiss
except Exception as e:
    faiss = None
    st.warning("faiss not available: RAG will not work without faiss. Install faiss-cpu.")

# Try to import Groq client (optional)
try:
    from groq import Groq
except Exception:
    Groq = None

# Try sentence-transformers fallback
try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

# Defaults / files
INDEX_PATH = "rag_index.faiss"
DOCS_PATH = "rag_docs.pkl"
EMB_MODEL_DEFAULT = "all-MiniLM-L6-v2"  # for sentence-transformers fallback
GROQ_EMB_MODEL = "text-embedding-3-small"  # if using Groq embeddings


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 100):
    """
    Chunk long text into overlapping pieces (by characters).
    chunk_size and overlap are in characters for simplicity.
    """
    if not text:
        return []
    chunks = []
    start = 0
    length = len(text)
    while start < length:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk.strip())
        start = max(end - overlap, end)
    return [c for c in chunks if c]


class RAGStore:
    """
    RAG store using FAISS for retrieval. Supports embedding via Groq (if configured)
    or sentence-transformers fallback.
    """

    def __init__(self):
        self.embeddings = None  # np.ndarray of shape (n, d)
        self.docs = []  # list of dicts: {"text":..., "meta": {...}}
        self.index = None
        self.dim = None

        # Try load existing index + docs
        if os.path.exists(DOCS_PATH) and faiss is not None and os.path.exists(INDEX_PATH):
            try:
                with open(DOCS_PATH, "rb") as f:
                    self.docs = pickle.load(f)
                self.index = faiss.read_index(INDEX_PATH)
                # infer dim
                self.dim = self.index.d
                # Recreate embeddings array if needed
                n = self.index.ntotal
                if n > 0:
                    # we can't extract vectors directly from IndexFlatIP,
                    # but we can store embeddings separately next time. For now keep docs only.
                    pass
                st.info("RAG index loaded from disk.")
            except Exception as e:
                st.warning(f"Failed to load existing RAG index: {e}")

        # Initialize embedding clients lazily
        self._groq_client = None
        self._st_model = None

    def _init_groq(self):
        if self._groq_client is not None:
            return self._groq_client
        if Groq is None:
            return None
        key = st.secrets.get("GROQ_API_KEY", None)
        if not key:
            return None
        try:
            self._groq_client = Groq(api_key=key)
            return self._groq_client
        except Exception:
            self._groq_client = None
            return None

    def _init_sentence_transformer(self):
        if self._st_model is not None:
            return self._st_model
        if SentenceTransformer is None:
            return None
        try:
            self._st_model = SentenceTransformer(EMB_MODEL_DEFAULT)
            return self._st_model
        except Exception:
            self._st_model = None
            return None

    def _get_embeddings(self, texts):
        """
        Return numpy array of shape (len(texts), dim).
        Tries Groq embeddings if key present; else falls back to sentence-transformers.
        """
        # Try Groq
        groq = self._init_groq()
        if groq is not None:
            try:
                # Groq embeddings API uses a list of strings
                model = st.secrets.get("EMB_MODEL", GROQ_EMB_MODEL)
                resp = groq.embeddings.create(
                    model=model,
                    input=texts
                )
                vectors = [r["embedding"] for r in resp["data"]]
                arr = np.array(vectors, dtype=np.float32)
                # normalize for inner-product similarity
                norms = np.linalg.norm(arr, axis=1, keepdims=True)
                norms[norms == 0] = 1.0
                arr = arr / norms
                return arr
            except Exception as e:
                st.warning(f"Groq embeddings failed: {e} — falling back to local model.")

        # Fallback: sentence-transformers
        st_model = self._init_sentence_transformer()
        if st_model is not None:
            try:
                arr = st_model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
                arr = arr.astype(np.float32)
                # normalize
                norms = np.linalg.norm(arr, axis=1, keepdims=True)
                norms[norms == 0] = 1.0
                arr = arr / norms
                return arr
            except Exception as e:
                st.error(f"Local sentence-transformers embedding failed: {e}")

        raise RuntimeError("No embedding method available. Provide GROQ_API_KEY in secrets or install sentence-transformers.")

    def _ensure_index(self, dim):
        if faiss is None:
            raise RuntimeError("faiss is required but not installed.")
        if self.index is None:
            # use inner-product search on normalized vectors (cosine via dot product)
            self.index = faiss.IndexFlatIP(dim)
            self.dim = dim

    def add_documents(self, uploaded_files):
        """
        uploaded_files: list of Streamlit UploadedFile objects or paths.
        Extract text, chunk, embed, add to faiss index, persist docs.
        """
        new_texts = []
        new_metas = []
        for f in uploaded_files:
            try:
                # if it's a Streamlit UploadedFile, read bytes; else path
                if hasattr(f, "read"):
                    content = f.read()
                    # pdfplumber expects a file-like object; open from bytes via BytesIO
                    from io import BytesIO
                    pdf_stream = BytesIO(content)
                    with pdfplumber.open(pdf_stream) as pdf:
                        full_text = "\n\n".join([p.extract_text() or "" for p in pdf.pages])
                else:
                    # assume file path
                    with pdfplumber.open(f) as pdf:
                        full_text = "\n\n".join([p.extract_text() or "" for p in pdf.pages])
            except Exception as e:
                st.warning(f"Failed to read PDF {getattr(f,'name',str(f))}: {e}")
                continue

            chunks = chunk_text(full_text, chunk_size=1200, overlap=200)
            for i, c in enumerate(chunks):
                meta = {
                    "source": getattr(f, "name", str(f)),
                    "chunk_id": i
                }
                new_texts.append(c)
                new_metas.append(meta)

        if not new_texts:
            st.warning("No text found in uploaded PDF(s).")
            return

        # compute embeddings
        try:
            vectors = self._get_embeddings(new_texts)
        except Exception as e:
            st.error(f"Embedding error: {e}")
            return

        # ensure index
        dim = vectors.shape[1]
        try:
            self._ensure_index(dim)
        except Exception as e:
            st.error(f"Index init error: {e}")
            return

        # add vectors to index
        try:
            # faiss expects contiguous array
            vecs = np.ascontiguousarray(vectors)
            self.index.add(vecs)
        except Exception as e:
            st.error(f"Failed to add vectors to FAISS: {e}")
            return

        # append docs
        start_idx = len(self.docs)
        for i, text in enumerate(new_texts):
            self.docs.append({
                "text": text,
                "meta": new_metas[i],
                "vector_idx": start_idx + i
            })

        # persist index + docs
        try:
            faiss.write_index(self.index, INDEX_PATH)
            with open(DOCS_PATH, "wb") as f:
                pickle.dump(self.docs, f)
            st.success("PDF(s) indexed successfully!")
        except Exception as e:
            st.warning(f"Could not persist index to disk: {e}")

    def query(self, query_text: str, top_k: int = 3):
        """
        Given a query string, embed it and search FAISS for top_k results.
        Returns concatenated text snippets as the answer.
        """
        if self.index is None or len(self.docs) == 0:
            return "No documents indexed yet. Upload a PDF to enable RAG."

        try:
            q_vec = self._get_embeddings([query_text])
        except Exception as e:
            return f"Embedding error: {e}"

        # search
        try:
            D, I = self.index.search(q_vec, top_k)
            # D = similarities, I = indices
            results = []
            for idx in I[0]:
                if idx < 0 or idx >= len(self.docs):
                    continue
                results.append(self.docs[idx]["text"])
            if not results:
                return "No relevant content found in the indexed documents."
            # build result string: include top contexts and short guidance
            answer = "Here are the most relevant excerpts from your uploaded documents:\n\n"
            for i, r in enumerate(results, 1):
                answer += f"{i}. {r.strip()[:800].strip()}...\n\n"
            answer += "If you'd like a concise summary from these excerpts, ask: 'Summarize the PDF' or 'Short summary'."
            return answer
        except Exception as e:
            return f"RAG query failed: {e}"
