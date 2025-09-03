# hr_policy_chatbot_safe.py
"""
Crash-proof HR Policy Assistant Streamlit app.

Fixes applied:
- Never store full PDF text in st.session_state (which crashes Streamlit).
- Save each doc’s extracted text to indexes/doc_texts/{doc_id}.txt.
- Only store metadata (filename, pages, previews) in memory/JSON.
- Batch embeddings, skip empty chunks safely.
- Guard against FAISS crashes with try/except.
"""

import os
import io
import json
import time
import uuid
import hashlib
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Tuple

import streamlit as st
import pandas as pd
import numpy as np

from sentence_transformers import SentenceTransformer

try:
    import faiss  # type: ignore
except Exception:
    faiss = None

from pypdf import PdfReader
try:
    from groq import Groq
except Exception:
    Groq = None  # type: ignore

# ------------------------------
# Configuration
# ------------------------------
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_TOP_K = 5
CHUNK_SIZE = 900
CHUNK_OVERLAP = 200
INDEX_DIR = "./indexes"
DOC_TEXT_DIR = os.path.join(INDEX_DIR, "doc_texts")
FEEDBACK_LOG = "feedback_log.csv"
ANSWER_LOG = "answer_log.csv"
SUGGESTION_LOG = "suggestion_log.csv"
DOCMAP_LOG = "docmap_log.json"

# ------------------------------
# Utilities
# ------------------------------

def ensure_dirs():
    os.makedirs(INDEX_DIR, exist_ok=True)
    os.makedirs(DOC_TEXT_DIR, exist_ok=True)

def hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def pdf_to_text(pdf_bytes: bytes) -> Tuple[str, List[int]]:
    reader = PdfReader(io.BytesIO(pdf_bytes))
    texts = []
    page_starts = []
    cursor = 0
    for page in reader.pages:
        page_starts.append(cursor)
        t = page.extract_text() or ""
        t = "\n".join(line.strip() for line in t.splitlines())
        texts.append(t)
        cursor += len(t) + 1
    return "\n".join(texts), page_starts

def save_doc_text(doc_id: str, text: str):
    path = os.path.join(DOC_TEXT_DIR, f"{doc_id}.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)

def load_doc_text(doc_id: str) -> str:
    path = os.path.join(DOC_TEXT_DIR, f"{doc_id}.txt")
    if not os.path.exists(path):
        return ""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def find_page_for_offset(page_starts: List[int], offset: int) -> int:
    lo, hi = 0, len(page_starts) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if page_starts[mid] <= offset:
            lo = mid + 1
        else:
            hi = mid - 1
    return max(0, hi)

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[Tuple[str, int]]:
    chunks = []
    i, n = 0, len(text)
    while i < n:
        j = min(n, i + chunk_size)
        chunks.append((text[i:j], i))
        i = j - overlap
        if i <= 0:
            i = j
    return chunks

@dataclass
class ChunkMeta:
    id: str
    doc_id: str
    start: int
    end: int
    page: int
    preview: str

class SimpleVectorIndex:
    def __init__(self, index_path: str, dim: int):
        self.index_path = index_path
        self.dim = dim
        self.meta_path = index_path + ".meta.json"
        self.docmap_path = index_path + ".docs.json"
        self.ids_path = index_path + ".ids"
        self.id_map: List[str] = []
        self.meta: Dict[str, ChunkMeta] = {}
        self.docmap: Dict[str, Dict[str, Any]] = {}
        self.faiss_index = None

    def _create(self):
        if faiss is None:
            raise RuntimeError("FAISS not installed. Please install faiss-cpu.")
        self.faiss_index = faiss.IndexFlatIP(self.dim)

    def save(self):
        if self.faiss_index is None:
            return
        try:
            faiss.write_index(self.faiss_index, self.index_path)
        except Exception as e:
            st.error(f"FAISS save error: {e}")
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump({k: asdict(v) for k, v in self.meta.items()}, f)
        with open(self.docmap_path, "w", encoding="utf-8") as f:
            json.dump(self.docmap, f)
        with open(self.ids_path, "w", encoding="utf-8") as f:
            json.dump(self.id_map, f)

    def load(self) -> bool:
        if not (os.path.exists(self.index_path) and os.path.exists(self.meta_path) and os.path.exists(self.ids_path)):
            return False
        if faiss is None:
            raise RuntimeError("FAISS not installed.")
        try:
            self.faiss_index = faiss.read_index(self.index_path)
        except Exception as e:
            st.error(f"FAISS load error: {e}")
            return False
        with open(self.meta_path, "r", encoding="utf-8") as f:
            meta_dict = json.load(f)
            self.meta = {k: ChunkMeta(**v) for k, v in meta_dict.items()}
        with open(self.docmap_path, "r", encoding="utf-8") as f:
            self.docmap = json.load(f)
        with open(self.ids_path, "r", encoding="utf-8") as f:
            self.id_map = json.load(f)
        self.dim = int(self.faiss_index.d)
        return True

    def add(self, vectors: np.ndarray, metas: List[ChunkMeta]):
        if self.faiss_index is None:
            self._create()
        vectors = vectors.astype("float32")
        norms = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-10
        vectors = vectors / norms
        self.faiss_index.add(vectors)
        for m in metas:
            self.id_map.append(m.id)
            self.meta[m.id] = m

    def search(self, query_vec: np.ndarray, top_k: int = DEFAULT_TOP_K) -> List[Tuple[ChunkMeta, float]]:
        if self.faiss_index is None:
            return []
        q = query_vec.astype("float32")
        q = q / (np.linalg.norm(q) + 1e-10)
        D, I = self.faiss_index.search(q.reshape(1, -1), top_k)
        results = []
        for score, idx in zip(D[0], I[0]):
            if idx == -1:
                continue
            cid = self.id_map[idx]
            results.append((self.meta[cid], float(score)))
        return results

@st.cache_resource(show_spinner=False)
def get_embedder():
    return SentenceTransformer(EMBED_MODEL_NAME)

def embed_texts(model: SentenceTransformer, texts: List[str]) -> np.ndarray:
    out_vecs = []
    batch_size = 32
    for i in range(0, len(texts), batch_size):
        batch = [t for t in texts[i:i+batch_size] if t.strip()]
        if not batch:
            continue
        vecs = model.encode(batch, show_progress_bar=False, convert_to_numpy=True)
        out_vecs.append(vecs)
    if out_vecs:
        return np.vstack(out_vecs)
    else:
        return np.zeros((0, model.get_sentence_embedding_dimension()), dtype="float32")

@st.cache_resource(show_spinner=False)
def get_groq_client():
    api_key = ""
    if hasattr(st, "secrets") and st.secrets.get("GROQ_API_KEY"):
        api_key = st.secrets["GROQ_API_KEY"]
    api_key = api_key or os.getenv("GROQ_API_KEY", "")
    if api_key and Groq is not None:
        return Groq(api_key=api_key)
    return None

def call_llm_groq(client, system_prompt: str, user_prompt: str, model: str) -> str:
    if client is None:
        return "[LLM disabled: set GROQ_API_KEY in secrets or env]"
    try:
        res = client.chat.completions.create(
            model=model,
            temperature=0.2,
            max_tokens=800,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return res.choices[0].message.content or ""
    except Exception as e:
        st.error(f"LLM error: {e}")
        return "[LLM error]"

def append_csv(path: str, row: Dict[str, Any]):
    df = pd.DataFrame([row])
    header = not os.path.exists(path)
    df.to_csv(path, mode="a", header=header, index=False)

SYSTEM_PROMPT = (
    "You are an HR Policy Assistant. Answer clearly, concisely, and only using the supplied CONTEXT. "
    "If the answer is not in CONTEXT, say so. Cite sources as [id @ page]."
)

# ------------------------------
# Streamlit App
# ------------------------------

def main():
    st.set_page_config(page_title="HR Policy Assistant", page_icon="🧭", layout="wide")
    st.title("🧭 HR Policy Assistant Chatbot")
    st.caption("Upload an HR policy PDF, build the index, and ask questions.")

    ensure_dirs()

    with st.sidebar:
        st.header("📄 Documents & Index")
        uploaded_files = st.file_uploader("Upload HR policy PDFs", type=["pdf"], accept_multiple_files=True)
        use_sample = st.checkbox("Use sample HR_Policy.pdf if found", value=False)
        index_name = st.text_input("Index name", value="hr_policy_index")
        build_btn = st.button("(Re)Build Index", type="primary")
        st.divider()
        st.header("⚙️ Settings")
        top_k = st.slider("Top-K retrieval", 3, 10, DEFAULT_TOP_K)
        model_choice = st.selectbox("Groq model", ["llama-3.1-8b-instant", "mixtral-8x7b-32768"], index=0)

    index_path = os.path.join(INDEX_DIR, f"{index_name}.faiss")

    if build_btn:
        if faiss is None:
            st.error("FAISS not installed.")
            st.stop()
        with st.spinner("Building index..."):
            embedder = get_embedder()
            vec_dim = embedder.get_sentence_embedding_dimension()
            index = SimpleVectorIndex(index_path, vec_dim)
            for p in [index_path, index_path+".meta.json", index_path+".docs.json", index_path+".ids"]:
                if os.path.exists(p):
                    os.remove(p)
            datas = []
            if uploaded_files:
                for f in uploaded_files:
                    datas.append((f.name, f.read()))
            if use_sample and not uploaded_files:
                if os.path.exists("HR_Policy.pdf"):
                    with open("HR_Policy.pdf", "rb") as f:
                        datas.append(("HR_Policy.pdf", f.read()))
            if not datas:
                st.error("Please upload a PDF or enable sample.")
                st.stop()
            index._create()
            docmap_save = {}
            for fname, fbytes in datas:
                doc_id = hash_bytes(fbytes)[:12]
                text, page_starts = pdf_to_text(fbytes)
                save_doc_text(doc_id, text)
                chunks = chunk_text(text)
                texts, metas = [], []
                for chunk, start in chunks:
                    if not chunk.strip():
                        continue
                    cid = str(uuid.uuid4())[:8]
                    page = find_page_for_offset(page_starts, start)
                    preview = (chunk.strip().split("\n")[0])[:140]
                    metas.append(ChunkMeta(cid, doc_id, start, start+len(chunk), page, preview))
                    texts.append(chunk)
                if not texts:
                    continue
                vecs = embed_texts(embedder, texts)
                if vecs.shape[0] != len(metas):
                    continue
                index.add(vecs, metas)
                index.docmap[doc_id] = {"filename": fname, "pages": len(page_starts)}
                docmap_save[doc_id] = {"filename": fname, "pages": len(page_starts)}
            index.save()
            with open(DOCMAP_LOG, "w") as f:
                json.dump(docmap_save, f)
            st.success("Index built.")

    st.subheader("💬 Ask about HR policies")
    q = st.text_input("Your question", placeholder="e.g., What is the notice period?")
    if st.button("Ask") and q.strip():
        embedder = get_embedder()
        index = SimpleVectorIndex(index_path, embedder.get_sentence_embedding_dimension())
        if not index.load():
            st.error("No index found. Build it first.")
            st.stop()
        qvec = embed_texts(embedder, [q])[0]
        hits = index.search(qvec, top_k=top_k)
        context_lines, srcs = [], []
        for meta, score in hits:
            doc_text = load_doc_text(meta.doc_id)
            snippet = doc_text[meta.start:meta.end] if doc_text else meta.preview
            snippet = snippet.replace("\n", " ")
            context_lines.append(f"[{meta.id} p{meta.page+1}] {snippet}")
            srcs.append({"id": meta.id, "page": meta.page+1, "score": score, "preview": meta.preview})
        context = "\n\n".join(context_lines)
        user_prompt = f"CONTEXT:\n{context}\n\nQUESTION: {q}\n\nAnswer only using CONTEXT."
        client = get_groq_client()
        ans = call_llm_groq(client, SYSTEM_PROMPT, user_prompt, model_choice)
        st.subheader("Answer")
        st.write(ans)
        with st.expander("Sources"):
            st.json(srcs)

if __name__ == "__main__":
    main()
