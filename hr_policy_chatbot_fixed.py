# hr_policy_chatbot_fixed.py
"""
Fixed and hardened HR Policy Assistant Streamlit app.

Key fixes applied:
- Avoid storing full document text inside Streamlit session_state or a single large JSON.
  Instead save per-document text files under INDEX_DIR/doc_texts/{doc_id}.txt
- Save only lightweight docmap metadata (filename, pages) to docmap_log.json
- Build embeddings in batches and skip empty chunks
- Ensure metas and vectors remain aligned
- Wrap FAISS write/read in try/except and log errors
- Use st.secrets for GROQ API key if available (falls back to sidebar or env)
- Better error handling and debug expander to inspect the context sent to the LLM
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
# Groq client import is optional; app will run without it but LLM will be disabled.
try:
    from groq import Groq
except Exception:
    Groq = None  # type: ignore

# ------------------------------
# Configuration
# ------------------------------
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # small, fast
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
    """Return full text and the starting char index of each page."""
    reader = PdfReader(io.BytesIO(pdf_bytes))
    texts = []
    page_starts = []
    cursor = 0
    for page in reader.pages:
        page_starts.append(cursor)
        t = page.extract_text() or ""
        # normalize whitespace a bit
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
    """FAISS index + metadata.
    Save files: *.faiss, *.meta.json, *.docs.json, *.ids
    """
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
            raise RuntimeError("FAISS is not available. Please install faiss-cpu.")
        self.faiss_index = faiss.IndexFlatIP(self.dim)

    def save(self):
        if self.faiss_index is None:
            return
        try:
            faiss.write_index(self.faiss_index, self.index_path)
        except Exception as e:
            st.error(f"Error saving FAISS index: {e}")
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump({k: asdict(v) for k, v in self.meta.items()}, f, ensure_ascii=False)
        with open(self.docmap_path, "w", encoding="utf-8") as f:
            json.dump(self.docmap, f, ensure_ascii=False)
        with open(self.ids_path, "w", encoding="utf-8") as f:
            json.dump(self.id_map, f)

    def load(self) -> bool:
        if not (os.path.exists(self.index_path) and os.path.exists(self.meta_path) and os.path.exists(self.ids_path)):
            return False
        if faiss is None:
            raise RuntimeError("FAISS is not available. Please install faiss-cpu.")
        try:
            self.faiss_index = faiss.read_index(self.index_path)
        except Exception as e:
            st.error(f"Error reading FAISS index: {e}")
            return False
        with open(self.meta_path, "r", encoding="utf-8") as f:
            meta_dict = json.load(f)
            self.meta = {k: ChunkMeta(**v) for k, v in meta_dict.items()}
        with open(self.docmap_path, "r", encoding="utf-8") as f:
            self.docmap = json.load(f)
        with open(self.ids_path, "r", encoding="utf-8") as f:
            self.id_map = json.load(f)
        # dimension present on index
        self.dim = int(self.faiss_index.d)
        return True

    def add(self, vectors: np.ndarray, metas: List[ChunkMeta]):
        if self.faiss_index is None:
            self._create()
        # normalize for cosine via inner product
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
        results: List[Tuple[ChunkMeta, float]] = []
        for score, idx in zip(D[0], I[0]):
            if idx == -1:
                continue
            cid = self.id_map[idx]
            results.append((self.meta[cid], float(score)))
        return results

# ------------------------------
# Cached resources
# ------------------------------
@st.cache_resource(show_spinner=False)
def get_embedder():
    return SentenceTransformer(EMBED_MODEL_NAME)

def embed_texts(model: SentenceTransformer, texts: List[str]) -> np.ndarray:
    """Batch-embed texts and return numpy array (preserves order)."""
    out_vecs = []
    batch_size = 32
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        # filter empties
        batch = [t for t in batch if t.strip()]
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
    # Prefer st.secrets (Streamlit Cloud / local secrets), then env var.
    api_key = ""
    if hasattr(st, "secrets") and st.secrets.get("GROQ_API_KEY"):
        api_key = st.secrets["GROQ_API_KEY"]
    api_key = api_key or os.getenv("GROQ_API_KEY", "")
    if api_key:
        if Groq is None:
            st.warning("Groq client library not installed; LLM calls will be disabled.")
            return None
        return Groq(api_key=api_key)
    return None

def call_llm_groq(client: Any, system_prompt: str, user_prompt: str, model: str) -> str:
    if client is None:
        return "[LLM disabled: set GROQ_API_KEY in streamlit secrets or environment]"
    try:
        res = client.chat.completions.create(
            model=model,
            temperature=0.2,
            max_tokens=1000,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return res.choices[0].message.content or ""
    except Exception as e:
        st.error(f"LLM call failed: {e}")
        return "[LLM error]"

def append_csv(path: str, row: Dict[str, Any]):
    df = pd.DataFrame([row])
    header = not os.path.exists(path)
    df.to_csv(path, mode="a", header=header, index=False, encoding="utf-8")

def save_docmap(docmap: Dict[str, Any]):
    # docmap contains only small metadata (filename, pages)
    with open(DOCMAP_LOG, "w", encoding="utf-8") as f:
        json.dump(docmap, f, ensure_ascii=False)

def load_docmap() -> Dict[str, Any]:
    if not os.path.exists(DOCMAP_LOG):
        return {}
    with open(DOCMAP_LOG, "r", encoding="utf-8") as f:
        return json.load(f)

def load_feedback_weights() -> Dict[str, float]:
    if not os.path.exists(FEEDBACK_LOG):
        return {}
    df = pd.read_csv(FEEDBACK_LOG)
    weights: Dict[str, float] = {}
    for _, r in df.iterrows():
        doc_id = str(r.get("doc_id", ""))
        helpful = str(r.get("helpful", "")).lower() == "yes"
        if not doc_id:
            continue
        weights.setdefault(doc_id, 0.0)
        weights[doc_id] += 0.5 if helpful else -0.25
    for k, v in list(weights.items()):
        weights[k] = float(np.clip(1.0 + v, 0.5, 2.0))
    return weights

SYSTEM_PROMPT = (
    "You are an HR Policy Assistant. Answer clearly, concisely, and only using the supplied CONTEXT. "
    "If the answer is not in CONTEXT, say so. Always cite sources as [id @ page]."
)

# ------------------------------
# Streamlit App
# ------------------------------
def main():
    st.set_page_config(page_title="HR Policy Assistant", page_icon="🧭", layout="wide")
    st.title("🧭 HR Policy Assistant Chatbot")
    st.caption("Upload an HR policy PDF, build the index, and ask questions. Answers cite your document.")

    ensure_dirs()

    # Sidebar controls
    with st.sidebar:
        st.header("📄 Documents & Index")
        uploaded_files = st.file_uploader("Upload HR policy PDFs", type=["pdf"], accept_multiple_files=True)
        use_sample = st.checkbox("Load sample HR_Policy.pdf if present in working directory", value=False)
        index_name = st.text_input("Index name", value="hr_policy_index")
        build_btn = st.button("(Re)Build Index", type="primary")
        st.divider()
        st.header("⚙️ Settings")
        top_k = st.slider("Top-K retrieval", 3, 10, DEFAULT_TOP_K)
        model_choice = st.selectbox("Groq model", ["llama-3.1-8b-instant", "mixtral-8x7b-32768"], index=0)
        # we still allow sidebar entry for convenience
        api_key_input = st.text_input("GROQ_API_KEY (optional; secrets preferred)", type="password")
        if api_key_input:
            os.environ["GROQ_API_KEY"] = api_key_input

    index_path = os.path.join(INDEX_DIR, f"{index_name}.faiss")

    # Build index block
    if build_btn:
        if faiss is None:
            st.error("FAISS (faiss-cpu) is not installed. Please install it in your environment.")
            st.stop()
        with st.spinner("Building index from PDFs..."):
            embedder = get_embedder()
            vec_dim = embedder.get_sentence_embedding_dimension()
            index = SimpleVectorIndex(index_path, vec_dim)

            # remove existing files for a fresh build
            for p in [index_path, index_path + ".meta.json", index_path + ".docs.json", index_path + ".ids"]:
                try:
                    if os.path.exists(p):
                        os.remove(p)
                except Exception:
                    pass

            datas: List[Tuple[str, bytes]] = []
            if uploaded_files:
                for f in uploaded_files:
                    datas.append((f.name, f.read()))
            if use_sample and not uploaded_files:
                sample_path = "HR_Policy.pdf"
                if os.path.exists(sample_path):
                    with open(sample_path, "rb") as f:
                        datas.append((os.path.basename(sample_path), f.read()))
                else:
                    st.warning("Sample HR_Policy.pdf not found in working directory.")

            if not datas:
                st.error("Please upload at least one PDF or enable the sample option.")
                st.stop()

            try:
                index._create()
            except Exception as e:
                st.error(f"Failed to create FAISS index: {e}")
                st.stop()

            docmap_save: Dict[str, Any] = {}

            for fname, fbytes in datas:
                doc_id = hash_bytes(fbytes)[:12]
                full_text, page_starts = pdf_to_text(fbytes)
                # save full text to per-doc file (avoid storing big text in session)
                try:
                    save_doc_text(doc_id, full_text)
                except Exception as e:
                    st.warning(f"Failed to save doc text for {fname}: {e}")

                chunks = chunk_text(full_text, CHUNK_SIZE, CHUNK_OVERLAP)
                metas: List[ChunkMeta] = []
                texts: List[str] = []
                starts: List[int] = []
                ends: List[int] = []

                for chunk_text_item, start in chunks:
                    if not chunk_text_item.strip():
                        continue
                    cid = str(uuid.uuid4())[:8]
                    page = find_page_for_offset(page_starts, start)
                    preview = (chunk_text_item.strip().split("\n")[0] or chunk_text_item)[:140]
                    metas.append(ChunkMeta(id=cid, doc_id=doc_id, start=start, end=start + len(chunk_text_item), page=page, preview=preview))
                    texts.append(chunk_text_item)
                    starts.append(start)
                    ends.append(start + len(chunk_text_item))

                if not texts:
                    st.warning(f"No text chunks extracted from {fname}; skipping.")
                    continue

                # embed in batches; embed_texts returns aligned vectors for texts list
                try:
                    vectors = embed_texts(embedder, texts)
                except Exception as e:
                    st.error(f"Embedding failed for {fname}: {e}")
                    continue

                if vectors.shape[0] != len(metas):
                    # This would be unexpected; try to align by using min length
                    use_n = min(vectors.shape[0], len(metas))
                    if use_n == 0:
                        st.warning(f"No embeddings produced for {fname}; skipping.")
                        continue
                    vectors = vectors[:use_n]
                    metas = metas[:use_n]

                try:
                    index.add(vectors, metas)
                except Exception as e:
                    st.error(f"Failed to add vectors to FAISS index for {fname}: {e}")
                    continue

                index.docmap[doc_id] = {"filename": fname, "pages": len(page_starts)}
                docmap_save[doc_id] = {"filename": fname, "pages": len(page_starts)}

            # Save index and lightweight docmap
            index.save()
            save_docmap(docmap_save)
            st.success("Index built and saved successfully.")

    # Q&A UI
    st.subheader("💬 Ask about HR policies")
    question = st.text_input("Your question", key="question_input", placeholder="e.g., What is the notice period for resignation?")
    ask_btn = st.button("Ask")

    if ask_btn and question.strip():
        embedder = get_embedder()
        index = SimpleVectorIndex(index_path, embedder.get_sentence_embedding_dimension())
        if not index.load():
            st.error("No index found. Build the index from the sidebar first.")
            st.stop()

        # encode question
        try:
            qvecs = embed_texts(embedder, [question])
            if qvecs.shape[0] == 0:
                st.error("Failed to embed the question.")
                st.stop()
            qvec = qvecs[0]
        except Exception as e:
            st.error(f"Failed to embed question: {e}")
            st.stop()

        hits = index.search(qvec, top_k=top_k)

        # apply feedback weights
        weights = load_feedback_weights()
        if weights:
            hits = [(m, s * weights.get(m.doc_id, 1.0)) for (m, s) in hits]
            hits.sort(key=lambda x: x[1], reverse=True)

        # build context by loading per-doc text files (not from session_state)
        full_text_map = load_docmap()  # docmap has filenames/pages
        # context lines and source list
        context_lines = []
        srcs = []
        for meta, score in hits:
            doc_text = load_doc_text(meta.doc_id)
            snippet = doc_text[meta.start:meta.end] if doc_text else meta.preview
            snippet = snippet.replace("\n", " ")
            context_lines.append(f"[Source id={meta.id} page={meta.page+1} score={score:.3f}] {snippet}")
            srcs.append({"id": meta.id, "page": meta.page + 1, "score": score, "preview": meta.preview, "doc_id": meta.doc_id})

        context = "\n\n".join(context_lines)
        user_prompt = (
            f"CONTEXT (policy excerpts):\n{context}\n\n"
            f"USER QUESTION: {question}\n\n"
            "Instructions: Answer using only the CONTEXT. Cite sources as [id @ page]."
        )

        # optional debug expander
        with st.expander("Debug: context sent to LLM (trimmed)"):
            preview_len = 1000
            st.code(context[:preview_len] + ("..." if len(context) > preview_len else ""), language="text")

        client = get_groq_client()
        answer = call_llm_groq(client, SYSTEM_PROMPT, user_prompt, model_choice)

        st.subheader("Answer")
        st.write(answer or "(no response)")

        with st.expander("Sources (from uploaded PDFs)"):
            for s in srcs:
                fn = (index.docmap.get(s["doc_id"], {}) or {}).get("filename", "")
                st.markdown(f"• **{s['id']}** — page {s['page']} — _{s['preview']}_ — file: **{fn}** (score {s['score']:.2f})")

        append_csv(ANSWER_LOG, {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "question": question,
            "answer": answer,
            "sources": json.dumps(srcs, ensure_ascii=False),
        })

        # feedback UI
        st.subheader("Was this answer helpful?")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("👍 Yes"):
                for s in srcs:
                    append_csv(FEEDBACK_LOG, {
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "doc_id": s.get("doc_id", ""),
                        "chunk_id": s.get("id", ""),
                        "helpful": "Yes",
                        "question": question,
                    })
                st.success("Thanks for the feedback!")
        with c2:
            if st.button("👎 No"):
                for s in srcs:
                    append_csv(FEEDBACK_LOG, {
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "doc_id": s.get("doc_id", ""),
                        "chunk_id": s.get("id", ""),
                        "helpful": "No",
                        "question": question,
                    })
                st.info("Thanks — we'll use this to improve retrieval.")

        # persist last Q&A for suggestion UI
        st.session_state["last_question"] = question
        st.session_state["last_answer"] = answer

    # Suggestions
    st.subheader("💡 Policy Suggestions")
    suggestion = st.text_area("Do you have any suggestions or modifications regarding this policy?", "", placeholder="e.g., Consider allowing carry-forward of 5 Casual Leave days.")
    if st.button("Submit Suggestion"):
        q = st.session_state.get("last_question", "")
        a = st.session_state.get("last_answer", "")
        if suggestion.strip():
            append_csv(SUGGESTION_LOG, {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "question": q,
                "answer": a,
                "suggestion": suggestion.strip(),
            })
            st.success("✅ Your suggestion has been recorded and will be reviewed by HR.")
        else:
            st.warning("Please enter a suggestion before submitting.")

if __name__ == "__main__":
    main()
