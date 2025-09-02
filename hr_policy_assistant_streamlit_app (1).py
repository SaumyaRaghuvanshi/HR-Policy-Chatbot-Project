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
from groq import Groq

# ------------------------------
# Configuration
# ------------------------------
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # small, fast, free
DEFAULT_TOP_K = 5
CHUNK_SIZE = 900
CHUNK_OVERLAP = 200
INDEX_DIR = "./indexes"
FEEDBACK_LOG = "feedback_log.csv"
ANSWER_LOG = "answer_log.csv"
SUGGESTION_LOG = "suggestion_log.csv"
DOCMAP_LOG = "docmap_log.json"

# ------------------------------
# Utilities
# ------------------------------

def ensure_dirs():
    os.makedirs(INDEX_DIR, exist_ok=True)


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
            raise RuntimeError("FAISS is not installed. Please install faiss-cpu.")
        self.faiss_index = faiss.IndexFlatIP(self.dim)

    def save(self):
        if self.faiss_index is None:
            return
        faiss.write_index(self.faiss_index, self.index_path)
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
            raise RuntimeError("FAISS is not installed. Please install faiss-cpu.")
        self.faiss_index = faiss.read_index(self.index_path)
        with open(self.meta_path, "r", encoding="utf-8") as f:
            meta_dict = json.load(f)
            self.meta = {k: ChunkMeta(**v) for k, v in meta_dict.items()}
        with open(self.docmap_path, "r", encoding="utf-8") as f:
            self.docmap = json.load(f)
        with open(self.ids_path, "r", encoding="utf-8") as f:
            self.id_map = json.load(f)
        self.dim = self.faiss_index.d
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
    api_key = os.getenv("GROQ_API_KEY", "")
    return Groq(api_key=api_key) if api_key else None


def call_llm_groq(client: Groq, system_prompt: str, user_prompt: str, model: str) -> str:
    if client is None:
        return "[LLM disabled: set GROQ_API_KEY in environment or sidebar]"
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


def append_csv(path: str, row: Dict[str, Any]):
    df = pd.DataFrame([row])
    header = not os.path.exists(path)
    df.to_csv(path, mode="a", header=header, index=False, encoding="utf-8")


def save_docmap(docmap: Dict[str, str]):
    with open(DOCMAP_LOG, "w", encoding="utf-8") as f:
        json.dump(docmap, f, ensure_ascii=False)


def load_docmap() -> Dict[str, str]:
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
    # convert to multiplier around 1.0
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

    # --- Header / Welcome ---
    st.title("🧭 HR Policy Assistant Chatbot")
    st.markdown(
        """
        👋 Welcome to your **HR Policy Assistant**!
        Here’s how you can use this app:
        - 📄 Upload your HR policy PDF from the sidebar, or load the default sample.
        - 🔍 Ask any HR-related questions (leave rules, benefits, code of conduct, etc.).
        - 📚 Get answers backed by your HR documents with clear citations.
        - ✅ Provide feedback on whether the answer was helpful.
        - 💡 Suggest modifications or improvements to policies for review by the HR team.

        Start by uploading a policy document or using the sample from the sidebar →
        """
    )
    
    ensure_dirs()

    with st.sidebar:
        st.header("📄 Documents & Index")
        uploaded_files = st.file_uploader("Upload HR policy PDFs", type=["pdf"], accept_multiple_files=True)
        use_sample = st.checkbox("Load sample HR_Policy.pdf if found in working directory", value=False)
        index_name = st.text_input("Index name", value="hr_policy_index")
        build_btn = st.button("(Re)Build Index", type="primary")
        st.divider()
        st.header("⚙️ Settings")
        top_k = st.slider("Top-K retrieval", 3, 10, DEFAULT_TOP_K)
        model_choice = st.selectbox("Groq model", ["llama-3.1-8b-instant", "mixtral-8x7b-32768"], index=0)
        api_key_input = st.text_input("GROQ_API_KEY (optional if set in env)", type="password")
        if api_key_input:
            os.environ["GROQ_API_KEY"] = api_key_input

    # --- Build index ---
    index_path = os.path.join(INDEX_DIR, f"{index_name}.faiss")

    if build_btn:
        if faiss is None:
            st.error("FAISS is not installed. Please add 'faiss-cpu' to requirements and reinstall.")
            st.stop()
        with st.spinner("Building index from PDFs..."):
            embedder = get_embedder()
            vec_dim = embedder.get_sentence_embedding_dimension()
            index = SimpleVectorIndex(index_path, vec_dim)
            # clean existing
            for p in [index_path, index_path+".meta.json", index_path+".docs.json", index_path+".ids"]:
                try:
                    os.remove(p)
                except FileNotFoundError:
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
                st.error("Please upload at least one PDF or enable sample.")
                st.stop()

            index._create()
            st.session_state["_doc_text_map"] = {}
            docmap_save: Dict[str, str] = {}

            for fname, fbytes in datas:
                doc_id = hash_bytes(fbytes)[:12]
                full_text, page_starts = pdf_to_text(fbytes)
                st.session_state["_doc_text_map"][doc_id] = full_text
                docmap_save[doc_id] = full_text
                chunks = chunk_text(full_text, CHUNK_SIZE, CHUNK_OVERLAP)
                texts = [c[0] for c in chunks if c[0].strip()]
                starts = [c[1] for c in chunks if c[0].strip()]
                ends = [s + len(t) for s, t in zip(starts, texts)]

                metas: List[ChunkMeta] = []
                for t, s, e in zip(texts, starts, ends):
                    cid = str(uuid.uuid4())[:8]
                    page = find_page_for_offset(page_starts, s)
                    preview = (t.strip().split("\n")[0] or t)[:140]
                    metas.append(ChunkMeta(id=cid, doc_id=doc_id, start=s, end=e, page=page, preview=preview))

                vectors = embed_texts(embedder, texts)
                if vectors.shape[0] > 0:
                    index.add(vectors, metas)
                index.docmap[doc_id] = {"filename": fname, "pages": len(page_starts)}

            index.save()
            save_docmap(docmap_save)
            st.success("Index built and saved.")

    # --- Q&A ---
    st.subheader("💬 Ask about HR policies")
    question = st.text_input("Your question", key="question_input", placeholder="e.g., What is the notice period for resignation?")
    ask_btn = st.button("Ask")

    if ask_btn and question.strip():
        # Load index
        embedder = get_embedder()
        index = SimpleVectorIndex(index_path, embedder.get_sentence_embedding_dimension())
        if not index.load():
            st.error("No index found. Build the index from the sidebar first.")
            st.stop()

        # Retrieve
        qvec = embed_texts(embedder, [question])[0]
        hits = index.search(qvec, top_k=top_k)

        # Feedback-based weighting
        weights = load_feedback_weights()
        if weights:
            hits = [(m, s * weights.get(m.doc_id, 1.0)) for (m, s) in hits]
            hits.sort(key=lambda x: x[1], reverse=True)

        # Build context for LLM
        full_text_map: Dict[str, str] = st.session_state.get("_doc_text_map", {})
        if not full_text_map:
            full_text_map = load_docmap()
        context_lines = []
        srcs = []
        for meta, score in hits:
            snippet = full_text_map.get(meta.doc_id, "")[meta.start:meta.end] if full_text_map else meta.preview
            snippet = snippet.replace("\n", " ")
            context_lines.append(f"[Source id={meta.id} page={meta.page+1} score={score:.3f}] {snippet}")
            srcs.append({"id": meta.id, "page": meta.page + 1, "score": score, "preview": meta.preview, "doc_id": meta.doc_id})

        context = "\n\n".join(context_lines)
        user_prompt = (
            f"CONTEXT (policy excerpts):\n{context}\n\n"
            f"USER QUESTION: {question}\n\n"
            "Instructions: Answer using only the CONTEXT. Cite sources as [id @ page]."
        )

        # Call Groq
        client = get_groq_client()
        answer = call_llm_groq(client, SYSTEM_PROMPT, user_prompt, model_choice)

        # Display
        st.subheader("Answer")
        st.write(answer or "(no response)")
        with st.expander("Sources (from uploaded PDFs)"):
            for s in srcs:
                st.markdown(f"• **{s['id']}** — page {s['page']} — _{s['preview']}_ (score {s['score']:.2f})")
        
        # Persist last Q&A for suggestions section
        st.session_state["last_question"] = question
        st.session_state["last_answer"] = answer

        # Log answer
        append_csv(ANSWER_LOG, {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "question": question,
            "answer": answer,
            "sources": json.dumps(srcs, ensure_ascii=False),
        })

        # Feedback UI
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

    # --- Suggestions ---
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

