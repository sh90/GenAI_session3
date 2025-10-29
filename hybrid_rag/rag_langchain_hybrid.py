# rag_langchain_hybrid.py
# Hybrid RAG with LangChain:
# - Loaders + chunking (PyPDFLoader/TextLoader + RecursiveCharacterTextSplitter)
# - OpenAI embeddings (text-embedding-3-*)
# - FAISS vector store (cosine/IP) for dense retrieval
# - BM25Retriever (langchain_community) for lexical retrieval
# - Weighted fusion (BM25 + dense with min-max normalize)
# - Optional Cross-Encoder (MiniLM) reranker
# - GPT-4o/4o-mini for answers via ChatOpenAI

import os
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from dotenv import load_dotenv

# LangChain core
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

# Loaders & splitters
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Vector store (FAISS)
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.faiss import DistanceStrategy
from langchain_community.docstore.in_memory import InMemoryDocstore

# BM25 retriever
from langchain_community.retrievers import BM25Retriever

# Optional reranker
from sentence_transformers import CrossEncoder

import faiss
import re

load_dotenv()

# ---------------------- Paths & settings ----------------------
BASE_DIR        = Path(".")
STORE_DIR       = BASE_DIR / "store_langchain_hybrid"
VSTORE_DIR      = STORE_DIR / "faiss_index"
MANIFEST_PATH   = STORE_DIR / "manifest.json"     # {docs:{doc_id:{files, combined_hash, vector_ids}}}
BM25_CORPUS     = STORE_DIR / "bm25_corpus.jsonl" # one JSON per line: {"text":..., "metadata":...}

CHUNK_SIZE      = int(os.getenv("CHUNK_SIZE", "900"))
CHUNK_OVERLAP   = int(os.getenv("CHUNK_OVERLAP", "150"))

EMBED_MODEL     = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
LLM_MODEL       = os.getenv("LLM_MODEL", "gpt-4o-mini")
FAISS_DISTANCE  = os.getenv("FAISS_DISTANCE", "COSINE").upper()  # "COSINE" or "L2"
RERANK_MODEL    = os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

# ---------------------- Utils ----------------------
def _ensure_store():
    STORE_DIR.mkdir(parents=True, exist_ok=True)
    if not MANIFEST_PATH.exists():
        MANIFEST_PATH.write_text(json.dumps({"docs": {}}, indent=2), encoding="utf-8")

def _load_manifest() -> Dict:
    _ensure_store()
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

def _save_manifest(m: Dict):
    MANIFEST_PATH.write_text(json.dumps(m, indent=2), encoding="utf-8")

def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def _combined_hash(paths: List[str]) -> str:
    hashes = sorted(_sha256_file(p) for p in paths)
    return hashlib.sha256("|".join(hashes).encode()).hexdigest()

def _embedding() -> OpenAIEmbeddings:
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("Missing OPENAI_API_KEY")
    return OpenAIEmbeddings(model=EMBED_MODEL, api_key=key)

def _llm(model: Optional[str] = None) -> ChatOpenAI:
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("Missing OPENAI_API_KEY")
    return ChatOpenAI(model=model or LLM_MODEL, temperature=0.2, api_key=key)

def _split_docs(raw_docs: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    return splitter.split_documents(raw_docs)

def _load_files_as_docs(file_paths: List[str], doc_id: str) -> List[Document]:
    out: List[Document] = []
    for p in file_paths:
        ext = Path(p).suffix.lower()
        if ext == ".pdf":
            docs = PyPDFLoader(p).load()
        elif ext in (".txt", ".md", ".markdown"):
            docs = TextLoader(p, autodetect_encoding=True).load()
        else:
            raise ValueError(f"Unsupported file type: {ext}")
        for d in docs:
            d.metadata = {**(d.metadata or {}), "doc_id": doc_id, "source": Path(p).name}
        out.extend(docs)
    return out

# ---------- FAISS store helpers ----------
def _empty_faiss_cosine() -> FAISS:
    emb = _embedding()
    dim = len(emb.embed_query("dim probe"))
    index = faiss.IndexFlatIP(dim)  # inner product + normalize → cosine
    return FAISS(
        embedding_function=emb,
        index=index,
        docstore=InMemoryDocstore({}),
        index_to_docstore_id={},
        normalize_L2=True,
        distance_strategy=DistanceStrategy.COSINE,
    )

def _empty_faiss_l2() -> FAISS:
    emb = _embedding()
    dim = len(emb.embed_query("dim probe"))
    index = faiss.IndexFlatL2(dim)
    return FAISS(
        embedding_function=emb,
        index=index,
        docstore=InMemoryDocstore({}),
        index_to_docstore_id={},
        normalize_L2=False,
        distance_strategy=DistanceStrategy.EUCLIDEAN,
    )

def _load_vectorstore() -> Optional[FAISS]:
    if not VSTORE_DIR.exists():
        return None
    try:
        return FAISS.load_local(str(VSTORE_DIR), _embedding(), allow_dangerous_deserialization=True)
    except Exception:
        return None

def _save_vectorstore(vs: FAISS):
    VSTORE_DIR.mkdir(parents=True, exist_ok=True)
    vs.save_local(str(VSTORE_DIR))

# ---------- BM25 corpus helpers ----------
def _save_bm25_corpus(docs: List[Document], append: bool = True):
    BM25_CORPUS.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append and BM25_CORPUS.exists() else "w"
    with BM25_CORPUS.open(mode, encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps({"text": d.page_content, "metadata": d.metadata}, ensure_ascii=False) + "\n")

def _load_bm25_retriever(k: int = 32) -> Optional[BM25Retriever]:
    if not BM25_CORPUS.exists():
        return None
    docs: List[Document] = []
    with BM25_CORPUS.open("r", encoding="utf-8") as f:
        for line in f:
            j = json.loads(line)
            docs.append(Document(page_content=j["text"], metadata=j["metadata"]))
    if not docs:
        return None
    ret = BM25Retriever.from_documents(docs)
    ret.k = k
    return ret

def _rewrite_query(original: str) -> Dict:
    """
    Return {'semantic_query': ..., 'bm25_terms': ...}
    via ChatOpenAI with a strict JSON-only response.
    """
    llm = _llm("gpt-4o-mini")
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Return ONLY compact JSON with keys 'semantic_query' and 'bm25_terms'. "
         "No code fences, no commentary."),
        ("human",
         "Rewrite the user's query in two ways:\n"
         "1) 'semantic_query': clearer natural-language version.\n"
         "2) 'bm25_terms': SHORT keyword string for BM25 (include synonyms; use quotes for exact phrases).\n\n"
         "User query:\n{q}")
    ])
    try:
        resp = (prompt | llm).invoke({"q": original})
        txt = (resp.content or "").strip()
        j = json.loads(re.search(r"\{[\s\S]*\}", txt).group(0)) if "{" in txt else {}
        sem = j.get("semantic_query", original)
        kw  = j.get("bm25_terms", original)
        return {"semantic_query": sem, "bm25_terms": kw}
    except Exception:
        return {"semantic_query": original, "bm25_terms": original}

# ---------------------- Public API: ingest / delete / rebuild ----------------------
def ingest(doc_id: str, file_paths: List[str]) -> Dict:
    _ensure_store()
    man = _load_manifest()
    old = man["docs"].get(doc_id)

    combo = _combined_hash(file_paths)
    if old and old.get("combined_hash") == combo:
        return {"status": "unchanged", "doc_id": doc_id}

    # delete previous (both FAISS vectors and BM25 docs)
    if old:
        delete_doc(doc_id)

    # load + chunk
    raw = _load_files_as_docs(file_paths, doc_id)
    chunks = _split_docs(raw)
    if not chunks:
        man["docs"][doc_id] = {"files": file_paths, "combined_hash": combo, "vector_ids": []}
        _save_manifest(man)
        return {"status": "ingested", "doc_id": doc_id, "chunks": 0}

    # vector store
    vs = _load_vectorstore()
    if vs is None:
        vs = _empty_faiss_cosine() if FAISS_DISTANCE == "COSINE" else _empty_faiss_l2()

    vector_ids = vs.add_documents(chunks)   # returns list[str]
    _save_vectorstore(vs)

    # BM25: append this doc's chunks to corpus
    # (We keep a flat corpus; delete_doc rewrites it.)
    _save_bm25_corpus(chunks, append=True)

    # manifest
    man["docs"][doc_id] = {"files": file_paths, "combined_hash": combo, "vector_ids": vector_ids}
    _save_manifest(man)
    return {"status": "ingested", "doc_id": doc_id, "chunks": len(chunks)}

def delete_doc(doc_id: str) -> Dict:
    _ensure_store()
    man = _load_manifest()
    entry = man["docs"].get(doc_id)
    if not entry:
        return {"deleted": False, "reason": "doc_id not found"}

    # remove from FAISS
    vs = _load_vectorstore()
    removed = 0
    if vs is not None:
        ids = entry.get("vector_ids", [])
        if ids:
            try:
                vs.delete(ids)
                removed = len(ids)
            except Exception:
                # fallback: rebuild index from all remaining docs
                rebuild()
                man = _load_manifest()
                man["docs"].pop(doc_id, None)
                _save_manifest(man)
                return {"deleted": True, "doc_id": doc_id, "removed": removed}
        _save_vectorstore(vs)

    # rewrite BM25 corpus without this doc_id
    if BM25_CORPUS.exists():
        kept: List[str] = []
        with BM25_CORPUS.open("r", encoding="utf-8") as f:
            for line in f:
                j = json.loads(line)
                if j.get("metadata", {}).get("doc_id") != doc_id:
                    kept.append(line)
        with BM25_CORPUS.open("w", encoding="utf-8") as f:
            for line in kept:
                f.write(line)

    man["docs"].pop(doc_id, None)
    _save_manifest(man)
    return {"deleted": True, "doc_id": doc_id, "removed": removed}

def rebuild() -> Dict:
    """
    Rebuild FAISS + BM25 from manifest and source files.
    """
    _ensure_store()
    man = _load_manifest()
    docs = man.get("docs", {})

    # wipe stores
    if VSTORE_DIR.exists():
        for p in VSTORE_DIR.glob("*"):
            p.unlink()
        VSTORE_DIR.rmdir()
    if BM25_CORPUS.exists():
        BM25_CORPUS.unlink()

    total = 0
    for did, meta in docs.items():
        files = meta.get("files", [])
        res = ingest(did, files)
        total += int(res.get("chunks", 0))
    return {"status": "rebuilt", "documents": len(docs), "total_chunks": total}

# ---------------------- Retrieval & fusion ----------------------
def _dense_candidates(query: str, k: int) -> List[Tuple[Document, float]]:
    vs = _load_vectorstore()
    if vs is None:
        return []
    # similarity_search_with_score returns (Document, score)
    # For COSINE in LC, higher = closer; for L2, lower = closer.
    return vs.similarity_search_with_score(query, k=k)

# --- replace this whole function ---
def _bm25_candidates(query_terms: str, k: int) -> List[Tuple[Document, float]]:
    ret = _load_bm25_retriever(k=k)
    if ret is None:
        return []

    # LangChain >= 0.2 retrievers are Runnables; prefer .invoke()
    try:
        docs: List[Document] = ret.invoke(query_terms)  # returns List[Document]
    except AttributeError:
        # Fallback for older versions
        docs = ret.get_relevant_documents(query_terms)

    # BM25Retriever doesn’t expose scores; use a lightweight token-overlap proxy
    toks = re.findall(r'\w+|\"[^\"]+\"', query_terms.lower())

    def score(doc: Document) -> float:
        txt = doc.page_content.lower()
        s = 0.0
        for t in toks:
            t0 = t.strip('"')
            if t0 and t0 in txt:
                s += 1.0
        return s

    pairs = [(d, score(d)) for d in docs]
    pairs.sort(key=lambda x: x[1], reverse=True)
    return pairs[:k]


def _minmax(vals: List[float]) -> Tuple[float, float]:
    if not vals:
        return (0.0, 1.0)
    return (min(vals), max(vals))

def hybrid_search_with_rerank(
    user_query: str,
    use_query_rewrite: bool = True,
    k_final: int = 6,
    k_bm25: int = 32,
    k_dense: int = 32,
    w_bm25: float = 0.5,
    w_dense: float = 0.5,
    use_reranker: bool = True,
) -> Tuple[List[Dict], Dict]:
    """
    Returns (results, debug) where results is:
      [{ "text", "meta", "scores": {"bm25", "dense", "merged", "rerank"?} }, ... ]
    """
    # 1) rewrite
    if use_query_rewrite:
        wr = _rewrite_query(user_query)
        q_sem = wr["semantic_query"]
        q_kw  = wr["bm25_terms"]
    else:
        q_sem = q_kw = user_query

    debug = {"semantic_query": q_sem, "bm25_terms": q_kw}

    # 2) candidate pools
    dense_pairs = _dense_candidates(q_sem, k_dense)           # [(doc, dense_score)]
    bm25_pairs  = _bm25_candidates(q_kw, k_bm25)              # [(doc, bm25_score_proxy)]

    # Map by a stable doc key (content hash + minimal meta) to fuse
    def key_of(d: Document) -> str:
        meta_part = f"{d.metadata.get('doc_id','')}|{d.metadata.get('source','')}|{d.metadata.get('page','')}"
        h = hashlib.sha1((d.page_content + "|" + meta_part).encode("utf-8")).hexdigest()
        return h

    dense_map: Dict[str, float] = {key_of(d): s for d, s in dense_pairs}
    bm25_map:  Dict[str, float] = {key_of(d): s for d, s in bm25_pairs}
    doc_map:   Dict[str, Document] = {key_of(d): d for d, _ in dense_pairs + bm25_pairs}

    all_keys = set(dense_map.keys()) | set(bm25_map.keys())
    if not all_keys:
        return [], debug

    bm_vals = [bm25_map.get(k, 0.0) for k in all_keys]
    de_vals = [dense_map.get(k, 0.0) for k in all_keys]
    bm_min,bm_max = _minmax(bm_vals)
    de_min,de_max = _minmax(de_vals)

    def norm(v, a, b):
        return 0.0 if b <= a else (v - a) / (b - a)

    fused: List[Tuple[str, float, float, float]] = []
    for k in all_keys:
        bm = bm25_map.get(k, 0.0)
        de = dense_map.get(k, 0.0)
        mg = w_bm25 * norm(bm, bm_min, bm_max) + w_dense * norm(de, de_min, de_max)
        fused.append((k, bm, de, mg))

    fused.sort(key=lambda x: x[3], reverse=True)
    prelim = fused[: max(k_final * 3, 30)]

    # 3) optional rerank with CrossEncoder
    out: List[Dict] = []
    if use_reranker and prelim:
        ce = CrossEncoder(RERANK_MODEL)
        pairs = [(q_sem, doc_map[k].page_content) for (k, _, _, _) in prelim]
        ce_scores = ce.predict(pairs)  # higher is better
        for (k, bm, de, mg), ce_s in zip(prelim, ce_scores):
            d = doc_map[k]
            out.append({
                "text": d.page_content,
                "meta": d.metadata,
                "scores": {"bm25": float(bm), "dense": float(de), "merged": float(mg), "rerank": float(ce_s)}
            })
        out.sort(key=lambda x: x["scores"]["rerank"], reverse=True)
        return out[:k_final], debug

    # no reranker
    for (k, bm, de, mg) in prelim[:k_final]:
        d = doc_map[k]
        out.append({"text": d.page_content, "meta": d.metadata, "scores": {"bm25": float(bm), "dense": float(de), "merged": float(mg)}})
    return out, debug

# ---------------------- Dense-only helper ----------------------
def search_dense_only(query: str, k: int = 6) -> List[Dict]:
    pairs = _dense_candidates(query, k)
    out = []
    for d, s in pairs:
        out.append({"text": d.page_content, "meta": d.metadata, "score": float(s)})
    return out

# ---------------------- Answering ----------------------
_SYSTEM_PROMPT = """You are a precise tutor. Answer ONLY using the provided context.
Use inline citations like [1], [2] to refer to the snippet indices provided.
If the context is insufficient, say so explicitly."""

_USER_TMPL = """QUESTION:
{question}

CONTEXT SNIPPETS:
{context}"""

def _format_context(hits: List[Dict], max_chars: int = 6000) -> str:
    ctx, used = [], 0
    for i, h in enumerate(hits, start=1):
        t = (h.get("text") or "").strip()
        if not t:
            continue
        if used + len(t) > max_chars:
            break
        m = h.get("meta", {}) or {}
        ctx.append(f"[{i}] (doc:{m.get('doc_id')} · src:{m.get('source')} · page:{m.get('page','?')})\n{t}")
        used += len(t)
    return "\n\n---\n".join(ctx) if ctx else "(none)"

def answer_with_llm(question: str, hits: List[Dict], model: Optional[str] = None) -> str:
    llm = _llm(model or LLM_MODEL)
    prompt = ChatPromptTemplate.from_messages([
        ("system", _SYSTEM_PROMPT),
        ("human", _USER_TMPL),
    ])
    try:
        resp = (prompt | llm).invoke({"question": question, "context": _format_context(hits)})
        return resp.content if hasattr(resp, "content") else str(resp)
    except Exception as e:
        return f"(Answer error: {e})"

# convenience
def list_doc_ids() -> List[str]:
    man = _load_manifest()
    return sorted(list(man.get("docs", {}).keys()))
