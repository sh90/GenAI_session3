# rag_langchain.py
# LangChain-based RAG: loaders, chunking, OpenAI embeddings, FAISS store, retrieval, and GPT-4o answers.

import os
import json
import hashlib
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from langchain_community.vectorstores.faiss import DistanceStrategy

from dotenv import load_dotenv

# LangChain core
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

# Loaders & text split
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Embeddings & vector store
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore

import faiss  # for creating an empty FAISS index

load_dotenv()

# ---------------------- Paths & settings ----------------------
BASE_DIR        = Path(".")
STORE_DIR       = BASE_DIR / "store_langchain"
VSTORE_DIR      = STORE_DIR / "faiss_index"
MANIFEST_PATH   = STORE_DIR / "manifest.json"      # {docs: {doc_id: {files, combined_hash, vector_ids: []}}}
CHUNK_SIZE      = int(os.getenv("CHUNK_SIZE", "2000"))
CHUNK_OVERLAP   = int(os.getenv("CHUNK_OVERLAP", "200"))
EMBED_MODEL     = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")  # 1536-d by default
LLM_MODEL       = os.getenv("LLM_MODEL", "gpt-4o-mini")  # or "gpt-4o"
FAISS_DISTANCE = os.getenv("FAISS_DISTANCE", "COSINE")  # or "L2"

# ---------------------- Utilities ----------------------
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
    # order-insensitive content hash
    hashes = sorted(_sha256_file(p) for p in paths)
    return hashlib.sha256("|".join(hashes).encode()).hexdigest()

def _embedding() -> OpenAIEmbeddings:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY")
    return OpenAIEmbeddings(model=EMBED_MODEL, api_key=api_key)

def _llm() -> ChatOpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY")
    # temperature low for precise, grounded answers
    return ChatOpenAI(model=LLM_MODEL, temperature=0.2, api_key=api_key)

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

def _split_docs(raw_docs: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    return splitter.split_documents(raw_docs)

def _load_files_as_docs(file_paths: List[str], doc_id: str) -> List[Document]:
    out: List[Document] = []
    for p in file_paths:
        ext = Path(p).suffix.lower()
        if ext == ".pdf":
            loader = PyPDFLoader(p)
            docs = loader.load()
        elif ext in (".txt", ".md", ".markdown"):
            loader = TextLoader(p, autodetect_encoding=True)
            docs = loader.load()
        else:
            raise ValueError(f"Unsupported file type: {ext}")
        # Attach metadata
        for d in docs:
            d.metadata = {**(d.metadata or {}), "doc_id": doc_id, "source": Path(p).name}
        out.extend(docs)
    return out

def _empty_vectorstore() -> FAISS:
    emb = _embedding()
    dim = len(emb.embed_query("dim probe"))
    if FAISS_DISTANCE.upper() == "COSINE":
        index = faiss.IndexFlatIP(dim)  # dot product on normalized vectors = cosine
        return FAISS(
            embedding_function=emb,
            index=index,
            docstore=InMemoryDocstore({}),
            index_to_docstore_id={},
            normalize_L2=True,  # normalize at add/query time
            distance_strategy=DistanceStrategy.COSINE
        )
    else:
        index = faiss.IndexFlatL2(dim)
        return FAISS(
            embedding_function=emb,
            index=index,
            docstore=InMemoryDocstore({}),
            index_to_docstore_id={},
            normalize_L2=False,
        )

# ---------------------- Public API ----------------------
def ingest(doc_id: str, file_paths: List[str]) -> Dict:
    """
    Idempotent ingest/update:
    - If file bytes unchanged → no-op
    - Else delete previous vectors for doc_id and add new ones
    """
    _ensure_store()
    man = _load_manifest()
    old = man["docs"].get(doc_id)

    combo = _combined_hash(file_paths)
    if old and old.get("combined_hash") == combo:
        return {"status": "unchanged", "doc_id": doc_id}

    # load + split
    raw_docs = _load_files_as_docs(file_paths, doc_id)
    chunks = _split_docs(raw_docs)
    if not chunks:
        # Allow empty, but still update manifest hash
        man["docs"][doc_id] = {"files": file_paths, "combined_hash": combo, "vector_ids": []}
        _save_manifest(man)
        return {"status": "ingested", "doc_id": doc_id, "chunks": 0}

    # delete previous vectors for this doc_id (if any)
    if old:
        delete_doc(doc_id)

    # create or load vectorstore
    vs = _load_vectorstore()
    if vs is None:
        vs = _empty_vectorstore()

    # add documents and capture vector IDs
    vector_ids = vs.add_documents(chunks)  # List[str]
    _save_vectorstore(vs)

    # update manifest
    man["docs"][doc_id] = {
        "files": file_paths,
        "combined_hash": combo,
        "vector_ids": vector_ids,
    }
    _save_manifest(man)
    return {"status": "ingested", "doc_id": doc_id, "chunks": len(chunks)}

def delete_doc(doc_id: str) -> Dict:
    _ensure_store()
    man = _load_manifest()
    entry = man["docs"].get(doc_id)
    if not entry:
        return {"deleted": False, "reason": "doc_id not found"}

    vs = _load_vectorstore()
    if vs is None:
        # nothing to delete from index, clear manifest
        man["docs"].pop(doc_id, None)
        _save_manifest(man)
        return {"deleted": True, "doc_id": doc_id, "removed": 0}

    ids = entry.get("vector_ids", [])
    removed = 0
    if ids:
        try:
            vs.delete(ids)
            removed = len(ids)
        except Exception:
            # Fallback: rebuild the entire store by re-adding all other docs
            rebuild()
            man = _load_manifest()
            man["docs"].pop(doc_id, None)
            _save_manifest(man)
            return {"deleted": True, "doc_id": doc_id, "removed": removed}

    _save_vectorstore(vs)
    man["docs"].pop(doc_id, None)
    _save_manifest(man)
    return {"deleted": True, "doc_id": doc_id, "removed": removed}

def rebuild() -> Dict:
    """
    Rebuild FAISS from manifest (useful if index corrupted or after manual edits).
    """
    _ensure_store()
    man = _load_manifest()
    docs = man.get("docs", {})
    # wipe the index directory
    if VSTORE_DIR.exists():
        for p in VSTORE_DIR.glob("*"):
            p.unlink()
        VSTORE_DIR.rmdir()
    total_chunks = 0
    # re-ingest each doc from its files
    for did, meta in docs.items():
        files = meta.get("files", [])
        res = ingest(did, files)
        total_chunks += int(res.get("chunks", 0))
    return {"status": "rebuilt", "documents": len(docs), "total_chunks": total_chunks}

def search(query: str, k: int = 6) -> List[Dict]:
    """
    Similarity search with scores; returns [{text, meta, score}, ...]
    """
    vs = _load_vectorstore()
    if vs is None:
        return []
    results: List[Tuple[Document, float]] = vs.similarity_search_with_score(query, k=k)
    hits: List[Dict] = []
    for doc, score in results:
        hits.append({
            "text": doc.page_content,
            "meta": doc.metadata,
            "score": float(score),
        })
    return hits

# ---------------------- Answering (LLM via LangChain) ----------------------
_SYSTEM_PROMPT = """You are a precise tutor. Answer ONLY using the provided context.
Use inline citations like [1], [2] to refer to the snippet indices provided.
If the context is insufficient, say so explicitly."""
USER_PROMPT_TMPL = """QUESTION:
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
        src = h.get("meta", {}).get("source")
        did = h.get("meta", {}).get("doc_id")
        ctx.append(f"[{i}] (doc:{did} · src:{src})\n{t}")
        used += len(t)
    return "\n\n---\n".join(ctx) if ctx else "(none)"

def answer_with_llm(question: str, hits: List[Dict], model: Optional[str] = None) -> str:
    """
    Answer with LangChain's ChatOpenAI (gpt-4o-mini / gpt-4o) using a proper system prompt.
    """
    model_name = model or LLM_MODEL
    llm = ChatOpenAI(model=model_name, temperature=0.2, api_key=os.getenv("OPENAI_API_KEY"))

    # Build the chat prompt with a system message and a human message
    prompt = ChatPromptTemplate.from_messages([
        ("system", _SYSTEM_PROMPT),
        ("human", USER_PROMPT_TMPL),
    ])

    # Prepare variables
    variables = {
        "question": question,
        "context": _format_context(hits),
    }

    try:
        # Compose and invoke
        chain = prompt | llm
        resp = chain.invoke(variables)
        return resp.content if hasattr(resp, "content") else str(resp)
    except Exception as e:
        return f"(Answer error: {e})"


# convenience: expose manifest for UI
def list_doc_ids() -> List[str]:
    man = _load_manifest()
    return sorted(list(man.get("docs", {}).keys()))
