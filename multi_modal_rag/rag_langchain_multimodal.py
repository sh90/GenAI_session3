# rag_langchain_multimodal.py
# LangChain-style Multimodal RAG
# - text: OpenAI embeddings -> FAISS
# - images: PDF pages -> PNG -> CLIP embeddings -> FAISS (IDMap)
# - tolerant to missing Poppler: pdf2image -> PyMuPDF -> else skip
# - GPT-4o multimodal answer

import os
import json
import hashlib
from pathlib import Path
from typing import List, Dict, Optional, Tuple

from dotenv import load_dotenv

# LangChain core / loaders
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# OpenAI via LangChain
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

# Vectorstores
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.faiss import DistanceStrategy
from langchain_community.docstore.in_memory import InMemoryDocstore

import faiss
import numpy as np
from PIL import Image

load_dotenv()

# ------------------------------------------------------------------ #
# 1) CONFIG / PATHS
# ------------------------------------------------------------------ #
BASE_DIR         = Path(".")
STORE_DIR        = BASE_DIR / "store_langchain_mm"
TEXT_INDEX_DIR   = STORE_DIR / "faiss_text"
IMG_INDEX_DIR    = STORE_DIR / "faiss_img"
MANIFEST_PATH    = STORE_DIR / "manifest.json"
IMG_META_PATH    = STORE_DIR / "img_meta.jsonl"
IMG_DIR          = STORE_DIR / "images"

CHUNK_SIZE       = int(os.getenv("CHUNK_SIZE", "900"))
CHUNK_OVERLAP    = int(os.getenv("CHUNK_OVERLAP", "150"))
OPENAI_EMBED     = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
LLM_MODEL        = os.getenv("LLM_MODEL", "gpt-4o")
FAISS_DISTANCE   = os.getenv("FAISS_DISTANCE", "COSINE").upper()
PAGE_DPI         = int(os.getenv("PAGE_DPI", "170"))  # PDF→image

# ------------------------------------------------------------------ #
# 2) CLIP fallback
# ------------------------------------------------------------------ #
def _build_clip_from_sentence_transformers():
    from sentence_transformers import SentenceTransformer

    class STClipWrapper:
        def __init__(self, model_name: str = "clip-ViT-B-32"):
            self.model = SentenceTransformer(model_name)

        def embed_image(self, image_path: str) -> List[float]:
            img = Image.open(image_path).convert("RGB")
            vec = self.model.encode(img)
            return vec.tolist()

        def embed_query(self, text: str) -> List[float]:
            vec = self.model.encode(text)
            return vec.tolist()

    return STClipWrapper()


def _clip_embeddings():
    # Try LC open-clip style first
    try:
        from langchain_community.embeddings import OpenCLIPEmbeddings  # type: ignore
        return OpenCLIPEmbeddings(
            model_name="openai/clip-vit-base-patch32",
        )
    except Exception:
        # Fallback to sentence-transformers
        return _build_clip_from_sentence_transformers()

# ------------------------------------------------------------------ #
# 3) BASIC UTILS
# ------------------------------------------------------------------ #
def _ensure_store():
    STORE_DIR.mkdir(parents=True, exist_ok=True)
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    if not MANIFEST_PATH.exists():
        MANIFEST_PATH.write_text(json.dumps({"docs": {}}, indent=2), encoding="utf-8")
    if not IMG_META_PATH.exists():
        IMG_META_PATH.write_text("", encoding="utf-8")

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

# ------------------------------------------------------------------ #
# 4) TEXT SIDE
# ------------------------------------------------------------------ #
def _text_embedding() -> OpenAIEmbeddings:
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("Missing OPENAI_API_KEY")
    return OpenAIEmbeddings(model=OPENAI_EMBED, api_key=key)

def _llm() -> ChatOpenAI:
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("Missing OPENAI_API_KEY")
    return ChatOpenAI(model=LLM_MODEL, temperature=0.2, api_key=key)

def _split_docs(docs: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    return splitter.split_documents(docs)

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

def _empty_text_vs() -> FAISS:
    emb = _text_embedding()
    dim = len(emb.embed_query("probe dim"))
    if FAISS_DISTANCE == "COSINE":
        index = faiss.IndexFlatIP(dim)
        return FAISS(
            embedding_function=emb,
            index=index,
            docstore=InMemoryDocstore({}),
            index_to_docstore_id={},
            normalize_L2=True,
            distance_strategy=DistanceStrategy.COSINE,
        )
    else:
        index = faiss.IndexFlatL2(dim)
        return FAISS(
            embedding_function=emb,
            index=index,
            docstore=InMemoryDocstore({}),
            index_to_docstore_id={},
            normalize_L2=False,
            distance_strategy=DistanceStrategy.EUCLIDEAN,
        )

def _load_text_vs() -> Optional[FAISS]:
    if not TEXT_INDEX_DIR.exists():
        return None
    try:
        return FAISS.load_local(
            str(TEXT_INDEX_DIR),
            _text_embedding(),
            allow_dangerous_deserialization=True,
        )
    except Exception:
        return None

def _save_text_vs(vs: FAISS):
    TEXT_INDEX_DIR.mkdir(parents=True, exist_ok=True)
    vs.save_local(str(TEXT_INDEX_DIR))

# ------------------------------------------------------------------ #
# 5) IMAGE SIDE (Poppler → PyMuPDF) + IDMap
# ------------------------------------------------------------------ #
def _append_img_meta(rec: Dict):
    with IMG_META_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

def _load_img_meta() -> List[Dict]:
    if not IMG_META_PATH.exists():
        return []
    rows = []
    with IMG_META_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def _empty_img_index() -> faiss.IndexIDMap:
    # sentence-transformers CLIP is 512-d
    dim = 512
    base = faiss.IndexFlatIP(dim)
    return faiss.IndexIDMap(base)

def _load_img_index() -> Optional[faiss.IndexIDMap]:
    if not IMG_INDEX_DIR.exists():
        return None
    fp = IMG_INDEX_DIR / "img.faiss"
    if not fp.exists():
        return None
    idx = faiss.read_index(str(fp))
    # wrap if needed
    if not isinstance(idx, faiss.IndexIDMap):
        idx = faiss.IndexIDMap(idx)
    return idx

def _save_img_index(idx: faiss.IndexIDMap):
    IMG_INDEX_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(idx, str(IMG_INDEX_DIR / "img.faiss"))

def _pdf_to_images(pdf_path: str, doc_id: str, dpi: int = PAGE_DPI) -> List[Path]:
    """
    Try pdf2image (needs poppler). If not available -> try PyMuPDF.
    If both fail -> return [].
    """
    # 1) try pdf2image
    try:
        from pdf2image import convert_from_path
        pages = convert_from_path(pdf_path, dpi=dpi)
        out_paths: List[Path] = []
        for i, page in enumerate(pages):
            out_p = IMG_DIR / f"{doc_id}__page_{i+1}.png"
            page.save(out_p, "PNG")
            out_paths.append(out_p)
        return out_paths
    except Exception as e:
        print(f"[multimodal] pdf2image failed: {e} – trying PyMuPDF")

    # 2) try PyMuPDF
    try:
        import fitz  # pymupdf
        doc = fitz.open(pdf_path)
        out_paths: List[Path] = []
        for i, page in enumerate(doc):
            pix = page.get_pixmap(dpi=dpi)
            out_p = IMG_DIR / f"{doc_id}__page_{i+1}.png"
            pix.save(str(out_p))
            out_paths.append(out_p)
        return out_paths
    except Exception as e2:
        print(f"[multimodal] PyMuPDF also failed: {e2} – skipping images")
        return []

def _embed_images(img_paths: List[Path]) -> Tuple[faiss.IndexIDMap, List[Dict]]:
    clip = _clip_embeddings()
    idx = _load_img_index()
    if idx is None:
        idx = _empty_img_index()

    vecs = []
    for p in img_paths:
        emb = clip.embed_image(str(p))
        vecs.append(emb)
    mat = np.array(vecs, dtype="float32")

    # create ids
    start_id = idx.ntotal
    ids = np.arange(start_id, start_id + len(img_paths), dtype="int64")

    # now add with ids (this is what failed before)
    idx.add_with_ids(mat, ids)
    _save_img_index(idx)

    metas = []
    for img_id, p in zip(ids, img_paths):
        metas.append({"img_id": int(img_id), "path": str(p)})
    return idx, metas

# ------------------------------------------------------------------ #
# 6) INGEST
# ------------------------------------------------------------------ #
def ingest(doc_id: str, file_paths: List[str]) -> Dict:
    """
    - Text: LC loaders -> split -> FAISS
    - Images: PDF pages -> image -> CLIP -> FAISS (IDMap)
    """
    _ensure_store()
    man = _load_manifest()
    old = man["docs"].get(doc_id)
    combo = _combined_hash(file_paths)

    unchanged = bool(old and old.get("combined_hash") == combo)

    # -------- TEXT PART --------
    if not unchanged:
        if old:
            delete_doc(doc_id)
        raw_docs = _load_files_as_docs(file_paths, doc_id)
        chunks = _split_docs(raw_docs)

        vs = _load_text_vs()
        if vs is None:
            vs = _empty_text_vs()

        if chunks:
            vec_ids = vs.add_documents(chunks)
            _save_text_vs(vs)
        else:
            vec_ids = []

        man["docs"][doc_id] = {
            "files": file_paths,
            "combined_hash": combo,
            "text_vector_ids": vec_ids,
        }
        _save_manifest(man)

    # -------- IMAGE PART --------
    img_meta_records = []
    for p in file_paths:
        if Path(p).suffix.lower() == ".pdf":
            page_imgs = _pdf_to_images(p, doc_id, dpi=PAGE_DPI)
            if not page_imgs:
                # images couldn't be extracted, continue silently
                continue
            idx, metas = _embed_images(page_imgs)
            for m, pg_path in zip(metas, page_imgs):
                # try getting page no
                stem = pg_path.stem  # e.g. docid__page_3
                try:
                    pg_no = int(stem.split("_")[-1])
                except Exception:
                    pg_no = None
                rec = {
                    "img_id": m["img_id"],
                    "doc_id": doc_id,
                    "page": pg_no,
                    "path": m["path"],
                    "source": Path(p).name,
                }
                _append_img_meta(rec)
                img_meta_records.append(rec)

    return {
        "status": "ingested" if not unchanged else "unchanged_but_images_refreshed",
        "doc_id": doc_id,
        "text_chunks": len(man["docs"][doc_id].get("text_vector_ids", [])),
        "images_added": len(img_meta_records),
    }

# ------------------------------------------------------------------ #
# 7) DELETE / REBUILD
# ------------------------------------------------------------------ #
def _rebuild_image_index_from_meta(meta_rows: List[Dict]):
    # wipe old
    if IMG_INDEX_DIR.exists():
        for p in IMG_INDEX_DIR.glob("*"):
            p.unlink()
        IMG_INDEX_DIR.rmdir()
    if not meta_rows:
        return
    clip = _clip_embeddings()
    idx = _empty_img_index()
    vecs = []
    ids = []
    for r in meta_rows:
        emb = clip.embed_image(r["path"])
        vecs.append(emb)
        ids.append(int(r["img_id"]))
    mat = np.array(vecs, dtype="float32")
    idx.add_with_ids(mat, np.array(ids, dtype="int64"))
    _save_img_index(idx)

def delete_doc(doc_id: str) -> Dict:
    _ensure_store()
    man = _load_manifest()
    entry = man["docs"].get(doc_id)
    if not entry:
        return {"deleted": False, "reason": "doc_id not found"}

    # text side
    vs = _load_text_vs()
    if vs is not None:
        ids = entry.get("text_vector_ids", [])
        if ids:
            try:
                vs.delete(ids)
            except Exception:
                rebuild()
                man = _load_manifest()
                man["docs"].pop(doc_id, None)
                _save_manifest(man)
                return {"deleted": True, "doc_id": doc_id}
        _save_text_vs(vs)

    # image side
    all_meta = _load_img_meta()
    kept = [r for r in all_meta if r.get("doc_id") != doc_id]
    with IMG_META_PATH.open("w", encoding="utf-8") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    _rebuild_image_index_from_meta(kept)

    man["docs"].pop(doc_id, None)
    _save_manifest(man)
    return {"deleted": True, "doc_id": doc_id}

def rebuild() -> Dict:
    """
    Full rebuild from manifest (re-ingests all files).
    """
    _ensure_store()
    man = _load_manifest()
    docs = man.get("docs", {})

    # wipe text index
    if TEXT_INDEX_DIR.exists():
        for p in TEXT_INDEX_DIR.glob("*"):
            p.unlink()
        TEXT_INDEX_DIR.rmdir()

    # wipe image index + meta
    if IMG_INDEX_DIR.exists():
        for p in IMG_INDEX_DIR.glob("*"):
            p.unlink()
        IMG_INDEX_DIR.rmdir()

    if IMG_META_PATH.exists():
        IMG_META_PATH.unlink()
        IMG_META_PATH.write_text("", encoding="utf-8")

    total_text = 0
    total_imgs = 0
    for did, meta in docs.items():
        files = meta.get("files", [])
        res = ingest(did, files)
        total_text += res.get("text_chunks", 0)
        total_imgs += res.get("images_added", 0)

    return {"status": "rebuilt", "text_chunks": total_text, "images": total_imgs}

# ------------------------------------------------------------------ #
# 8) RETRIEVAL
# ------------------------------------------------------------------ #
def search_text(query: str, k: int = 6) -> List[Dict]:
    vs = _load_text_vs()
    if vs is None:
        return []
    hits = vs.similarity_search_with_score(query, k=k)
    out = []
    for doc, score in hits:
        out.append({
            "text": doc.page_content,
            "meta": doc.metadata,
            "score": float(score),
        })
    return out

def search_images(query: str, k: int = 4) -> List[Dict]:
    idx = _load_img_index()
    if idx is None or idx.ntotal == 0:
        return []
    clip = _clip_embeddings()
    q_emb = clip.embed_query(query)
    qv = np.array([q_emb], dtype="float32")
    D, I = idx.search(qv, min(k, idx.ntotal))
    img_meta = _load_img_meta()
    meta_by_id = {m["img_id"]: m for m in img_meta}
    out = []
    for score, img_id in zip(D[0], I[0]):
        m = meta_by_id.get(int(img_id))
        if not m:
            continue
        out.append({
            "score": float(score),
            "img_id": int(img_id),
            "meta": m,
        })
    return out

# ------------------------------------------------------------------ #
# 9) GPT-4o MULTIMODAL
# ------------------------------------------------------------------ #
_SYSTEM_MM = """You are a math tutor. You will receive:
1) text chunks from a math chapter
2) some page images/diagrams

Explain step by step, return LaTeX where needed, and cite chunks like [1], [2].
If an image shows a geometry figure, describe it briefly.
"""
import base64
import os

def _image_file_to_data_url(path: str) -> str:
    # guess mime type quickly
    ext = os.path.splitext(path)[1].lower()
    if ext in [".jpg", ".jpeg"]:
        mime = "image/jpeg"
    elif ext in [".png"]:
        mime = "image/png"
    else:
        # fallback
        mime = "image/png"
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def answer_with_gpt4o_multimodal(
    question: str,
    text_hits: list[dict],
    image_hits: list[dict],
    max_ctx_chars: int = 6000,
) -> str:
    """
    Multimodal RAG:
    - text chunks → input_text
    - local images → base64 data URLs → input_image
    """
    from openai import OpenAI
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return "(Set OPENAI_API_KEY)"
    client = OpenAI(api_key=api_key)

    # 1) pack text chunks
    ctx_lines = []
    used = 0
    for i, h in enumerate(text_hits, start=1):
        t = (h.get("text") or "").strip()
        if not t:
            continue
        if used + len(t) > max_ctx_chars:
            break
        m = h.get("meta") or {}
        ctx_lines.append(f"[{i}] (doc:{m.get('doc_id')} · src:{m.get('source')})\n{t}")
        used += len(t)

    content = [
        {"type": "input_text", "text": (f"""
            You are an education-focused assistant. 
            You will receive:
            - TEXT context chunks, each already numbered like [1], [2], [3] ...
            - (Optionally) IMAGE references, numbered like [img1], [img2] ...
            
            Your job:
            1. Answer the learner’s question using ONLY the given context.
            2. Whenever you use a fact from a text chunk, cite it inline like this: ... [1] or ... [2], [3]
            3. Whenever your explanation depends on a diagram/page image, mention it like this: (see [img1]) or (refer [img2]).

      
           If you need to show a formula, definition, or derivation, write it in LaTeX so it can be rendered by Streamlit.

            Rules:
            - Use plain text / bullet points for explanations.
            -  "Whenever you show a formula or mathematical equation, wrap it in $$ ... $$ so that Streamlit markdown can render it.\n"
                "Example:\n"
                "$$ A = \\pi r^2 $$\n"
                "Cite text chunks like [1], [2] and images like (see [img1]).\n"
                "Do not wrap the whole answer in latex."
    
            - Do NOT put the answer in code fences.
            - If the context is not enough, say so briefly and then give the general method.
          """
                    )},
        {"type": "input_text", "text": f"QUESTION:\n{question}"},
        {"type": "input_text",
         "text": "CONTEXT (text):\n" + "\n\n---\n".join(ctx_lines) if ctx_lines else "CONTEXT (text):\n(none)"},
    ]

    # 2) attach images as base64 data URLs
    for ih in image_hits:
        img_path = ih["meta"]["path"]
        if not os.path.exists(img_path):
            continue
        data_url = _image_file_to_data_url(img_path)
        #  now it's a valid "URL" string (data:...)
        content.append(
            {
                "type": "input_image",
                "image_url": data_url,
            }
        )

    resp = client.responses.create(
        model="gpt-4o",
        input=[{"role": "user", "content": content}],
    )
    return resp.output_text



# ------------------------------------------------------------------ #
# 10) Convenience
# ------------------------------------------------------------------ #
def list_doc_ids() -> List[str]:
    man = _load_manifest()
    return sorted(list(man.get("docs", {}).keys()))
