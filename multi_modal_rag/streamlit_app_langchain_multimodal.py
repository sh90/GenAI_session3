# streamlit_app_langchain_multimodal.py
import streamlit as st
from pathlib import Path
from dotenv import load_dotenv
from PIL import Image

from rag_langchain_multimodal import (
    ingest, delete_doc, rebuild, list_doc_ids,
    search_text, search_images, answer_with_gpt4o_multimodal,
    STORE_DIR
)

load_dotenv()
st.set_page_config(page_title="Multimodal RAG — LangChain + GPT-4o", layout="wide")
st.title("📘 Multimodal RAG — Text + Diagrams from Scanned PDFs")

tab_ingest, tab_query, tab_manage = st.tabs(["📥 Ingest/Update", "🔎 Query", "🧰 Manage"])

# ---------------- Ingest ----------------
with tab_ingest:
    st.subheader("Add / Update Documents")
    doc_id = st.text_input("Document ID", placeholder="e.g. math_ch_10_circles")
    files = st.file_uploader("Upload PDF/TXT/MD (scanned PDF allowed)", type=["pdf","txt","md"], accept_multiple_files=True)

    if st.button("Ingest / Update"):
        if not doc_id or not files:
            st.error("Provide document ID and at least one file.")
        else:
            Path("data").mkdir(exist_ok=True)
            saved_paths = []
            for f in files:
                p = Path("data") / f.name
                p.write_bytes(f.read())
                saved_paths.append(str(p))
            res = ingest(doc_id, saved_paths)
            st.success(res)

# ---------------- Query ----------------
with tab_query:
    st.subheader("Text + Image retrieval")
    query = st.text_input("Your question", placeholder="Prove that the lengths of tangents from an external point are equal.")
    k_txt = st.slider("Top text chunks", 1, 20, 6)
    k_img = st.slider("Top images (diagrams/pages)", 0, 8, 3)
    auto = st.checkbox("Generate GPT-4o multimodal answer", value=True)

    if st.button("Search"):
        if not query.strip():
            st.error("Enter a question.")
        else:
            text_hits = search_text(query, k=k_txt)
            img_hits  = search_images(query, k=k_img)

            st.markdown("### Retrieved text chunks")
            if not text_hits:
                st.info("No text hits.")
            else:
                for i, h in enumerate(text_hits, start=1):
                    meta = h["meta"] or {}
                    st.markdown(
                        f"**{i}.** score `{h['score']:.4f}` · doc_id `{meta.get('doc_id')}` · source `{meta.get('source')}`"
                    )
                    st.code((h["text"] or "")[:1200] + ("..." if len(h["text"])>1200 else ""))
                    st.divider()

            st.markdown("### Retrieved images / pages")
            if not img_hits:
                st.info("No image hits.")
            else:
                cols = st.columns(len(img_hits))
                for col, ih in zip(cols, img_hits):
                    meta = ih["meta"]
                    col.caption(f"score {ih['score']:.4f} · p.{meta.get('page')}")
                    img = Image.open(meta["path"])
                    col.image(img, width="stretch")

            if auto:
                st.markdown("### GPT-4o Answer (multimodal)")
                ans = answer_with_gpt4o_multimodal(query, text_hits, img_hits).strip()
                st.markdown(ans)

# ---------------- Manage ----------------
with tab_manage:
    st.subheader("Delete / Rebuild")
    ids = list_doc_ids()
    st.markdown("**Indexed docs:** " + (", ".join(ids) if ids else "_none_"))

    c1, c2 = st.columns(2)
    with c1:
        target = st.selectbox("Delete doc_id", ["-- choose --"] + ids)
        if st.button("Delete selected"):
            if target == "-- choose --":
                st.error("Pick a doc_id")
            else:
                st.success(delete_doc(target))
    with c2:
        if st.button("Rebuild all"):
            st.success(rebuild())

