# streamlit_app.py
import io
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv

from rag_langchain import (
    ingest, delete_doc, rebuild, search, answer_with_llm, list_doc_ids
)

load_dotenv()
st.set_page_config(page_title="Vanilla RAG (LangChain + OpenAI + FAISS)", layout="wide")
st.title("Vanilla RAG — LangChain + OpenAI Embeddings + FAISS")

tab_ingest, tab_query, tab_manage = st.tabs([" Ingest/Update", " Query ", " Manage "])

with tab_ingest:
    st.subheader("Add or Update Documents")
    st.caption("Supports .pdf (digital text), .txt, .md")
    doc_id = st.text_input("Document ID", placeholder="e.g., calc_ch12")
    files = st.file_uploader("Upload files", type=["pdf","txt","md"], accept_multiple_files=True)

    if st.button("Ingest / Update"):
        if not doc_id or not files:
            st.error("Provide a Document ID and at least one file.")
        else:
            import time
            start_time = time.time()
            Path("data").mkdir(exist_ok=True)
            saved_paths = []
            for f in files:
                p = Path("data")/f.name
                p.write_bytes(f.read())
                saved_paths.append(str(p))
            res = ingest(doc_id, saved_paths)
            st.success(res)
            end_time = time.time()
            time_taken = (end_time-start_time)
            print(time_taken)
            st.write(time_taken)

with tab_query:
    st.subheader("Semantic Search + Answer")
    query = st.text_input("Your query", placeholder="State the Fundamental Theorem of Calculus.")
    top_k = st.slider("Top-K (retrieved chunks)", min_value=1, max_value=20, value=6)
    auto_answer = st.checkbox("Generate answer automatically (gpt-4o-mini)", value=True)
    model_name = st.selectbox("LLM model", ["gpt-4o-mini", "gpt-4o"], index=0)
    run = st.button("Search")

    if run:
        if not query.strip():
            st.error("Enter a query.")
        else:
            hits = search(query, k=top_k)
            if not hits:
                st.warning("No results (index empty or not matching).")
                st.stop()

            st.markdown("### Retrieved chunks")
            for i, h in enumerate(hits, start=1):
                meta = h["meta"] or {}
                st.markdown(
                    f"**{i}.** score `{h['score']:.4f}` · **doc_id** `{meta.get('doc_id')}` · "
                    f"**source** `{meta.get('source')}`"
                )
                st.code(h["text"])
                st.divider()

            if auto_answer:
                st.markdown("### LLM Answer")
                ans = answer_with_llm(query, hits[:top_k], model=model_name)
                st.write(ans)
            else:
                if st.button(" Generate Answer "):
                    st.markdown("### LLM Answer")
                    ans = answer_with_llm(query, hits[:top_k], model=model_name)
                    st.write(ans)

with tab_manage:
    st.subheader("Delete / Rebuild")
    ids = list_doc_ids()
    st.markdown("**Indexed docs:** " + (", ".join(ids) if ids else "_none_"))

    col1, col2 = st.columns(2)
    with col1:
        target = st.selectbox("Select doc_id to delete", ["-- choose --"] + ids)
        if st.button("Delete selected"):
            if target == "-- choose --":
                st.error("Pick a doc_id.")
            else:
                st.success(delete_doc(target))

    with col2:
        if st.button("Rebuild entire index"):
            st.success(rebuild())
