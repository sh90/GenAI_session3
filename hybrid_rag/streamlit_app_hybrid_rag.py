# streamlit_app_langchain_hybrid.py
import streamlit as st
from pathlib import Path
from dotenv import load_dotenv

from rag_langchain_hybrid import (
    ingest, delete_doc, rebuild, list_doc_ids,
    search_dense_only, hybrid_search_with_rerank, answer_with_llm,
    FAISS_DISTANCE
)

load_dotenv()
st.set_page_config(page_title="Hybrid RAG — LangChain + FAISS + BM25 + CE", layout="wide")
st.title("Hybrid RAG — LangChain (OpenAI Embeddings + BM25 + Cross-Encoder)")

tab_ingest, tab_dense, tab_hybrid, tab_manage = st.tabs(
    [" Ingest/Update", " Dense Only", " Hybrid + Rerank", " Manage"]
)

# ---------------- Ingest ----------------
with tab_ingest:
    st.subheader("Add / Update Documents")
    st.caption("Supports .pdf (digital text), .txt, .md")
    doc_id = st.text_input("Document ID", placeholder="e.g., calc_ch12")
    files = st.file_uploader("Upload files", type=["pdf","txt","md"], accept_multiple_files=True)

    if st.button("Ingest / Update"):
        if not doc_id or not files:
            st.error("Provide a Document ID and at least one file.")
        else:
            Path("data").mkdir(exist_ok=True)
            saved_paths = []
            for f in files:
                p = Path("data") / f.name
                p.write_bytes(f.read())
                saved_paths.append(str(p))
            res = ingest(doc_id, saved_paths)
            st.success(res)

# ---------------- Dense only ----------------
with tab_dense:
    st.subheader("Baseline: Dense Retrieval Only")
    q = st.text_input("Your query (dense)", placeholder="State the Fundamental Theorem of Calculus.")
    k = st.slider("Top-K", 1, 20, 6)
    if st.button("Search (Dense)"):
        hits = search_dense_only(q, k)
        if not hits:
            st.warning("No results.")
        else:
            label = "cosine sim (higher is closer)" if FAISS_DISTANCE == "COSINE" else "distance (L2; lower is closer)"
            for i, h in enumerate(hits, start=1):
                meta = h.get("meta") or {}
                st.markdown(
                    f"**{i}.** {label}: `{h['score']:.4f}` · "
                    f"**doc_id** `{meta.get('doc_id')}` · **source** `{meta.get('source')}`"
                )
                st.code((h["text"] or "")[:1200] + ("..." if len(h["text"]) > 1200 else ""))
                st.divider()

            if st.button(" Answer with GPT-4o (dense)"):
                st.markdown("### LLM Answer")
                st.write(answer_with_llm(q, hits[:k]))

# ---------------- Hybrid + Rerank ----------------
with tab_hybrid:
    st.subheader("Hybrid Retrieval (BM25 + Dense) with Cross-Encoder Rerank")
    qh = st.text_input("Your query (hybrid)", placeholder="integration by parts examples and proof")

    colA, colB = st.columns(2)
    with colA:
        k_final = st.slider("Final Top-K", 1, 20, 6)
        k_bm25  = st.slider("BM25 candidates", 8, 128, 32, step=4)
    with colB:
        k_dense = st.slider("Dense candidates", 8, 128, 32, step=4)
        use_rr  = st.checkbox("Use cross-encoder reranker", value=True)

    colW1, colW2, colQ, colAuto = st.columns([1,1,2,2])
    with colW1:
        w_bm25 = st.slider("Weight BM25", 0.0, 1.0, 0.5, 0.05)
    with colW2:
        w_dense = 1.0 - w_bm25
        st.caption(f"Weight Dense = {w_dense:.2f}")
    with colQ:
        use_qr = st.checkbox("Rewrite query (gpt-4o-mini)", value=True,
                             help="Expands synonyms/aliases and BM25 keyword string")
    with colAuto:
        auto_answer = st.checkbox("Generate LLM answer automatically", value=True)

    if st.button("Search (Hybrid + Rerank)"):
        if not qh.strip():
            st.error("Enter a query.")
        else:
            results, dbg = hybrid_search_with_rerank(
                user_query=qh,
                use_query_rewrite=use_qr,
                k_final=k_final,
                k_bm25=k_bm25,
                k_dense=k_dense,
                w_bm25=w_bm25,
                w_dense=w_dense,
                use_reranker=use_rr
            )
            with st.expander("Show rewritten queries / debug", expanded=True):
                st.markdown(f"**Semantic rewrite:** {dbg.get('semantic_query')}")
                st.markdown(f"**BM25 terms:** `{dbg.get('bm25_terms')}`")

            if not results:
                st.warning("No results.")
            else:
                st.markdown("### Top Results")
                for i, r in enumerate(results, start=1):
                    meta = r["meta"] or {}
                    scores = r["scores"]
                    label = f"**{i}.** "
                    if "rerank" in scores:
                        label += f"CE `{scores['rerank']:.4f}` · "
                    label += (
                        f"BM25 `{scores['bm25']:.4f}` · Dense `{scores['dense']:.4f}` · "
                        f"Merged `{scores['merged']:.4f}` · "
                        f"**doc_id** `{meta.get('doc_id')}` · **source** `{meta.get('source')}`"
                    )
                    st.markdown(label)
                    st.code((r["text"] or "")[:1200] + ("..." if len(r["text"]) > 1200 else ""))
                    st.divider()

                if auto_answer:
                    st.markdown("### LLM Answer")
                    st.write(answer_with_llm(qh, results[:k_final]))
                else:
                    if st.button("### Answer with GPT-4o (hybrid)"):
                        st.markdown("### LLM Answer")
                        st.write(answer_with_llm(qh, results[:k_final]))

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
                st.error("Pick a doc_id.")
            else:
                st.success(delete_doc(target))
    with c2:
        if st.button("Rebuild entire index"):
            st.success(rebuild())
