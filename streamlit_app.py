"""Streamlit UI for PYQ-to-NCERT paragraph retrieval."""

from __future__ import annotations

import html
from pathlib import Path

import streamlit as st

from src.retrieval.intelligent_query import IntelligentQueryEngine

st.set_page_config(page_title="PYQ to NCERT Retriever", page_icon="", layout="centered")

CUSTOM_CSS = """
<style>
.main > div {
    max-width: 950px;
    margin: 0 auto;
}
.stApp {
    color: #f8fafc;
}
.card {
    border: 1px solid #334155;
    border-radius: 14px;
    padding: 16px 18px;
    margin-bottom: 14px;
    background: linear-gradient(180deg, #111827 0%, #0b1220 100%);
    box-shadow: 0 10px 28px rgba(2, 6, 23, 0.45);
}
.card h4,
.card p,
.card span,
.card li,
.card div {
    color: #f8fafc !important;
}
.title {
    font-size: 1.65rem;
    font-weight: 700;
    color: #f8fafc;
}
.subtitle {
    color: #cbd5e1;
    margin-bottom: 1rem;
}
div[data-testid="stSelectbox"] label,
div[data-testid="stSelectbox"] div,
div[data-testid="stCaptionContainer"] p,
div[data-testid="stMarkdownContainer"] p,
div[data-testid="stText"] {
    color: #f8fafc !important;
}
button[kind="primary"] {
    background-color: #2563eb !important;
    color: #ffffff !important;
    border: 1px solid #2563eb !important;
}
.pyq-text {
    margin-top: 0.5rem;
    font-size: 1.05rem;
    line-height: 1.65;
    color: #ffffff;
}
.meta {
    margin-top: 0.75rem;
    color: #cbd5e1;
    font-size: 0.9rem;
}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
st.markdown("<div class='title'>CBSE PYQ -> NCERT Paragraph Retriever</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='subtitle'>Select a PYQ and find the most relevant NCERT paragraph(s).</div>",
    unsafe_allow_html=True,
)

if "engine" not in st.session_state:
    st.session_state.engine = IntelligentQueryEngine(
        chroma_dir=Path("chroma_db"),
        collection_name="ncert_chemistry",
        chunks_path=Path("output/chunks/all_chunks.json"),
        device="cpu",
    )

pyq_rows = st.session_state.engine.get_pyq_list(limit=500)
if not pyq_rows:
    st.error("No PYQs found in output/chunks/all_chunks.json. Run pipeline first.")
    st.stop()

labels = {
    row["pyq_id"]: f"{row['file_name']} | {row['pyq_id']} | {row['text'][:110]}..."
    for row in pyq_rows
}

with st.sidebar:
    st.subheader("Select PYQ")
    selected_pyq_id = st.selectbox("PYQ", options=list(labels.keys()), format_func=lambda key: labels[key])

selected_pyq = next(row for row in pyq_rows if row["pyq_id"] == selected_pyq_id)

selected_text = html.escape(selected_pyq["text"])
selected_source = html.escape(selected_pyq["file_name"])
selected_para = html.escape(str(selected_pyq["paragraph_number"]))
st.markdown(
    (
        "<div class='card'>"
        "<h4>Selected PYQ</h4>"
        f"<div class='pyq-text'>{selected_text}</div>"
        f"<div class='meta'>Source: {selected_source} | Paragraph: {selected_para}</div>"
        "</div>"
    ),
    unsafe_allow_html=True,
)

if st.button("Find Matching NCERT Paragraph", use_container_width=True):
    with st.spinner("Retrieving and reranking NCERT paragraphs..."):
        result = st.session_state.engine.query_from_pyq(pyq_id=selected_pyq_id, top_k=10)

    best = result.get("best_matching_ncert_paragraph")
    second = result.get("second_supporting_paragraph")

    st.markdown("<div class='card'><h4>Best Matching NCERT Paragraph</h4>", unsafe_allow_html=True)
    if best:
        st.write(best["text"])
        st.caption(
            f"File: {best['metadata'].get('file_name', '')} | "
            f"Paragraph: {best['metadata'].get('paragraph_number', '')} | "
            f"Score: {best['score']:.4f}"
        )
    else:
        st.write("No NCERT paragraph found.")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='card'><h4>Optional Second Supporting Paragraph</h4>", unsafe_allow_html=True)
    if second:
        st.write(second["text"])
        st.caption(
            f"File: {second['metadata'].get('file_name', '')} | "
            f"Paragraph: {second['metadata'].get('paragraph_number', '')} | "
            f"Score: {second['score']:.4f}"
        )
    else:
        st.write("No second paragraph selected.")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='card'><h4>Debug Details</h4>", unsafe_allow_html=True)
    st.write(f"Second paragraph status: {result['debug']['second_paragraph_status']}")
    st.write("Top cross-encoder scores:")
    for row in result["debug"]["top_cross_encoder_scores"]:
        st.write(
            f"- {row['chunk_id']} | {row['score']:.4f} | "
            f"{row['file_name']} | p{row['paragraph_number']}"
        )
    st.write("Retrieved files:")
    for item in result["debug"]["retrieved_files"]:
        st.write(f"- {item}")
    st.markdown("</div>", unsafe_allow_html=True)
