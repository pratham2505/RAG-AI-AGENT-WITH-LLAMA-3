# # app.py
# import os, streamlit as st
# from huggingface_hub import login

# # — Auth —
# hf_token = st.secrets["HUGGINGFACEHUB_API_TOKEN"]
# os.environ["HUGGINGFACEHUB_API_TOKEN"] = hf_token
# os.environ["HF_HUB_TOKEN"]            = hf_token
# login(token=hf_token)

# # — Build engine once —  
# @st.cache_resource
# def init_engine():
#     from rag_agent import make_query_engine
#     return make_query_engine(data_path="data/")

# with st.spinner("🔄 Initializing model + index (one‑time)…"):
#     query_engine = init_engine()

# # — UI —
# st.set_page_config(page_title="PDF Q&A", layout="wide")
# st.title("📖 Ask Your Binder PDF")

# question = st.text_input("Your question:")
# if st.button("Ask"):
#     if not question:
#         st.warning("Type a question first.")
#     else:
#         with st.spinner("🤔 Thinking…"):
#             answer = query_engine.query(question)
#         st.subheader("Answer")
#         st.text_area("✏️ Response", answer, height=400)

# app.py
# Streamlit front-end for the CPU-only RAG agent

import os
# ─── Load HuggingFace Token & Pin CPU threads before importing torch ─────────────────
os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HUGGINGFACEHUB_API_TOKEN")  # will be populated from .streamlit/secrets.toml
os.environ["OMP_NUM_THREADS"] = "16"       # set to your physical core count
os.environ["MKL_NUM_THREADS"] = "16"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
torch.set_num_threads(16)

import streamlit as st
from rag_agent import make_query_engine


def main():
    st.set_page_config(page_title="RAG AI Agent (CPU)", layout="wide")
    st.title("📚 Retrieval-Augmented Generation (CPU-Only)")
    st.markdown(
        "Ask questions about your PDF/documents (in data/) and get detailed, multi-paragraph answers."
    )

    # Ensure HuggingFace token is set in environment
    if not os.getenv("HUGGINGFACEHUB_API_TOKEN"):
        st.error("HuggingFace Hub API token not found. Please add it to .streamlit/secrets.toml under HUGGINGFACEHUB_API_TOKEN.")
        return

    question = st.text_input("Enter your question:")
    if st.button("Submit") and question:
        # Build (or fetch cached) query engine
        with st.spinner("Thinking… this may take a few seconds on CPU…"):
            qe = make_query_engine()
            # Streaming answer
            response = qe.query(question)

        st.markdown("---")
        st.markdown("**Answer:**")
        st.write(response)


if __name__ == "__main__":
    main()
