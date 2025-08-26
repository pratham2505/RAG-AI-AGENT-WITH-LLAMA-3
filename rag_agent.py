# # rag_agent.py
# import logging, os, torch
# from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
# from llama_index.core.settings import Settings
# from llama_index.llms.huggingface import HuggingFaceLLM
# from llama_index.core.prompts import PromptTemplate
# from llama_index.embeddings.langchain import LangchainEmbedding
# from langchain_huggingface import HuggingFaceEmbeddings

# logging.getLogger("pypdf").setLevel(logging.ERROR)

# # Prompts…
# SYSTEM_PROMPT = """…"""
# QUERY_PROMPT = PromptTemplate(
#     "<|begin_of_text|><|start_header_id|>user<|end_header_id|>{query_str}"
#     "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
# )

# # Fetch token
# hf_token = os.environ["HUGGINGFACEHUB_API_TOKEN"]

# # 1) Init LLaMA (quantized 4‑bit on CPU!)
# llm = HuggingFaceLLM(
#     context_window=8192,
#     max_new_tokens=1024,
#     generate_kwargs={"do_sample": False, "temperature": 0.0},
#     system_prompt=SYSTEM_PROMPT,
#     query_wrapper_prompt=QUERY_PROMPT,
#     tokenizer_name="meta-llama/Llama-3.2-3B-Instruct",
#     model_name="meta-llama/Llama-3.2-3B-Instruct",
#     device_map="auto",
#     tokenizer_kwargs={"trust_remote_code": True},
#     model_kwargs={
#         "torch_dtype": torch.float16,
#         "load_in_4bit": True,            # ← 4‑bit quantization
#         "bnb_4bit_quant_type": "nf4",
#         "token": hf_token,               # new Transformers arg
#     },
# )

# # 2) Embedding model
# embed_model = LangchainEmbedding(
#     HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
# )

# # 3) Apply settings
# Settings.llm = llm
# Settings.embed_model = embed_model
# Settings.chunk_size = 512

# # 4) Build / cache index
# _index = None
# def load_index(data_path: str = "data/"):
#     global _index
#     if _index is None:
#         docs = SimpleDirectoryReader(data_path).load_data()
#         _index = VectorStoreIndex.from_documents(docs)
#     return _index

# # 5) Build one query‐engine
# def make_query_engine(data_path: str = "data/"):
#     idx = load_index(data_path)
#     return idx.as_query_engine(
#         response_mode="tree_summarize",
#         similarity_top_k=5,    # smaller K = faster
#         streaming=False
#     )

# rag_agent.py
# CPU-only RAG agent with dynamic quantization, persisted index, and HuggingFace token support

import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# llama-index imports (v0.12.45+ layout)
from llama_index.core.storage.storage_context import StorageContext
from llama_index.core.indices.vector_store.base import VectorStoreIndex
from llama_index.core import SimpleDirectoryReader
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
# Load HuggingFace API token (populated via Streamlit secrets)
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
if hf_token:
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = hf_token

# Configuration
MODEL_NAME = "meta-llama/Llama-2-7b-chat-hf"
DATA_PATH   = "data/"
PERSIST_DIR = "storage/"

SYSTEM_PROMPT = """
You are an expert research assistant. For every question, produce a clear,
multi-paragraph answer with background, explanation, and examples.
Avoid one-line replies—explain your reasoning.
"""

def load_model_and_tokenizer(model_name: str = MODEL_NAME):
    # 1) Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_auth_token=hf_token,
    )

    # 2) Load model in float32
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        use_auth_token=hf_token,
    )
    model.eval()

    # 3) Dynamic quantization on CPU (Linear → int8)
    model = torch.quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8
    )

    model.to("cpu")
    return model, tokenizer

def load_or_build_index(data_path: str = DATA_PATH, persist_dir: str = PERSIST_DIR):
    # try loading existing…
    if os.path.isdir(persist_dir):
        try:
            storage_ctx = StorageContext.from_defaults(persist_dir=persist_dir)
            return VectorStoreIndex.from_storage_context(storage_ctx)
        except Exception:
            print("⚠️ invalid index, rebuilding…")

    os.makedirs(persist_dir, exist_ok=True)

    # **1) Read docs**
    docs = SimpleDirectoryReader(data_path).load_data()

    # **2) Create your HF embedding model**
    hf_embed = HuggingFaceEmbedding(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        encode_kwargs={"normalize_embeddings": True}
    )

    # **3) Build the index with the HuggingFace embedder**
    index = VectorStoreIndex.from_documents(
        docs,
        embed_model=hf_embed,           # <-- here!
    )

    # 4) Persist for next time
    index.storage_context.persist(persist_dir=persist_dir)
    return index

_index = None
def get_index(data_path: str = DATA_PATH):
    global _index
    if _index is None:
        _index = load_or_build_index(data_path, PERSIST_DIR)
    return _index

def make_query_engine(data_path: str = DATA_PATH):
    # 1) Load or build the vector index
    idx = get_index(data_path)

    # 2) Load (and quantize) the LLM + tokenizer
    model, tokenizer = load_model_and_tokenizer()
    llm = HuggingFaceLLM(
        model=model,
        tokenizer=tokenizer,
        system_prompt=SYSTEM_PROMPT,
        generate_kwargs={
            "do_sample": True,
            "temperature": 0.7,
            "top_p": 0.9,
        },
    )

    # 3) Return a streaming “refine” query engine
    return idx.as_query_engine(
        llm=llm,
        response_mode="refine",
        similarity_top_k=3,
        streaming=True,
    )
