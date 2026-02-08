# ===============================
# AI FAISS READER (NO CHANGES)
# ===============================

import faiss
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()
client = OpenAI()

# ---- Paths (must match Original) ----
FAISS_INDEX_PATH = "faiss_index_ai.bin"
TEXT_STORE_PATH  = "stored_texts_ai.txt"

# ---- Load FAISS + texts once ----
if not os.path.exists(FAISS_INDEX_PATH) or not os.path.exists(TEXT_STORE_PATH):
    raise FileNotFoundError("FAISS index or stored texts not found. Run AI Original first.")

index = faiss.read_index(FAISS_INDEX_PATH)

with open(TEXT_STORE_PATH, "r", encoding="utf-8") as f:
    stored_texts = [line.strip() for line in f if line.strip()]

# ---- Embedding helper ----
def embed_texts(texts):
    resp = client.embeddings.create(
        model="text-embedding-3-large",
        input=texts
    )
    return np.array([d.embedding for d in resp.data], dtype="float32")

# ---- Semantic search ----
def semantic_search(query: str, top_k: int = 10, threshold: float = 0.625):
    query = f"Explain or test knowledge about: {query}"

    q_emb = embed_texts([query])
    faiss.normalize_L2(q_emb)

    scores, indices = index.search(q_emb, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if score >= threshold:
            results.append((float(score), stored_texts[idx]))

    return results
