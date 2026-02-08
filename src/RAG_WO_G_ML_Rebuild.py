from langchain_community.document_loaders import Docx2txtLoader

#📘 MODULE 0 — Imports & Environment Setup

# Core
import os
import re
import math
from typing import List, Tuple

# Word document reading
from docx import Document

# Embeddings & LLM
from openai import OpenAI
import tiktoken

# Vector store
import faiss
import numpy as np

# Utilities
from dotenv import load_dotenv
load_dotenv()

client = OpenAI()

# 📘 MODULE 1 — Load Word Document

DOC_PATH = "ML_Question_Bank.docx"

def load_docx_text(path: str) -> str:
    doc = Document(path)
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    return "\n".join(paragraphs)

raw_text = load_docx_text(DOC_PATH)

# 📘 MODULE 2 — Normalize Text

def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n")
    text = re.sub(r"\n+", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()

text = normalize_text(raw_text)

# 📘 MODULE 3 — Remove Separator Lines

def remove_separator_lines(text: str) -> str:
    lines = text.splitlines()
    cleaned = []
    for line in lines:
        if len(re.findall(r"[A-Za-z0-9]", line)) < 3:
            continue
        cleaned.append(line)
    return "\n".join(cleaned)

text = remove_separator_lines(text)

# 📘 MODULE 4 — Strip MCQ Options (A/B/C/D)

def strip_mcq_options(text: str) -> str:
    # Removes options like A. ..., B) ..., etc.
    pattern = r"(?:\n|\s)[A-D][\.\)]\s.*?(?=(?:\n[A-D][\.\)]|\n|$))"
    return re.sub(pattern, "", text, flags=re.DOTALL)

text = strip_mcq_options(text)

# 📘 MODULE 5 — Split Into Questions

def split_questions(text: str) -> List[str]:
    parts = re.split(r"(?<=[\.\?])", text)
    return [p.strip() for p in parts if p.strip()]

questions = split_questions(text)

unique_questions = list(dict.fromkeys(questions))

questions = unique_questions


# 📘 MODULE 6 — Token Counting (Embedding Model)

encoding = tiktoken.get_encoding("cl100k_base")

def count_tokens(text: str) -> int:
    return len(encoding.encode(text))

# 📘 MODULE 7 — OpenAI Embedding Function

def embed_texts(texts: List[str]) -> np.ndarray:
    response = client.embeddings.create(
        model="text-embedding-3-large",
        input=texts
    )
    return np.array([d.embedding for d in response.data], dtype="float32")

# 📘 MODULE 8 — Summarization Function (Strict)

def summarize_question(question: str) -> str:
    prompt = (
        "Summarize the following question for semantic search.\n"
        "Preserve ALL constraints, technical terms, and intent.\n"
        "Do NOT add new information.\n\n"
        f"{question}"
    )

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0
    )
    return response.choices[0].message.content.strip()

# 📘 MODULE 9 — Semantic Verification

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

# 📘 MODULE 10 — Build FAISS Index (Option A Logic)


MAX_TOKENS = 500
VERIFY_THRESHOLD = 0.70

stored_texts = []
stored_embeddings = []

for q in questions:
    token_count = count_tokens(q)

    # Normal case
    if token_count <= MAX_TOKENS:
        emb = embed_texts([q])[0]
        stored_texts.append(q)
        stored_embeddings.append(emb)
        continue

    # Long question → summarize
    summary1 = summarize_question(q)
    emb_q, emb_s1 = embed_texts([q, summary1])

    sim1 = cosine_similarity(emb_q, emb_s1)

    if sim1 >= VERIFY_THRESHOLD:
        stored_texts.append(q)
        stored_embeddings.append(emb_s1)
        continue

    # Regenerate once
    summary2 = summarize_question(q)
    emb_s2 = embed_texts([summary2])[0]
    sim2 = cosine_similarity(emb_q, emb_s2)

    if sim2 >= VERIFY_THRESHOLD:
        stored_texts.append(q)
        stored_embeddings.append(emb_s2)
    else:
        # FALLBACK → embed original
        stored_texts.append(q)
        stored_embeddings.append(emb_q)


# 📘 MODULE 11 — Create FAISS Index

dimension = len(stored_embeddings[0])
index = faiss.IndexFlatIP(dimension)

vectors = np.vstack(stored_embeddings)
faiss.normalize_L2(vectors)
index.add(vectors)

# 📘 MODULE 12 — Save FAISS Index + Stored Texts (Mode A)


FAISS_INDEX_PATH = "faiss_index_ml.bin"
TEXT_STORE_PATH = "stored_texts_ml.txt"



def save_faiss_index(index, stored_texts_ml):
    # Save FAISS index
    faiss.write_index(index, FAISS_INDEX_PATH)

    # Save texts (one question per line)
    with open(TEXT_STORE_PATH, "w", encoding="utf-8") as f:
        for text in stored_texts:
            f.write(text.replace("\n", " ") + "\n")

    print("FAISS index and texts saved to disk.")

# Call this AFTER building index
save_faiss_index(index, stored_texts)


# 📘 MODULE 13 — Semantic Search Query

def semantic_search(query: str, top_k: int = 10, threshold: float = 0.625):
    query = f"Explain or test knowledge about: {query}"               
    q_emb = embed_texts([query])
    faiss.normalize_L2(q_emb)

    scores, indices = index.search(q_emb, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if score >= threshold:
            results.append((score, stored_texts[idx]))

    return results

# 📘 MODULE 14 — Run a Test Query

while True:
    query = input("\nEnter your query (type 'close' to exit): ").strip()
    
    if query.lower() == "close":
        print("Closing search.")
        break

    results = semantic_search(query)

    if not results:
        print("No results above threshold.")
        continue

    for score, text in results:
        print(f"\nScore: {score:.3f}\n{text}")


print("ML DB rebuild COMPLETE")



