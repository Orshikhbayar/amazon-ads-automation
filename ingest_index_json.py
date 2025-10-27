#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ingest_index_json.py
Build a cosine-similarity FAISS index from keyword_with_answer.json.

- Accepts EITHER:
  1) {"keyword A": "answer A", "keyword B": "answer B", ...}  <-- your file
  2) [{"keyword": "...", "answer": "..."}, ...]
- Embeds "Keyword: <k>\\nText: <answer>" so both fields influence retrieval
- L2-normalizes embeddings and uses Inner Product (cosine)

Env:
  EMBEDDING_BACKEND=openai|local
  EMBEDDING_MODEL=...   (e.g., text-embedding-3-small or BAAI/bge-small-en-v1.5)

Outputs:
  data/faiss.index
  data/docs.jsonl
"""

import os, json, sys
import numpy as np
import faiss
from utils.embedding import get_embeddings_batch

def _path(*parts):
    p1 = os.path.join("data", *parts)
    p2 = os.path.join("Data", *parts)
    return p1 if os.path.exists(p1) or not os.path.exists(p2) else p2

INPUT_JSON   = _path("docs_japan.json")
OUTPUT_INDEX = _path("faiss2.index")
OUTPUT_DOCS  = _path("docs2.jsonl")

if not os.path.exists(INPUT_JSON):
    sys.exit(f"❌ Missing input: {INPUT_JSON}")

# -------- Load & normalize to a list of {keyword, answer} --------
with open(INPUT_JSON, "r", encoding="utf-8") as f:
    raw = json.load(f)

docs = []
if isinstance(raw, dict):
    # mapping: keyword -> answer
    for k, a in raw.items():
        k_norm = (k or "").strip()
        a_norm = (a or "").strip()
        if not k_norm or not a_norm:
            continue
        docs.append({"keyword": k_norm, "answer": a_norm})
elif isinstance(raw, list):
    for d in raw:
        if not isinstance(d, dict): continue
        k = (d.get("keyword") or "").strip()
        a = (d.get("answer")  or "").strip()
        if not k or not a:
            continue
        docs.append({"keyword": k, "answer": a})
else:
    sys.exit("❌ Unsupported JSON structure. Use a dict mapping or a list of objects.")

if not docs:
    sys.exit("❌ No valid (keyword, answer) pairs found after cleaning.")

# -------- Build embedding corpus --------
texts = [f"Keyword: {d['keyword']}\nText: {d['answer']}" for d in docs]
print(f"Loaded {len(texts)} docs. Embedding...")

emb = get_embeddings_batch(texts).astype("float32")

# L2-normalize => cosine via inner product
norms = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12
emb = emb / norms

# -------- Build FAISS (IP) index --------
dim = emb.shape[1]
index = faiss.IndexFlatIP(dim)
index.add(emb)
faiss.write_index(index, OUTPUT_INDEX)

# -------- Save paired metadata --------
os.makedirs(os.path.dirname(OUTPUT_DOCS), exist_ok=True)
with open(OUTPUT_DOCS, "w", encoding="utf-8") as f:
    for d in docs:
        f.write(json.dumps({"keyword": d["keyword"], "text": d["answer"]}, ensure_ascii=False) + "\n")

print(f"✅ Indexed {len(docs)} docs → {OUTPUT_INDEX}")
print(f"🧾 Metadata saved → {OUTPUT_DOCS}")
