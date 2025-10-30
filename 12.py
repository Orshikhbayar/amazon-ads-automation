#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
12.py — Unified Retrieval + Generation Engine

Features:
- Hybrid FAISS + BM25 retrieval
- Redis caching for embeddings & results
- Compatible with Flask current_app context
"""

import os, sys, json, numpy as np
import faiss
from flask import current_app
from embedding import get_embeddings_batch
from rank_bm25 import BM25Okapi
from openai import OpenAI
from flask import has_app_context, current_app
# ---------------------------
# CONFIG
# ---------------------------
TOP_K_DEFAULT = 5
KW_WEIGHT_DEFAULT = 0.4


# ---------------------------
# EMBEDDING UTIL
# ---------------------------
def embed_text(text, cache_key=None, rdb=None):
    """Embed a single text with Redis caching."""
    if rdb and cache_key and rdb.exists(cache_key):
        return np.array(json.loads(rdb.get(cache_key)), dtype="float32")

    vec = get_embeddings_batch([text])[0].astype("float32")

    if rdb and cache_key:
        rdb.set(cache_key, json.dumps(vec.tolist()))
    return vec


# ---------------------------
# RETRIEVAL CORE
# ---------------------------
def retrieve_docs(brief, top_k=TOP_K_DEFAULT, kw_weight=KW_WEIGHT_DEFAULT, index=None, docs=None, rdb=None):
    """Hybrid retrieval from FAISS + BM25, with Flask and standalone fallback."""

    try:
        from flask import has_app_context, current_app
        in_context = has_app_context()
    except Exception:
        in_context = False

    if in_context:
        index = index or getattr(current_app, "faiss_index", None)
        docs = docs or getattr(current_app, "docs", None)
        bm25 = getattr(current_app, "bm25", None)
        rdb = rdb or getattr(current_app, "rdb", None)
        print("🧠 Using Flask app context for retrieval.")
    else:
        # Fallback: manually enter app context
        try:
            from app import app
            with app.app_context():
                index = index or getattr(app, "faiss_index", None)
                docs = docs or getattr(app, "docs", None)
                bm25 = getattr(app, "bm25", None)
                rdb = rdb or getattr(app, "rdb", None)
                print("🔁 Loaded FAISS/docs via app.app_context() fallback.")
        except Exception as e:
            print(f"⚠️ Failed to load app context: {e}")

    if not index or not docs:
        raise RuntimeError("❌ FAISS index or docs not loaded (both context and fallback failed)")

    cache_key = f"embed:{brief}"
    q_emb = embed_text(brief, cache_key, rdb)

    q_emb = q_emb / (np.linalg.norm(q_emb) + 1e-12)
    sims, idxs = index.search(np.array([q_emb], dtype="float32"), top_k)
    sims, idxs = sims[0], idxs[0]

    # Normalize FAISS similarity scores to 0–100%
    max_sim = float(np.max(sims)) if np.max(sims) > 0 else 1.0
    faiss_results = [
        {
            "keyword": docs[i]["keyword"],
            "text": docs[i]["text"],
            "score": round((float(sims[j]) / max_sim) * 100, 1)
        }
        for j, i in enumerate(idxs)
        if i != -1
    ]

    bm25_results = []
    if 'bm25' in locals() and bm25:
        bm_scores = bm25.get_scores(brief.split())
        bm_ranked = np.argsort(bm_scores)[::-1][:top_k]
        bm25_results = [
            {"keyword": docs[i]["keyword"], "text": docs[i]["text"], "score": float(bm_scores[i])}
            for i in bm_ranked
        ]

    combined = {}
    for item in faiss_results:
        combined[item["keyword"]] = kw_weight * item["score"]
    for item in bm25_results:
        combined[item["keyword"]] = combined.get(item["keyword"], 0) + (1 - kw_weight) * item["score"]

    ranked = sorted(combined.items(), key=lambda x: x[1], reverse=True)[:top_k]
    retrieved = [
        {"keyword": k, "score": round(v, 2), "text": next(d["text"] for d in docs if d["keyword"] == k)}
        for k, v in ranked
    ]
    
    if rdb:
        rdb.set(f"retrieval:{brief}", json.dumps(retrieved, ensure_ascii=False))

    return retrieved



# ---------------------------
# GENERATION
# ---------------------------
def generate_segments(brief, retrieved_docs):
    """Generate audience segments via OpenAI."""
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    docs_text = "\n\n".join([f"- {d['keyword']}: {d['text']}" for d in retrieved_docs])
    prompt = f"""
You are a Japanese Amazon marketing strategist.
Given the campaign brief and related documents, identify 3–5 target audience segments.

Brief:
{brief}

Documents (each has a segment name and text):
{ "\n".join([f"{d['keyword']}: {d['text']}" for d in retrieved_docs]) }
Please write audience segments using the same names (keywords) as shown above when possible.

Format each segment as:
**Segment N: <segment name>**
**Why it fits:** ...
**Keywords:** ...
• headline1
• headline2
**Description:** ...
"""

    resp = client.chat.completions.create(
        model=os.getenv("OPENAI_GEN_MODEL", "gpt-4o-mini"),
        messages=[
            {"role": "system", "content": "You are an expert Japanese Amazon marketer."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.5,
        max_completion_tokens=800,
    )

    text = resp.choices[0].message.content.strip()
    return text


# ---------------------------
# MAIN ENTRY (CLI)
# ---------------------------
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--brief", required=True)
    parser.add_argument("--top-k", type=int, default=TOP_K_DEFAULT)
    parser.add_argument("--kw-weight", type=float, default=KW_WEIGHT_DEFAULT)
    parser.add_argument("--retrieval-only", action="store_true")
    args = parser.parse_args()

    retrieved = retrieve_docs(args.brief, args.top_k, args.kw_weight)

    print("💡 Retrieved Segments:")
    for i, r in enumerate(retrieved, 1):
        print(f"{i}) {r['keyword']} — match: {r['score']}%")

    if args.retrieval_only:
        return

    print("\n💡 Proposed Target Segments")
    generated = generate_segments(args.brief, retrieved)
    print(generated)


if __name__ == "__main__":
    main()
