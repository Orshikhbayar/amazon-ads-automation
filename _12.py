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
def generate_segments(brief, retrieved_docs, rdb=None):
    """
    Generate Japanese audience segments quickly and reuse cached results.
    """
    import json, os, time
    from openai import OpenAI

    cache_key = f"generate:{brief}"
    if rdb and rdb.exists(cache_key):
        print("⚡ Returning cached generation result")
        return json.loads(rdb.get(cache_key))

    start = time.time()
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    model = os.getenv("OPENAI_GEN_MODEL", "gpt-4o-mini")

    # Trim long texts for token efficiency
    docs_text = "\n\n".join([
        f"【セグメント名】{d['keyword']}\n{d['text'][:500]}"
        for d in retrieved_docs
    ])

    prompt = f"""
あなたは日本のAmazonマーケティング戦略担当者です。
以下のキャンペーン概要と関連する商品情報をもとに、3〜5個のターゲットオーディエンスセグメントをJSON形式で出力してください。
出力は必ず日本語で記述してください。

【キャンペーン概要】
{brief}

【関連ドキュメント】
{docs_text}

出力フォーマット（必ずJSON配列で出力してください）:
[
  {{
    "name": "セグメント名",
    "reason": "なぜこのセグメントが適しているのか（1〜2文）",
    "keywords": ["キーワード1", "キーワード2", "キーワード3"],
    "headlines": ["見出し1", "見出し2"],
    "description": "セグメントの詳細説明（2〜3文）"
  }}
]
"""

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "あなたは日本のAmazonマーケティング専門家です。必ず有効なJSON形式で応答してください。マークダウンや説明文は含めないでください。"},
                {"role": "user", "content": prompt},
            ],
            temperature=0.5,
            max_completion_tokens=800,
        )

        text = resp.choices[0].message.content.strip()
        
        # Strip markdown code fences if present
        if text.startswith("```"):
            lines = text.split("\n")
            # Remove first line (```json or ```) and last line (```)
            text = "\n".join(lines[1:-1]) if len(lines) > 2 else text
            text = text.strip()
        
        try:
            parsed = json.loads(text)
            # Validate structure
            if not isinstance(parsed, list):
                raise ValueError("Response must be a JSON array")
            
            # Fallback: fill reason if missing
            for seg in parsed:
                if not seg.get("reason") and seg.get("description"):
                    seg["reason"] = seg["description"].split("。")[0] + "。"
            
            # Ensure all required fields exist
            for segment in parsed:
                if "name" not in segment:
                    segment["name"] = "Unnamed Segment"
                if "reason" not in segment:
                    segment["reason"] = "No description"
                if "keywords" not in segment:
                    segment["keywords"] = []
                if "headlines" not in segment:
                    segment["headlines"] = []
                if "description" not in segment:
                    segment["description"] = segment.get("reason", "No description")
                    
        except Exception as e:
            print(f"⚠️ Could not parse JSON: {e}")
            print(f"Raw response: {text[:200]}...")
            # Return structured fallback
            parsed = [{
                "name": "Unnamed Segment",
                "reason": "No description",
                "keywords": [],
                "headlines": [],
                "description": text[:500]
            }]

        if rdb:
            rdb.setex(cache_key, 600, json.dumps(parsed, ensure_ascii=False))

        print(f"✅ Generation done in {round(time.time()-start,2)}s ({model})")
        return parsed

    except Exception as e:
        print("❌ Generation error:", e)
        return [{"error": str(e)}]



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
