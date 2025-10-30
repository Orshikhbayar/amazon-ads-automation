#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from flask import Flask, request, jsonify, send_from_directory, current_app
from flask_cors import CORS
import os
import sys
import json
import subprocess
import re
import faiss
import redis
import numpy as np
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi
import re
from _12 import retrieve_docs, generate_segments

# ---------------------------
# ENV + PATH HELPERS
# ---------------------------
load_dotenv(override=True)  # load .env at startup, override existing env vars

def _path(*parts):
    p1 = os.path.join("data", *parts)
    p2 = os.path.join("Data", *parts)
    return p1 if os.path.exists(p1) else p2

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PUBLIC_DIR = os.path.join(BASE_DIR, "public")

# ---------------------------
# LOAD JAPANESE MAPPINGS
# ---------------------------
JAPAN_MAP_PATH = _path("japan.json")
japanese_names = {}
if os.path.exists(JAPAN_MAP_PATH):
    with open(JAPAN_MAP_PATH, "r", encoding="utf-8") as f:
        japanese_names = json.load(f)

def get_japanese_name(english_name: str) -> str:
    return japanese_names.get(english_name, english_name)

# ---------------------------
# OPENAI TRANSLATION HELPER
# ---------------------------
def translate_keywords_to_japanese(keywords: list[str]) -> list[str]:
    """Translate a list of English keywords to Japanese using OpenAI."""
    if not keywords:
        return keywords
    
    from openai import OpenAI
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("⚠️ Missing OPENAI_API_KEY")
        return keywords

    client = OpenAI(api_key=api_key)
    keywords_text = ", ".join(keywords)
    sys_msg = (
        "Translate the following English keywords to Japanese. "
        "Return ONLY a JSON array of Japanese translations in the same order. "
        "Keep marketing and product terms natural for Japanese Amazon users."
    )

    messages = [
        {"role": "system", "content": sys_msg},
        {"role": "user", "content": keywords_text},
    ]

    try:
        resp = client.chat.completions.create(
            model=os.getenv("OPENAI_GEN_MODEL", "gpt-4o-mini"),
            messages=messages,
            temperature=0.3,
            max_completion_tokens=200
        )
        content = resp.choices[0].message.content.strip()
        translated = json.loads(content)
        if isinstance(translated, list) and len(translated) == len(keywords):
            return translated
    except Exception as e:
        print(f"⚠️ Translation error: {e}")
    return keywords

# ---------------------------
# FLASK APP SETUP
# ---------------------------
app = Flask(__name__)
CORS(app)

@app.route("/")
def index():
    return send_from_directory('public', 'index.html')

# ---------------------------
# ENV CHECKS
# ---------------------------
required_env = ['OPENAI_API_KEY', 'EMBEDDING_BACKEND', 'EMBEDDING_MODEL']
missing_env = [v for v in required_env if not os.getenv(v)]
if missing_env:
    print(f"⚠️ Missing environment variables: {', '.join(missing_env)}")

# ---------------------------
# 🔹 FAISS + REDIS PRELOAD
# ---------------------------
INDEX_PATH = _path("faiss.index")
DOCS_PATH = _path("docs.jsonl")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
try:
    rdb = redis.Redis.from_url(REDIS_URL, decode_responses=True)
    rdb.ping()
    print(f"✅ Connected to Redis at {REDIS_URL}")
except Exception as e:
    print(f"⚠️ Redis not available: {e}")
    rdb = None

import requests, tempfile

def load_index_from_url(url):
    print(f"🌐 Downloading FAISS index from {url}")
    r = requests.get(url)
    r.raise_for_status()
    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write(r.content)
        return faiss.read_index(f.name)

def load_docs_from_url(url):
    print(f"🌐 Downloading docs from {url}")
    r = requests.get(url)
    r.raise_for_status()
    return [json.loads(line) for line in r.text.splitlines()]

FAISS_URL = os.getenv("FAISS_URL")
DOCS_URL = os.getenv("DOCS_URL")

print("🔹 Loading FAISS + docs (local or remote)...")
try:
    if FAISS_URL and DOCS_URL:
        index = load_index_from_url(FAISS_URL)
        docs = load_docs_from_url(DOCS_URL)
    else:
        index = faiss.read_index(INDEX_PATH)
        docs = [json.loads(l) for l in open(DOCS_PATH, "r", encoding="utf-8")]

        index.nprobe = 10
        print(f"✅ Loaded {len(docs)} docs into memory")
except Exception as e:
    print(f"❌ Failed to load FAISS or docs: {e}")
    index, docs = None, []

    
# ---------------------------
# 🔹 BM25 HYBRID RETRIEVER
# ---------------------------


print("🔹 Building BM25 index...")
try:
    tokenized_corpus = [
        re.findall(r"\w+", (d.get("text", "") + " " + d.get("keyword", "")).lower())
        for d in docs
    ]
    bm25 = BM25Okapi(tokenized_corpus)
    print(f"✅ BM25 index built ({len(tokenized_corpus)} docs)")
except Exception as e:
    print(f"⚠️ BM25 build failed: {e}")
    bm25 = None


# Attach to Flask app (global state)
app.faiss_index = index
app.bm25 = bm25
app.docs = docs
app.rdb = rdb

# ---------------------------
# ROUTES
# ---------------------------
@app.route("/health")
def health():
    return jsonify({"status": "ok"})

# ---------------------------
# SUBPROCESS WRAPPER
# ---------------------------
def run_12py(args_list):
    """Run 12.py safely from absolute path."""
    script_path = os.path.join(BASE_DIR, "_12.py")
    if not os.path.exists(script_path):
        return 1, "", "12.py not found"
    cmd = [sys.executable, script_path] + args_list
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=BASE_DIR)
    return result.returncode, result.stdout, result.stderr

# ---------------------------
# RETRIEVE SEGMENTS
# ---------------------------
@app.route("/api/retrieve", methods=["POST"])
def retrieve_segments():
    try:
        data = request.get_json(force=True)
        campaign_brief = data.get("campaign_brief", "").strip()
        top_k = int(data.get("top_k", 5))
        keyword_weight = float(data.get("keyword_weight", 0.4))
        enable_keywords = bool(data.get("enable_keywords", True))

        if not campaign_brief:
            return jsonify({"error": "Campaign brief is required"}), 400

        # ✅ Access preloaded FAISS, BM25, Redis
        
        index = current_app.faiss_index
        docs = current_app.docs
        bm25 = current_app.bm25
        rdb = current_app.rdb

        cache_key = f"retrieve:{campaign_brief}:{top_k}:{keyword_weight}"
        if rdb and rdb.exists(cache_key):
            print("⚡ Returning cached retrieval result")
            return jsonify(json.loads(rdb.get(cache_key)))

        # ----------------------------
        # 🧠 BM25 fallback for short briefs
        # ----------------------------
        bm25_matches = []
        if bm25 and len(campaign_brief.split()) < 5:
            import re
            from rank_bm25 import BM25Okapi

            query_tokens = re.findall(r"\w+", campaign_brief.lower())
            scores = bm25.get_scores(query_tokens)
            top_bm25 = np.argsort(scores)[::-1][:top_k]
            for idx in top_bm25:
                bm25_matches.append({
                    "name": docs[idx]["keyword"],
                    "match_percent": round(scores[idx] * 100, 2)
                })
            if bm25_matches:
                print("🔹 Used BM25 fallback")
                result = {"segments": bm25_matches, "total_found": len(bm25_matches)}
                if rdb:
                    rdb.setex(cache_key, 300, json.dumps(result))
                return jsonify(result)

        # ----------------------------
        # 🧩 FAISS retrieval (default)
        # ----------------------------
        args = [
            "--brief", campaign_brief,
            "--top-k", str(top_k),
            "--kw-weight", str(keyword_weight),
            "--retrieval-only"
        ]
        if not enable_keywords:
            args.append("--no-extract")

               # ----------------------------
        # 🧩 FAISS retrieval (default)
        # ----------------------------
        segments = retrieve_docs(
            campaign_brief,
            top_k=top_k,
            kw_weight=keyword_weight, 
            index=current_app.faiss_index,
            docs=current_app.docs,
            rdb=current_app.rdb
        )

        result = {"segments": segments, "total_found": len(segments)}
        if rdb:
            rdb.setex(cache_key, 300, json.dumps(result))

        return jsonify(result)


        

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---------------------------
# GENERATE SEGMENTS
# ---------------------------
@app.route("/api/generate", methods=["POST"])
def generate_segments_route():
    try:
        data = request.get_json(force=True)
        campaign_brief = data.get("campaign_brief", "").strip()
        top_k = int(data.get("top_k", 3))
        keyword_weight = float(data.get("keyword_weight", 0.4))

        if not campaign_brief:
            return jsonify({"error": "Campaign brief is required"}), 400

        # ✅ Call retrieve_docs and generate_segments directly
        from _12 import retrieve_docs, generate_segments
        
        retrieved = retrieve_docs(
            campaign_brief,
            top_k=top_k,
            kw_weight=keyword_weight,
            index=current_app.faiss_index,
            docs=current_app.docs,
            rdb=current_app.rdb
        )
        
        # Format retrieved segments for response
        segments = [
            {
                "name": r["keyword"],
                "match_percent": r["score"]
            }
            for r in retrieved
        ]
        
        # Generate new segments
        generated = generate_segments(campaign_brief, retrieved, rdb=current_app.rdb)
        
        return jsonify({
            "segments": segments,
            "generated_segments": generated,
            "total_found": len(segments)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ---------------------------
# PARSERS
# ---------------------------
def parse_retrieval_output(output):
    segments = []
    lines = output.split("\n")
    for i, line in enumerate(lines):
        match = re.match(r"(\d+)\)\s+(.+)", line.strip())
        if match:
            name = match.group(2)
            if i + 1 < len(lines):
                score_line = lines[i + 1]
                m = re.search(r"match:\s*([\d.]+)%", score_line)
                if m:
                    pct = float(m.group(1))
                    jp_name = get_japanese_name(name)
                    segments.append({"name": jp_name, "match_percent": pct})
    return segments

def parse_full_output(output, brief=""):
    parts = output.split("💡 Proposed Target Segments")
    segs = parse_retrieval_output(parts[0])
    gens = []
    if len(parts) > 1:
        gens = parse_generated_segments(parts[1], brief)
    return segs, gens

def parse_generated_segments(md_text, brief=""):
    segs = []
    blocks = re.split(r"\*\*Segment \d+:", md_text)
    for block in blocks[1:]:
        seg = {}
        lines = block.strip().split("\n")
        if lines:
            seg["name"] = lines[0].replace("**", "").strip()
        why = re.search(r"\*\*Why it fits:\*\*\s*([^\*]+)", block)
        if why:
            seg["why_fits"] = why.group(1).strip()
        kws = re.search(r"\*\*Keywords:\*\*\s*([^\*]+)", block)
        if kws:
            kws_raw = [k.strip() for k in kws.group(1).split(",") if k.strip()]
            seg["keywords"] = translate_keywords_to_japanese(kws_raw)
        seg["headlines"] = [h.strip().strip('"\'') for h in re.findall(r"[•·]\s*([^\n]+)", block)]
        desc = re.search(r"\*\*Description:\*\*\s*([^\*]+)", block)
        if desc:
            seg["description"] = desc.group(1).strip()
        if seg.get("name"):
            segs.append(seg)
    return segs

# ---------------------------
# DEBUG & HEALTH ENDPOINTS
# ---------------------------

@app.route("/api/health", methods=["GET"])
def health_check():
    """Health check endpoint for monitoring and CI/CD."""
    try:
        # Check FAISS index
        faiss_loaded = hasattr(current_app, 'faiss_index') and current_app.faiss_index is not None
        
        # Check Redis connection
        redis_connected = False
        if hasattr(current_app, 'rdb') and current_app.rdb:
            try:
                current_app.rdb.ping()
                redis_connected = True
            except:
                pass
        
        # Check docs loaded
        docs_loaded = hasattr(current_app, 'docs') and current_app.docs is not None
        
        # Check BM25 index
        bm25_loaded = hasattr(current_app, 'bm25') and current_app.bm25 is not None
        
        status = {
            "status": "ok" if all([faiss_loaded, redis_connected, docs_loaded]) else "degraded",
            "faiss_loaded": faiss_loaded,
            "redis_connected": redis_connected,
            "docs_loaded": docs_loaded,
            "bm25_loaded": bm25_loaded,
            "docs_count": len(current_app.docs) if docs_loaded else 0,
            "timestamp": time.time()
        }
        
        return jsonify(status)
        
    except Exception as e:
        return jsonify({
            "status": "error",
            "error": str(e),
            "timestamp": time.time()
        }), 500


@app.route("/api/test", methods=["GET"])
def test_generation():
    """Static test endpoint for CI/smoke testing."""
    try:
        # Static test data
        test_brief = "テスト用キャンペーン"
        test_segments = [
            {
                "name": "テスト > サンプルセグメント",
                "reason": "これはテスト用のセグメントです。",
                "keywords": ["テスト", "サンプル", "デモ"],
                "headlines": ["テスト用見出し1", "テスト用見出し2"],
                "description": "システムの動作確認用のテストセグメントです。全ての機能が正常に動作していることを確認できます。"
            }
        ]
        
        return jsonify({
            "status": "test_ok",
            "campaign_brief": test_brief,
            "generated_segments": test_segments,
            "test_timestamp": time.time(),
            "message": "Test endpoint working correctly"
        })
        
    except Exception as e:
        return jsonify({
            "status": "test_error", 
            "error": str(e),
            "timestamp": time.time()
        }), 500


@app.route("/api/retrieve", methods=["POST"])
def retrieve_only():
    """Enhanced retrieve endpoint with debug information."""
    try:
        data = request.get_json(force=True)
        campaign_brief = data.get("campaign_brief", "").strip()
        top_k = int(data.get("top_k", 3))
        debug = data.get("debug", False)
        
        if not campaign_brief:
            return jsonify({"error": "Campaign brief is required"}), 400
        
        from _12 import retrieve_docs
        
        retrieved = retrieve_docs(
            campaign_brief,
            top_k=top_k,
            index=current_app.faiss_index,
            docs=current_app.docs,
            rdb=current_app.rdb
        )
        
        response = {
            "campaign_brief": campaign_brief,
            "retrieved_segments": retrieved,
            "total_found": len(retrieved)
        }
        
        # Add debug information if requested
        if debug:
            response["debug_info"] = {
                "faiss_scores": [r.get("faiss_score", 0) for r in retrieved],
                "bm25_scores": [r.get("bm25_score", 0) for r in retrieved],
                "matched_keywords": [r.get("keyword", "") for r in retrieved],
                "domain_boosts": [r.get("domain_boost", 1.0) for r in retrieved],
                "quality_score": retrieved[0].get("validation", {}).get("quality_score", 0) if retrieved else 0
            }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({"error": f"Retrieval failed: {str(e)}"}), 500


# ---------------------------
# MAIN
# ---------------------------
if __name__ == "__main__":
    import time
    print("🚀 Starting Flask server...")
    app.run(host="0.0.0.0", port=5000, debug=True)
