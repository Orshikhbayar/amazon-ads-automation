#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import sys
import json
import subprocess
import re

# ---------------------------
# PATH HELPERS
# ---------------------------
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

# Check environment vars
required_env = ['OPENAI_API_KEY', 'EMBEDDING_BACKEND', 'EMBEDDING_MODEL']
missing_env = [v for v in required_env if not os.getenv(v)]
if missing_env:
    print(f"⚠️ Missing environment variables: {', '.join(missing_env)}")

# ---------------------------
# ROUTES
# ---------------------------
@app.route("/")
def index():
    """Serve frontend HTML."""
    html_path = os.path.join(PUBLIC_DIR, "index.html")
    if os.path.exists(html_path):
        return send_from_directory(PUBLIC_DIR, "index.html")
    return "<h1>API is running</h1>", 200

@app.route("/health")
def health():
    return jsonify({"status": "ok"})

# ---------------------------
# SUBPROCESS WRAPPER
# ---------------------------
def run_12py(args_list):
    """Run 12.py safely from absolute path."""
    script_path = os.path.join(BASE_DIR, "12.py")
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

        args = [
            "--brief", campaign_brief,
            "--top-k", str(top_k),
            "--kw-weight", str(keyword_weight),
            "--retrieval-only"
        ]
        if not enable_keywords:
            args.append("--no-extract")

        code, out, err = run_12py(args)
        if code != 0:
            return jsonify({"error": err}), 500

        segments = parse_retrieval_output(out)
        return jsonify({"segments": segments, "total_found": len(segments)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ---------------------------
# GENERATE SEGMENTS
# ---------------------------
@app.route("/api/generate", methods=["POST"])
def generate_segments():
    try:
        data = request.get_json(force=True)
        campaign_brief = data.get("campaign_brief", "").strip()
        top_k = int(data.get("top_k", 3))
        keyword_weight = float(data.get("keyword_weight", 0.4))
        enable_keywords = bool(data.get("enable_keywords", True))

        if not campaign_brief:
            return jsonify({"error": "Campaign brief is required"}), 400

        args = [
            "--brief", campaign_brief,
            "--top-k", str(top_k),
            "--kw-weight", str(keyword_weight)
        ]
        if not enable_keywords:
            args.append("--no-extract")

        code, out, err = run_12py(args)
        if code != 0:
            return jsonify({"error": err}), 500

        segments, generated = parse_full_output(out, campaign_brief)
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
# MAIN
# ---------------------------
if __name__ == "__main__":
    print("🚀 Starting Flask server...")
    app.run(host="0.0.0.0", port=5000, debug=True)
