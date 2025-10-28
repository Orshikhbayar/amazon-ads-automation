#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
12.py — OpenAI-only (requires OPENAI_API_KEY)

- Embeddings: utils.embedding.get_embedding (set EMBEDDING_BACKEND=openai)
- Retrieval: FAISS cosine (index from ingest_index_json.py)
- Output 1: Scored matches (from YOUR data) with cosine, match %, est CTR
- Output 2: ONE formatted "Proposed Target Segments" section (grounded on your data)
  * Uses ONLY retrieved segment names (no inventing)
  * Descriptions/headlines must be based on the retrieved Text
- Output 3: Automatically saves generated segments to generated_segments.jsonl
"""

import os, sys, json, argparse, re
import numpy as np
import faiss
from openai import OpenAI, BadRequestError
from utils.embedding import get_embedding
from datetime import datetime

# ---------------------------
# Config
# ---------------------------
GEN_MODEL = os.getenv("OPENAI_GEN_MODEL", "gpt-4o-mini")
DEBUG = False  # Will be set from args

def _path(*parts):
    p1 = os.path.join("data", *parts)
    p2 = os.path.join("Data", *parts)
    return p1 if os.path.exists(p1) else p2

INDEX_PATH = _path("faiss.index")
DOCS_PATH = _path("docs.jsonl")
JAPAN_MAP_PATH = _path("japan.json")

# ---------------------------
# Sanity: index & docs
# ---------------------------
missing = [p for p in (INDEX_PATH, DOCS_PATH) if not os.path.exists(p)]
if missing:
    sys.exit("❌ Missing files:\n  " + "\n  ".join(missing) +
             "\nTip: re-run `ingest_index_json.py` with your current EMBEDDING_MODEL.")

index = faiss.read_index(INDEX_PATH)
docs = [json.loads(l) for l in open(DOCS_PATH, "r", encoding="utf-8")]

# Load Japanese name mapping
japanese_names = {}
if os.path.exists(JAPAN_MAP_PATH):
    with open(JAPAN_MAP_PATH, "r", encoding="utf-8") as f:
        japanese_names = json.load(f)
else:
    print(f"⚠️  WARNING: Japanese mapping file not found at {JAPAN_MAP_PATH}")
    print(f"    Segment names will remain in English.")

# ---------------------------
# Helpers
# ---------------------------
_translation_cache = {}

def debug_print(msg):
    if DEBUG:
        print(msg)

def has_japanese(text: str) -> bool:
    if not text:
        return False
    return any('\u3040' <= c <= '\u9FAF' for c in text)

def get_japanese_name(english_name: str) -> str:
    return japanese_names.get(english_name, english_name)

def _normalize(v: np.ndarray) -> np.ndarray:
    v = v.astype("float32")
    n = np.linalg.norm(v) + 1e-12
    return v / n

def _percent_from_cos(cos_val: float) -> float:
    return max(0.0, min(1.0, (cos_val + 1.0) / 2.0)) * 100.0

def _tokenize_lower(s: str) -> set:
    if not s:
        return set()
    english_terms = set(re.findall(r"[a-zA-Z0-9\-\+/#\.]+", s.lower()))
    japanese_terms = set(re.findall(r"[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FAF]+", s))
    return english_terms | japanese_terms

def estimate_ctr_percent(row: dict, base_ctr_pct: float = 1.0) -> float:
    score_factor = 0.5 + 0.5 * (row["match_pct"] / 100.0)
    hit_count = len(row.get("hits_brief", []))
    kw_bonus = min(1.25, 1.0 + 0.02 * hit_count)
    return round(base_ctr_pct * score_factor * kw_bonus, 2)

# ---------------------------
# OpenAI Client
# ---------------------------
def _openai_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set.")
    return OpenAI(api_key=api_key)

# ---------------------------
# Translation
# ---------------------------
def translate_japanese_to_english(text: str) -> str:
    cache_key = text[:100]
    if cache_key in _translation_cache:
        return _translation_cache[cache_key]
    if not has_japanese(text):
        return text

    client = _openai_client()
    sys_msg = (
        "Translate the following Japanese text to English accurately for marketing context. "
        "Return ONLY the English translation."
    )
    try:
        resp = client.chat.completions.create(
            model=GEN_MODEL,
            messages=[{"role": "system", "content": sys_msg},
                      {"role": "user", "content": text}],
            temperature=0.3,
            max_completion_tokens=300,
            timeout=30
        )
        english_text = resp.choices[0].message.content.strip()
        _translation_cache[cache_key] = english_text
        return english_text
    except Exception as e:
        print(f"⚠️  Translation failed: {e}, using original text")
        return text

# ---------------------------
# Question normalization
# ---------------------------
def normalize_question_to_statement(text: str) -> str:
    """Convert natural-language questions into descriptive statements."""
    if "？" in text or text.strip().endswith("?"):
        try:
            client = _openai_client()
            sys_msg = (
                "Rephrase the following Japanese or English question into a short declarative "
                "statement describing the target audience or intent, suitable for advertising. "
                "Return only one clean sentence."
            )
            resp = client.chat.completions.create(
                model=GEN_MODEL,
                messages=[{"role": "system", "content": sys_msg},
                          {"role": "user", "content": text}],
                temperature=0.3,
                max_completion_tokens=80
            )
            out = resp.choices[0].message.content.strip()
            debug_print(f"Normalized question → {out}")
            return out
        except Exception as e:
            debug_print(f"Normalization failed: {e}")
            return text
    return text

# ---------------------------
# Step 1: AI keyword extraction
# ---------------------------
def extract_keywords_ai(brief: str, max_terms: int = 12) -> list[str]:
    client = _openai_client()
    sys_msg = (
        "Extract up to 12 concise keywords (English and/or Japanese) from this brief "
        "that describe audience or product characteristics. Return ONLY a JSON array of strings."
    )
    try:
        resp = client.chat.completions.create(
            model=GEN_MODEL,
            messages=[{"role": "system", "content": sys_msg},
                      {"role": "user", "content": brief}],
            temperature=0.7,
            max_completion_tokens=150,
            timeout=30
        )
        content = resp.choices[0].message.content.strip()
        arr = json.loads(content)
        out = [x.strip() for x in arr if isinstance(x, str)]
        return out[:max_terms]
    except Exception:
        toks = [t for t in _tokenize_lower(brief) if len(t) > 1]
        return list(toks)[:max_terms]

# ---------------------------
# Step 2: Retrieval
# ---------------------------
def retrieve_segments_detailed(brief: str, top_k: int = 3, use_extract: bool = True, kw_weight: float = 0.4):
    try:
        # Normalize questions first
        brief = normalize_question_to_statement(brief)

        if not brief or len(brief.strip()) < 10:
            return [], [], "Campaign brief too short (minimum 10 characters)"

        english_brief = translate_japanese_to_english(brief)
        emb_brief = _normalize(get_embedding(english_brief))

        ai_kws = extract_keywords_ai(english_brief) if use_extract else []
        q_vec = emb_brief
        if ai_kws:
            emb_kw = _normalize(get_embedding(", ".join(ai_kws)))
            q_vec = _normalize((1 - kw_weight) * emb_brief + kw_weight * emb_kw)

        rows, seen = [], set()
        search_size = max(top_k * 5, 20)
        max_search_size = min(len(docs), 200)

        while len(rows) < top_k and search_size <= max_search_size:
            D, I = index.search(np.array([q_vec], dtype="float32"), search_size)
            for idx in I[0]:
                if len(rows) >= top_k: break
                rec = docs[idx]; key = rec.get("keyword", f"seg_{idx}")
                if key in seen: continue
                seen.add(key)
                cos = float(D[0][0]); cos = max(0.0, min(1.0, cos))
                if cos >= 0.5:
                    pct = _percent_from_cos(cos)
                    rows.append({
                        "keyword": key,
                        "text": rec.get("text") or rec.get("answer") or "",
                        "cosine": cos,
                        "match_pct": pct,
                        "hits_ai": [],
                        "hits_brief": []
                    })
            if len(rows) < top_k:
                search_size = min(search_size * 2, max_search_size)
            else:
                break

        if not rows:
            return [], [], "No matching segments found. Try different keywords or lower the match threshold."
        return rows, ai_kws, None
    except Exception as e:
        return [], [], f"Retrieval error: {e}"

# ---------------------------
# Step 3: Generation
# ---------------------------
STRICT_RULES = """
HARD RULES:
- Use ONLY segment names from Allowed Segment Names (verbatim).
- Do NOT invent or rephrase segment names.
- For **Keywords** and **Description**, use ONLY terms and facts found in the Text.
"""

INSTRUCTIONS = """
You are an Amazon Ads strategist.
Always respond entirely in Japanese.

Propose relevant target segments with:
- 'Why it fits' (1–2 lines in Japanese),
- 6–10 Keywords (from the Text),
- Two Headlines,
- A short Description (≤150 chars).
"""

def build_prompt_strict(campaign_brief, retrieved_rows, allowed_names):
    blocks = []
    for r in retrieved_rows:
        blocks.append(f"Keyword: {get_japanese_name(r['keyword'])}\nText: {r['text'][:320]}")
    return f"""{INSTRUCTIONS}
{STRICT_RULES}

=== Campaign Brief ===
{campaign_brief}

=== Retrieved Segments ===
{chr(10).join(blocks)}

=== Allowed Segment Names ===
{', '.join(allowed_names)}
""".strip()

def generate_with_openai(prompt: str, model: str = GEN_MODEL) -> str:
    client = _openai_client()
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_completion_tokens=600,
            timeout=30
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"⚠️ Generation failed: {e}"

# ---------------------------
# Display & Save
# ---------------------------
def print_matches(rows, ai_kws, base_ctr_pct=1.0):
    print("\n🔎 Matched segments (from YOUR data):\n")
    for i, r in enumerate(rows, 1):
        est_ctr = estimate_ctr_percent(r, base_ctr_pct)
        name = get_japanese_name(r['keyword'])
        print(f"{i}) {name}")
        print(f"   • score: {r['cosine']:.3f} | match: {r['match_pct']:.1f}% | est CTR: {est_ctr:.2f}%")

def save_generation(brief, ai_kws, rows, md_output, path="generated_segments.jsonl"):
    try:
        data = {
            "timestamp": datetime.utcnow().isoformat(),
            "brief": brief,
            "ai_keywords": ai_kws,
            "retrieved_segments": [r["keyword"] for r in rows],
            "scores": [{"keyword": r["keyword"], "match_pct": r["match_pct"], "cosine": r["cosine"]} for r in rows],
            "output_markdown": md_output
        }
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")
    except Exception as e:
        debug_print(f"⚠️ Could not save: {e}")

# ---------------------------
# CLI
# ---------------------------
def main():
    global DEBUG
    ap = argparse.ArgumentParser(description="Amazon Ads Automation - Segment Generator")
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--brief", type=str)
    ap.add_argument("--no-extract", action="store_true")
    ap.add_argument("--kw-weight", type=float, default=0.5)
    ap.add_argument("--retrieval-only", action="store_true")
    ap.add_argument("--base-ctr", type=float, default=1.0)
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()
    DEBUG = args.debug

    brief = args.brief or input("Enter campaign brief: ").strip()
    if not brief:
        print("❌ Campaign brief is required")
        return

    rows, ai_kws, error = retrieve_segments_detailed(
        brief,
        top_k=args.top_k,
        use_extract=not args.no_extract,
        kw_weight=max(0.0, min(1.0, args.kw_weight)),
    )
    if error:
        print(f"\n❌ {error}")
        return

    print_matches(rows, ai_kws, base_ctr_pct=args.base_ctr)

    if args.retrieval_only:
        return

    allowed = [get_japanese_name(r["keyword"]) for r in rows]
    prompt = build_prompt_strict(brief, rows, allowed)
    md = generate_with_openai(prompt)
    print("\n" + md + "\n")
    save_generation(brief, ai_kws, rows, md)

if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
