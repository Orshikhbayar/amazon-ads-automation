#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
123.py — OpenAI-only (requires OPENAI_API_KEY)

- Embeddings: utils.embedding.get_embedding (set EMBEDDING_BACKEND=openai)
- Retrieval: FAISS cosine (index from ingest_index_json.py)
- Output 1: Scored matches (from YOUR data) with cosine, match %, est CTR
- Output 2: ONE formatted "Proposed Target Segments" section (grounded on your data)
  * Uses ONLY retrieved segment names (no inventing)
  * Descriptions/headlines must be based on the retrieved Text
- Output 3: Automatically saves generated segments to generated_segments.jsonl

Usage:
  export OPENAI_API_KEY=sk-...
  export EMBEDDING_BACKEND=openai
  export EMBEDDING_MODEL="text-embedding-3-small"
  python3 123.py --top-k 3 --brief "Target SMB owners buying routers and labelers"

Flags:
  --no-extract       Disable AI keyword extraction (use brief only)
  --kw-weight 0.4    Blend weight for keyword embedding (0..1)
  --retrieval-only   Only print matches (no LLM output)
"""

import os, sys, json, argparse, re
import numpy as np
import faiss
from openai import OpenAI
from openai import BadRequestError
from utils.embedding import get_embedding  # uses EMBEDDING_BACKEND
from datetime import datetime   # ✅ added for saving

# ---------------------------
# Config
# ---------------------------
GEN_MODEL = os.getenv("OPENAI_GEN_MODEL", "gpt-4o-mini")

def _path(*parts):
    p1 = os.path.join("data", *parts)
    p2 = os.path.join("Data", *parts)
    return p1 if os.path.exists(p1) else p2

INDEX_PATH = _path("faiss.index")
DOCS_PATH  = _path("docs.jsonl")
JAPAN_MAP_PATH = _path("japan.json")

# ---------------------------
# Sanity: index & docs
# ---------------------------
missing = [p for p in (INDEX_PATH, DOCS_PATH) if not os.path.exists(p)]
if missing:
    sys.exit("❌ Missing files:\n  " + "\n  ".join(missing) +
             "\nTip: re-run `ingest_index_json.py` with your current EMBEDDING_MODEL.")

index = faiss.read_index(INDEX_PATH)
docs  = [json.loads(l) for l in open(DOCS_PATH, "r", encoding="utf-8")]

# Load Japanese name mapping
japanese_names = {}
if os.path.exists(JAPAN_MAP_PATH):
    with open(JAPAN_MAP_PATH, "r", encoding="utf-8") as f:
        japanese_names = json.load(f)
    print(f"Debug: Loaded {len(japanese_names)} Japanese name mappings")
    print(f"Debug: Sample mappings: {list(japanese_names.items())[:3]}")
else:
    print(f"Debug: Japanese mapping file not found at {JAPAN_MAP_PATH}")

# ---------------------------
# Helpers
# ---------------------------
def get_japanese_name(english_name: str) -> str:
    """
    Get Japanese name for English segment name, fallback to English if not found.
    """
    japanese_name = japanese_names.get(english_name, english_name)
    print(f"Debug: Mapping '{english_name}' → '{japanese_name}'")
    return japanese_name

def translate_keywords_to_japanese(keywords: list[str]) -> list[str]:
    """
    Translate a list of English keywords to Japanese using OpenAI.
    """
    if not keywords:
        return keywords
    
    client = _openai_client()
    keywords_text = ", ".join(keywords)
    
    sys_msg = (
        "Translate the following English keywords to Japanese. "
        "Return ONLY a JSON array of Japanese translations in the same order. "
        "Keep marketing and product terms natural for Japanese Amazon users."
    )
    
    messages = [{"role": "system", "content": sys_msg},
                {"role": "user", "content": keywords_text}]
    
    try:
        resp = client.chat.completions.create(
            model=GEN_MODEL,
            messages=messages,
            temperature=0.3,
            max_completion_tokens=200
        )
        content = resp.choices[0].message.content.strip()
        translated = json.loads(content)
        if isinstance(translated, list) and len(translated) == len(keywords):
            return translated
    except Exception as e:
        print(f"Translation error: {e}")
    
    # Fallback to original keywords if translation fails
    return keywords
def _normalize(v: np.ndarray) -> np.ndarray:
    v = v.astype("float32")
    n = np.linalg.norm(v) + 1e-12
    return v / n

def _percent_from_cos(cos_val: float) -> float:
    # cosine (-1..1) → 0..100%
    return max(0.0, min(1.0, (cos_val + 1.0) / 2.0)) * 100.0

def _tokenize_lower(s: str) -> set:
    return set(re.findall(r"[a-zA-Z0-9\-\+/#\.\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FAF]+", (s or "").lower()))

def estimate_ctr_percent(row: dict, base_ctr_pct: float = 1.0) -> float:
    """
    Heuristic CTR estimator (same as 123.py):
      est_ctr% = base_ctr_pct * score_factor * kw_bonus
      score_factor = 0.5 + 0.5 * (match_pct/100)
      kw_bonus = 1 + 0.02 * hit_count, capped at 1.25
    """
    score_factor = 0.5 + 0.5 * (row["match_pct"] / 100.0)
    hit_count = len(row.get("hits_brief", []))
    kw_bonus = min(1.25, 1.0 + 0.02 * hit_count)
    return round(base_ctr_pct * score_factor * kw_bonus, 2)

# ---------------------------
# Step 1: AI keyword extraction (optional)
# ---------------------------
def _openai_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set.")
    return OpenAI(api_key=api_key)

def extract_keywords_ai(brief: str, max_terms: int = 12) -> list[str]:
    """
    Extract compact keywords (English and Japanese) for better retrieval.
    Returns a list of strings. Falls back to naive split if parsing fails.
    """
    client = _openai_client()

    sys_msg = (
        "Extract up to 12 concise keywords (English and/or Japanese) from the campaign brief that are "
        "useful for matching Amazon audience/product segments. "
        "Return ONLY a JSON array of strings. Include both English and Japanese terms when relevant."
    )

    messages = [{"role":"system","content":sys_msg},
                {"role":"user","content":brief}]
    try:
        resp = client.chat.completions.create(
            model=GEN_MODEL,
            messages=messages,
            temperature=1,
            max_completion_tokens=180
        )
    except BadRequestError:
        resp = client.chat.completions.create(model=GEN_MODEL, messages=messages)

    content = resp.choices[0].message.content.strip()
    try:
        arr = json.loads(content)
        out, seen = [], set()
        for x in arr:
            if isinstance(x, str):
                t = x.strip()
                if t and t.lower() not in seen:
                    seen.add(t.lower()); out.append(t)
        return out[:max_terms]
    except Exception:
        toks = [t for t in re.findall(r"[a-zA-Z0-9\-\+/#\.\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FAF]+", brief.lower()) if len(t) > 1]
        out, seen = [], set()
        for t in toks:
            if t not in seen:
                seen.add(t); out.append(t)
        return out[:max_terms]

# ---------------------------
# Step 2: Retrieval
# ---------------------------
def translate_japanese_to_english(text: str) -> str:
    """Translate Japanese text to English for better embedding matching."""
    # Detect if text contains Japanese characters
    has_japanese = any('\u3040' <= char <= '\u309F' or '\u30A0' <= char <= '\u30FF' or '\u4E00' <= char <= '\u9FAF' for char in text)
    
    if not has_japanese:
        return text  # Already English, no translation needed
    
    client = _openai_client()
    
    sys_msg = (
        "Translate the following Japanese text to English. "
        "Keep the meaning accurate and preserve marketing/business terminology. "
        "Return ONLY the English translation."
    )
    
    messages = [{"role": "system", "content": sys_msg},
                {"role": "user", "content": text}]
    
    try:
        resp = client.chat.completions.create(
            model=GEN_MODEL,
            messages=messages,
            temperature=0.3,
            max_completion_tokens=500
        )
        english_text = resp.choices[0].message.content.strip()
        print(f"Debug: Translated Japanese to English: '{text[:50]}...' → '{english_text[:50]}...'")
        return english_text
    except Exception as e:
        print(f"Debug: Translation failed: {e}, using original text")
        return text

def retrieve_segments_detailed(
    brief: str,
    top_k: int = 3,
    use_extract: bool = True,
    kw_weight: float = 0.4
):
    print(f"Debug: Processing brief: '{brief[:100]}...'")
    
    # Translate Japanese input to English for better embedding matching
    english_brief = translate_japanese_to_english(brief)
    print(f"Debug: Using brief for embedding: '{english_brief[:100]}...'")
    
    emb_brief = _normalize(get_embedding(english_brief))
    print(f"Debug: Brief embedding shape: {emb_brief.shape}")
    
    ai_kws = extract_keywords_ai(english_brief) if use_extract else []
    print(f"Debug: Extracted keywords: {ai_kws}")

    if ai_kws:
        kw_text = ", ".join(ai_kws)
        emb_kw = _normalize(get_embedding(kw_text))
        q_vec = _normalize((1.0 - kw_weight) * emb_brief + kw_weight * emb_kw)
    else:
        q_vec = emb_brief

    brief_terms = _tokenize_lower(english_brief)  # Use English version for matching
    ai_terms = set([t.lower() for t in ai_kws])
    rows = []
    seen = set()
    
    # Start with a reasonable search size and expand if needed
    search_size = max(top_k * 5, 20)
    max_search_size = min(len(docs), 200)  # Don't search more than 200 or total docs
    
    while len(rows) < top_k and search_size <= max_search_size:
        # Debug: Check query vector normalization
        q_norm = np.linalg.norm(q_vec)
        print(f"Debug: Query vector norm: {q_norm}")
        
        D, I = index.search(np.array([q_vec], dtype="float32"), search_size)
        
        # Debug: Check returned distances
        print(f"Debug: First 5 distances: {D[0][:5]}")
        
        for rank, idx in enumerate(I[0]):
            if len(rows) >= top_k:
                break
                
            rec = docs[idx]
            key = rec.get("keyword") or f"seg_{idx}"
            if key in seen:
                continue
            seen.add(key)

            text = (rec.get("text") or rec.get("answer") or "")
            text_terms = _tokenize_lower(key + " " + text)
            hits_from_ai = sorted(list(ai_terms.intersection(text_terms)))
            hits_from_brief = sorted(list(brief_terms.intersection(text_terms)))

            cos = float(D[0][rank])
            
            # Debug: Print cosine scores to identify issues
            print(f"Debug: Segment '{key}' has cosine score: {cos}")
            
            # Clamp cosine score to valid range [0, 1] to handle numerical issues
            cos = max(0.0, min(1.0, cos))
            
            # Strict filtering: Only accept cosine scores in range [0.5, 1.0]
            # This excludes negative similarities and low-quality matches
            if 0.5 <= cos <= 1.0:
                print(f"Debug: ✅ Accepting segment '{key}' with score {cos}")
            else:
                print(f"Debug: ❌ Rejecting segment '{key}' with score {cos} (below 0.5 threshold)")
                continue
            
            if True:  # This replaces the if condition above
                pct = _percent_from_cos(cos)
                rows.append({
                    "keyword": key,
                    "text": text,
                    "cosine": cos,
                    "match_pct": pct,
                    "hits_ai": hits_from_ai,
                    "hits_brief": hits_from_brief,
                })
        
        # If we don't have enough results, expand search
        if len(rows) < top_k:
            print(f"Debug: Only found {len(rows)} segments, expanding search from {search_size} to {min(search_size * 2, max_search_size)}")
            search_size = min(search_size * 2, max_search_size)
        else:
            break

    print(f"Debug: Final result - found {len(rows)} segments")
    return rows, ai_kws

# ---------------------------
# Step 3: Prompt / Generation
# ---------------------------
STRICT_RULES = """
HARD RULES:
- Use ONLY segment names from Allowed Segment Names (verbatim).
- Do NOT invent or rephrase segment names.
- For **Keywords** and **Description**, use ONLY terms and facts found in each segment's Text.
- Keep outputs concise and ad-ready.
"""

INSTRUCTIONS = """
You are an Amazon Ads strategist.
Always respond entirely in Japanese.

Propose relevant target segments with:
- a brief 'Why it fits' (1–2 lines) in Japanese,
- 6–10 Keywords in Japanese taken FROM the segment Text,
- two Headlines in Japanese,
- a short Description in Japanese (≤150 chars), strictly grounded on the Text.
"""


def build_prompt_strict(campaign_brief: str, retrieved_rows: list[dict], allowed_names: list[str]) -> str:
    blocks = []
    for r in retrieved_rows:
        preview = r["text"][:320]
        japanese_name = get_japanese_name(r['keyword'])
        blocks.append(f"Keyword: {japanese_name}\nText: {preview}")
    context = "\n\n".join(blocks)
    allowed_block = ", ".join(allowed_names)

    return f"""{INSTRUCTIONS}
{STRICT_RULES}

=== Campaign Brief ===
{campaign_brief}

=== Retrieved Segments (from your data) ===
{context}

=== Allowed Segment Names (choose ONLY from this list) ===
{allowed_block}

Return ONLY the section below in clean markdown. Follow HARD RULES.

💡 Proposed Target Segments

**Segment 1: [Allowed Segment Name EXACT]**  
**Why it fits:** 1–2 lines.  
**Keywords:** 6–10 concise terms FROM its Text.  
**Headlines:**  
• “Headline 1”  
• “Headline 2”  
**Description:** ≤150 characters.

**Segment 2: [Allowed Segment Name EXACT]**  
**Why it fits:** 1–2 lines.  
**Keywords:** 6–10 concise terms FROM its Text.  
**Headlines:**  
• “Headline 1”  
• “Headline 2”  
**Description:** ≤150 characters.

**Segment 3: [Allowed Segment Name EXACT]**  
**Why it fits:** 1–2 lines.  
**Keywords:** 6–10 concise terms FROM its Text.  
**Headlines:**  
• “Headline 1”  
• “Headline 2”  
**Description:** ≤150 characters.
""".strip()

def generate_with_openai(prompt: str, model: str = GEN_MODEL) -> str:
    client = _openai_client()
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role":"user","content":prompt}],
            temperature=1,
            max_completion_tokens=600
        )
    except BadRequestError:
        resp = client.chat.completions.create(model=model, messages=[{"role":"user","content":prompt}])
    return resp.choices[0].message.content.strip()

# ---------------------------
# Pretty printing
# ---------------------------
def print_matches(rows, ai_kws, base_ctr_pct: float = 1.0, brief: str = ""):
    print("\n🔎 Matched segments (from YOUR data):\n")
    
    for i, r in enumerate(rows, 1):
        est_ctr = estimate_ctr_percent(r, base_ctr_pct=base_ctr_pct)
        
        # Always show Japanese segment names
        segment_name = get_japanese_name(r['keyword'])
            
        print(f"{i}) {segment_name}")
        print(f"   • score: {r['cosine']:.3f}  |  match: {r['match_pct']:.1f}%  |  est CTR: {est_ctr:.2f}%")
        if r["hits_ai"]:
            print(f"   • hits (AI terms): {', '.join(r['hits_ai'])}")
        if r["hits_brief"]:
            print(f"   • hits (brief terms): {', '.join(r['hits_brief'])}")

# ---------------------------
# ✅ Save generation output
# ---------------------------
def save_generation(brief, ai_kws, rows, md_output, path="generated_segments.jsonl"):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    data = {
        "timestamp": datetime.utcnow().isoformat(),
        "brief": brief,
        "ai_keywords": ai_kws,
        "retrieved_segments": [r["keyword"] for r in rows],
        "scores": [
            {"keyword": r["keyword"], "match_pct": r["match_pct"], "cosine": r["cosine"]}
            for r in rows
        ],
        "output_markdown": md_output
    }
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")
    # print(f"\n💾 Saved generation → {path}")  # Commented out to avoid appearing in web output

# ---------------------------
# CLI
# ---------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top-k", type=int, default=3, help="How many segments to retrieve.")
    ap.add_argument("--brief", type=str, default=None, help="Inline campaign brief (skip interactive prompt).")
    ap.add_argument("--no-extract", action="store_true", help="Disable AI keyword extraction step.")
    ap.add_argument("--kw-weight", type=float, default=0.5, help="Blend weight for keyword embedding (0..1).")
    ap.add_argument("--retrieval-only", action="store_true", help="Only print matches and exit.")
    ap.add_argument("--base-ctr", type=float, default=1.0, help="Base CTR %% prior (default 1.0).")
    args = ap.parse_args()

    brief = args.brief or input("Enter campaign brief: ").strip()

    rows, ai_kws = retrieve_segments_detailed(
        brief,
        top_k=args.top_k,
        use_extract=not args.no_extract,
        kw_weight=max(0.0, min(1.0, args.kw_weight)),
    )

    print_matches(rows, ai_kws, base_ctr_pct=args.base_ctr, brief=brief)

    if args.retrieval_only:
        return

    # Use Japanese names from japan.json for AI generation
    allowed = [get_japanese_name(r["keyword"]) for r in rows]
    prompt = build_prompt_strict(brief, rows, allowed)
    md = generate_with_openai(prompt, model=GEN_MODEL)

    print("\n" + md + "\n")

    # ✅ Added call to save the output
    save_generation(brief, ai_kws, rows, md_output=md)

if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
