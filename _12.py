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

    # Dynamic domain detection for targeted messaging
    domain_keywords = {
        'beauty': ['美容', 'スキンケア', 'コスメ', 'メイク', '化粧品', 'ビューティー'],
        'business': ['ビジネス', 'オフィス', '仕事', '会社', '企業', 'プロフェッショナル'],
        'electronics': ['電子機器', 'ガジェット', 'スマホ', 'パソコン', 'テクノロジー', 'デジタル'],
        'sports': ['スポーツ', 'フィットネス', '運動', 'トレーニング', 'アウトドア', 'キャンプ'],
        'home': ['ホーム', '家庭', '生活', 'インテリア', 'キッチン', '掃除'],
        'fashion': ['ファッション', '服', 'アパレル', 'スタイル', 'おしゃれ', 'トレンド']
    }
    
    detected_domain = 'general'
    brief_lower = brief.lower()
    for domain, keywords in domain_keywords.items():
        if any(keyword in brief for keyword in keywords):
            detected_domain = domain
            break
    
    # Domain-specific targeting and tone
    domain_config = {
        'beauty': {
            'age_group': '20〜40代の美容意識の高い女性',
            'tone': '美しさを追求し、自分らしさを大切にする',
            'context': 'スキンケアやメイクアップ、ヘアケアに関心が高く'
        },
        'business': {
            'age_group': '30〜50代のビジネスパーソン',
            'tone': '効率性と品質を重視し、プロフェッショナルな印象を求める',
            'context': '仕事の生産性向上や職場での印象アップに関心があり'
        },
        'electronics': {
            'age_group': '25〜45代のテクノロジー愛好者',
            'tone': '最新技術と機能性を重視し、便利さを追求する',
            'context': 'デジタルライフの充実や作業効率化に関心が高く'
        },
        'sports': {
            'age_group': '30〜50代のアクティブな層',
            'tone': '健康的なライフスタイルを重視し、アクティブに過ごしたい',
            'context': 'フィットネスやアウトドア活動、健康管理に関心があり'
        },
        'home': {
            'age_group': '30〜50代の家庭を持つ層',
            'tone': '快適で機能的な生活空間を求め、家族の幸せを大切にする',
            'context': '住環境の改善や家事の効率化に関心が高く'
        },
        'fashion': {
            'age_group': '20〜40代のファッション意識の高い層',
            'tone': '個性的でスタイリッシュな装いを求め、トレンドに敏感',
            'context': 'ファッションやライフスタイルの向上に関心があり'
        },
        'general': {
            'age_group': '30〜50代の幅広い層',
            'tone': '品質と価値を重視し、生活の質向上を求める',
            'context': '日常生活の充実や趣味の拡充に関心があり'
        }
    }
    
    config = domain_config[detected_domain]
    
    prompt = f"""
あなたは日本のAmazonマーケティング戦略担当者です。
以下のキャンペーン概要と関連商品情報をもとに、4〜5個のターゲットオーディエンスセグメントを提案してください。

**出力形式（必ずこの構造のJSON配列で出力）:**
[
  {{
    "name": "セグメント名",
    "reason": "なぜこのセグメントがキャンペーンに適しているか（1〜2文）",
    "keywords": ["キーワード1", "キーワード2", "キーワード3"],
    "headlines": ["広告見出し1", "広告見出し2"],
    "description": "ターゲット層や商品特徴を踏まえた詳細説明（2〜3文）"
  }}
]

**ターゲット設定:**
- 主要対象: {config['age_group']}
- 特徴: {config['tone']}
- 背景: {config['context']}

**品質要件:**
- Amazon Japanの自然で親しみやすいトーンで記述
- 各フィールドを必ず埋める（空欄・「説明がありません」等は禁止）
- 「最適」「豊富に揃う」「充実した」等の陳腐な表現を避ける
- 具体的で魅力的な表現を使用
- セグメント名は階層表記（例：「ビューティー > スキンケア」）も可
- 広告見出しは20文字以内で訴求力のあるものにする

**キャンペーン概要:**
{brief}

**関連商品・カテゴリ情報:**
{docs_text}

上記の情報をもとに、JSON配列のみを出力してください："""

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "あなたは日本のAmazonマーケティング専門家です。必ず有効なJSON形式で応答してください。マークダウンや説明文は含めないでください。"},
                {"role": "user", "content": prompt},
            ],
            temperature=0.4,
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
            
            # Enhanced validation and quality assurance
            for i, segment in enumerate(parsed):
                # Ensure all required fields exist with quality content
                if "name" not in segment or not segment["name"].strip():
                    segment["name"] = f"セグメント{i+1}"
                
                if "reason" not in segment or not segment["reason"].strip():
                    if segment.get("description"):
                        # Extract first sentence from description as reason
                        desc_sentences = segment["description"].split("。")
                        segment["reason"] = desc_sentences[0].strip() + "。" if desc_sentences[0].strip() else "このセグメントは対象キャンペーンに適しています。"
                    else:
                        segment["reason"] = "このセグメントは対象キャンペーンに適しています。"
                
                if "keywords" not in segment or not isinstance(segment["keywords"], list) or len(segment["keywords"]) == 0:
                    # Generate basic keywords from brief and segment name
                    brief_words = brief.replace('、', ' ').replace('。', ' ').split()
                    name_words = segment["name"].replace('>', ' ').replace('＆', ' ').split()
                    segment["keywords"] = (brief_words[:2] + name_words[:2])[:3] if brief_words or name_words else ["商品", "サービス", "おすすめ"]
                
                if "headlines" not in segment or not isinstance(segment["headlines"], list) or len(segment["headlines"]) == 0:
                    # Generate compelling headlines based on segment name and domain
                    name_clean = segment["name"].split('>')[-1].strip() if '>' in segment["name"] else segment["name"]
                    segment["headlines"] = [
                        f"{name_clean}で新しい体験を",
                        f"あなたにぴったりの{name_clean}"
                    ]
                
                if "description" not in segment or not segment["description"].strip():
                    # Generate description from reason and context
                    segment["description"] = f"{segment['reason']} {config['context']}、このセグメントの商品やサービスがお客様のニーズにお応えします。"
                
                # Quality checks - remove generic phrases and improve content
                for field in ["reason", "description"]:
                    if field in segment:
                        content = segment[field]
                        # Replace generic phrases with more specific ones
                        replacements = {
                            "最適": "ぴったり",
                            "豊富に揃う": "多彩な選択肢",
                            "充実した": "幅広い",
                            "説明がありません": "魅力的な商品をご提案",
                            "No description": "お客様のニーズにお応えする商品"
                        }
                        for old, new in replacements.items():
                            content = content.replace(old, new)
                        segment[field] = content
                
                # Ensure headlines are within character limit
                if "headlines" in segment:
                    segment["headlines"] = [h[:20] + "..." if len(h) > 20 else h for h in segment["headlines"][:2]]
                    
        except Exception as e:
            print(f"⚠️ Could not parse JSON: {e}")
            print(f"Raw response: {text[:200]}...")
            # Enhanced structured fallback with domain-specific content
            fallback_segments = []
            for i, doc in enumerate(retrieved_docs[:3]):  # Use top 3 retrieved docs
                segment_name = doc['keyword'] if 'keyword' in doc else f"セグメント{i+1}"
                fallback_segments.append({
                    "name": segment_name,
                    "reason": f"{segment_name}は{config['age_group']}に人気の商品カテゴリです。",
                    "keywords": brief.split()[:3] if brief.split() else ["商品", "サービス", "おすすめ"],
                    "headlines": [f"{segment_name}特集", f"人気の{segment_name}"],
                    "description": f"{config['context']}、{segment_name}の商品がお客様のライフスタイルを豊かにします。高品質で使いやすい商品を多数ご用意しております。"
                })
            
            parsed = fallback_segments if fallback_segments else [{
                "name": "おすすめ商品",
                "reason": f"{config['age_group']}におすすめの商品をご提案します。",
                "keywords": brief.split()[:3] if brief.split() else ["商品", "サービス", "おすすめ"],
                "headlines": ["おすすめ商品特集", "あなたにぴったり"],
                "description": f"{config['context']}、厳選された商品をお客様にお届けします。品質と価値を重視した商品選びをサポートいたします。"
            }]

        # Cache successful results for faster subsequent requests
        if rdb and parsed and not any('error' in seg for seg in parsed):
            # Extended cache time for high-quality results
            cache_duration = 1800  # 30 minutes
            rdb.setex(cache_key, cache_duration, json.dumps(parsed, ensure_ascii=False))
            print(f"💾 Cached result for {cache_duration}s")

        print(f"✅ Generation done in {round(time.time()-start,2)}s ({model}) - Domain: {detected_domain}")
        return parsed

    except Exception as e:
        print("❌ Generation error:", e)
        # Return domain-aware error fallback
        error_fallback = [{
            "name": "サービス一時停止中",
            "reason": "申し訳ございませんが、一時的にサービスが利用できません。",
            "keywords": ["サービス", "メンテナンス", "お知らせ"],
            "headlines": ["サービス復旧中", "しばらくお待ちください"],
            "description": "現在システムメンテナンス中のため、一時的にご利用いただけません。復旧まで今しばらくお待ちください。"
        }]
        return error_fallback



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
