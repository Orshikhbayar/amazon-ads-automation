#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
12.py — Unified Retrieval + Generation Engine

Features:
- Hybrid FAISS + BM25 retrieval
- Redis caching for embeddings & results
- Compatible with Flask current_app context
"""

import os, sys, json, numpy as np, time
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

# Domain-specific keyword heuristics for boosting relevance
DOMAIN_KEYWORDS = {
    'skincare': ['スキンケア', '美容液', '化粧水', '乳液', 'クリーム', '洗顔', 'パック', 'マスク', '保湿', 'アンチエイジング'],
    'outdoor': ['アウトドア', 'キャンプ', 'ハイキング', '登山', 'テント', 'バックパック', 'シュラフ', 'ランタン', 'コンロ', 'アウトドアウェア'],
    'electronics': ['スマートフォン', 'パソコン', 'タブレット', 'イヤホン', 'スピーカー', 'カメラ', 'ゲーム', 'テレビ', 'プリンター', 'ガジェット'],
    'fitness': ['フィットネス', 'トレーニング', 'ダンベル', 'ヨガ', 'ランニング', 'プロテイン', 'サプリメント', 'ウェア', 'シューズ', 'マット'],
    'home': ['ホーム', 'インテリア', 'キッチン', '掃除', '収納', '家具', '照明', 'カーテン', 'ラグ', '寝具'],
    'fashion': ['ファッション', '服', 'シャツ', 'パンツ', 'ドレス', 'バッグ', '靴', 'アクセサリー', '時計', 'ジュエリー']
}

def normalize(v):
    """Normalize vector to 0-1 range with numerical stability."""
    v = np.array(v)
    min_v, max_v = np.min(v), np.max(v)
    if max_v - min_v < 1e-8:
        return np.ones_like(v) * 0.5  # Return middle value if all same
    return (v - min_v) / (max_v - min_v + 1e-8)


# ---------------------------
# EMBEDDING UTIL
# ---------------------------
def embed_text(text, cache_key=None, rdb=None):
    """Enhanced embedding with intelligent caching and compression."""
    if rdb and cache_key and rdb.exists(cache_key):
        try:
            cached_data = json.loads(rdb.get(cache_key))
            return np.array(cached_data, dtype="float32")
        except Exception as e:
            print(f"⚠️ Cache read error for {cache_key}: {e}")
            # Continue to generate fresh embedding

    vec = get_embeddings_batch([text])[0].astype("float32")

    if rdb and cache_key:
        try:
            # Cache with longer TTL for embeddings (they rarely change)
            cache_ttl = 7200  # 2 hours
            rdb.setex(cache_key, cache_ttl, json.dumps(vec.tolist()))
        except Exception as e:
            print(f"⚠️ Cache write error for {cache_key}: {e}")
    
    return vec


def get_system_diagnostics(rdb=None):
    """Get system performance diagnostics and cache statistics."""
    diagnostics = {
        "timestamp": time.time(),
        "cache_stats": {},
        "memory_usage": {},
        "performance_metrics": {}
    }
    
    if rdb:
        try:
            # Redis cache statistics
            info = rdb.info()
            diagnostics["cache_stats"] = {
                "redis_connected": True,
                "used_memory": info.get("used_memory_human", "N/A"),
                "total_connections": info.get("total_connections_received", 0),
                "keyspace_hits": info.get("keyspace_hits", 0),
                "keyspace_misses": info.get("keyspace_misses", 0),
                "hit_rate": round(info.get("keyspace_hits", 0) / max(info.get("keyspace_hits", 0) + info.get("keyspace_misses", 0), 1) * 100, 2)
            }
            
            # Count cache keys by type
            keys_pattern = {
                "embeddings": "embed*",
                "retrievals": "retrieval*", 
                "generations": "generate*"
            }
            
            for key_type, pattern in keys_pattern.items():
                try:
                    count = len(rdb.keys(pattern))
                    diagnostics["cache_stats"][f"{key_type}_cached"] = count
                except:
                    diagnostics["cache_stats"][f"{key_type}_cached"] = 0
                    
        except Exception as e:
            diagnostics["cache_stats"] = {"redis_connected": False, "error": str(e)}
    
    # Memory usage (if psutil available)
    try:
        import psutil
        process = psutil.Process()
        diagnostics["memory_usage"] = {
            "rss_mb": round(process.memory_info().rss / 1024 / 1024, 2),
            "vms_mb": round(process.memory_info().vms / 1024 / 1024, 2),
            "cpu_percent": process.cpu_percent()
        }
    except ImportError:
        diagnostics["memory_usage"] = {"error": "psutil not available"}
    
    return diagnostics


def validate_retrieval_results(results, brief, min_score=10.0):
    """Validate retrieval results and provide quality metrics."""
    if not results:
        return {"valid": False, "reason": "No results returned", "quality_score": 0}
    
    validation = {
        "valid": True,
        "total_results": len(results),
        "quality_metrics": {},
        "issues": []
    }
    
    # Score distribution analysis
    scores = [r.get("score", 0) for r in results]
    validation["quality_metrics"] = {
        "avg_score": round(np.mean(scores), 2),
        "min_score": round(np.min(scores), 2),
        "max_score": round(np.max(scores), 2),
        "score_std": round(np.std(scores), 2)
    }
    
    # Quality checks
    if validation["quality_metrics"]["min_score"] < min_score:
        validation["issues"].append(f"Low minimum score: {validation['quality_metrics']['min_score']}")
    
    if validation["quality_metrics"]["score_std"] < 5.0:
        validation["issues"].append("Low score variance - results may be too similar")
    
    # Content diversity check
    keywords = [r.get("keyword", "") for r in results]
    unique_keywords = len(set(keywords))
    if unique_keywords < len(keywords):
        validation["issues"].append("Duplicate keywords in results")
    
    # Brief relevance check (simple keyword overlap)
    brief_words = set(brief.lower().split())
    keyword_overlap = []
    for result in results:
        keyword_words = set(result.get("keyword", "").lower().split())
        overlap = len(brief_words & keyword_words) / max(len(brief_words), 1)
        keyword_overlap.append(overlap)
    
    validation["quality_metrics"]["avg_relevance"] = round(np.mean(keyword_overlap), 3)
    
    if validation["quality_metrics"]["avg_relevance"] < 0.1:
        validation["issues"].append("Low keyword relevance to brief")
    
    validation["quality_score"] = min(100, max(0, 
        validation["quality_metrics"]["avg_score"] * 0.4 +
        validation["quality_metrics"]["avg_relevance"] * 60 +
        (unique_keywords / len(keywords)) * 20
    ))
    
    return validation


# ---------------------------
# RETRIEVAL CORE
# ---------------------------
def retrieve_docs(brief, top_k=TOP_K_DEFAULT, kw_weight=KW_WEIGHT_DEFAULT, index=None, docs=None, rdb=None):
    """
    Enhanced hybrid retrieval with improved scoring, caching, and diagnostics.
    Uses 0.7 * FAISS + 0.3 * BM25 hybrid scoring for better relevance.
    """
    import time
    start_time = time.time()
    
    # Check cache first for faster responses
    cache_key = f"retrieval_v2:{brief}:{top_k}:{kw_weight}"
    if rdb and rdb.exists(cache_key):
        cached_result = json.loads(rdb.get(cache_key))
        print(f"⚡ Cache hit for retrieval ({len(cached_result)} docs)")
        return cached_result

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

    # Enhanced embedding with caching
    embed_cache_key = f"embed_v2:{brief}"
    q_emb = embed_text(brief, embed_cache_key, rdb)
    q_emb = q_emb / (np.linalg.norm(q_emb) + 1e-12)

    # FAISS retrieval with expanded search for better hybrid scoring
    search_k = min(top_k * 3, len(docs))  # Search more candidates for better hybrid results
    sims, idxs = index.search(np.array([q_emb], dtype="float32"), search_k)
    sims, idxs = sims[0], idxs[0]

    # Enhanced FAISS score normalization with domain boost
    def detect_domain_and_boost(brief, results):
        """Detect domain and apply keyword-based boosting."""
        brief_lower = brief.lower()
        detected_domains = []
        
        # Detect relevant domains
        for domain, keywords in DOMAIN_KEYWORDS.items():
            if any(kw in brief_lower for kw in keywords):
                detected_domains.append(domain)
        
        if not detected_domains:
            return results, "general"
        
        # Apply domain-specific boosting
        for result in results:
            keyword_lower = result["keyword"].lower()
            text_lower = result["text"].lower()
            boost_factor = 1.0
            
            for domain in detected_domains:
                domain_keywords = DOMAIN_KEYWORDS[domain]
                # Count keyword matches in both keyword and text
                matches = sum(1 for kw in domain_keywords if kw in keyword_lower or kw in text_lower)
                if matches > 0:
                    boost_factor += 0.1 * matches  # 10% boost per matching keyword
            
            result["domain_boost"] = boost_factor
        
        return results, detected_domains[0] if len(detected_domains) == 1 else "multi-domain"

    # Normalize FAISS cosine similarity scores to 0-1 range
    faiss_scores_norm = normalize((sims + 1) / 2)  # Shift [-1,1] to [0,1] then normalize
    faiss_results = {}
    for j, i in enumerate(idxs):
        if i != -1 and i < len(docs):
            keyword = docs[i]["keyword"]
            faiss_results[keyword] = {
                "keyword": keyword,
                "text": docs[i]["text"],
                "faiss_score": float(faiss_scores_norm[j]),
                "faiss_rank": j + 1
            }

    # BM25 retrieval with improved scoring
    bm25_results = {}
    if 'bm25' in locals() and bm25:
        try:
            # Tokenize query for BM25
            query_tokens = brief.lower().replace('、', ' ').replace('。', ' ').split()
            bm_scores = bm25.get_scores(query_tokens)
            
            # Normalize BM25 scores using the standard normalize function
            bm_scores_norm = normalize(np.maximum(bm_scores, 0))  # Ensure non-negative before normalize
            bm_ranked = np.argsort(bm_scores_norm)[::-1][:search_k]
            
            for rank, i in enumerate(bm_ranked):
                if i < len(docs) and bm_scores_norm[i] > 0:
                    keyword = docs[i]["keyword"]
                    bm25_results[keyword] = {
                        "keyword": keyword,
                        "text": docs[i]["text"],
                        "bm25_score": float(bm_scores_norm[i]),
                        "bm25_rank": rank + 1
                    }
        except Exception as e:
            print(f"⚠️ BM25 scoring failed: {e}")
            bm25_results = {}

    # Hybrid scoring: 0.7 * FAISS + 0.3 * BM25 with domain boosting
    FAISS_WEIGHT = 0.7
    BM25_WEIGHT = 0.3
    
    combined_results = {}
    all_keywords = set(faiss_results.keys()) | set(bm25_results.keys())
    
    for keyword in all_keywords:
        faiss_score = faiss_results.get(keyword, {}).get("faiss_score", 0)
        bm25_score = bm25_results.get(keyword, {}).get("bm25_score", 0)
        
        # Base hybrid score calculation
        hybrid_score = FAISS_WEIGHT * faiss_score + BM25_WEIGHT * bm25_score
        
        # Get text from either result
        text = (faiss_results.get(keyword, {}).get("text") or 
                bm25_results.get(keyword, {}).get("text", ""))
        
        combined_results[keyword] = {
            "keyword": keyword,
            "text": text,
            "hybrid_score": hybrid_score,
            "faiss_score": faiss_score,
            "bm25_score": bm25_score,
            "faiss_rank": faiss_results.get(keyword, {}).get("faiss_rank", 999),
            "bm25_rank": bm25_results.get(keyword, {}).get("bm25_rank", 999)
        }
    
    # Apply domain-specific boosting
    combined_list = list(combined_results.values())
    boosted_results, detected_domain = detect_domain_and_boost(brief, combined_list)
    
    # Update hybrid scores with domain boost
    for result in boosted_results:
        boost = result.get("domain_boost", 1.0)
        result["hybrid_score"] *= boost
        result["boosted"] = boost > 1.0

    # Sort by boosted hybrid score and select top_k
    ranked_results = sorted(boosted_results, 
                          key=lambda x: x["hybrid_score"], 
                          reverse=True)[:top_k]
    
    # Format final results with enhanced metadata
    retrieved = []
    for i, result in enumerate(ranked_results):
        retrieved.append({
            "keyword": result["keyword"],
            "text": result["text"],
            "score": round(result["hybrid_score"] * 100, 2),  # Convert to percentage
            "rank": i + 1,
            "faiss_score": round(result["faiss_score"] * 100, 2),
            "bm25_score": round(result["bm25_score"] * 100, 2),
            "source": "hybrid"
        })
    
    # Enhanced caching with longer TTL for better performance
    if rdb and retrieved:
        cache_ttl = 3600  # 1 hour cache for retrieval results
        rdb.setex(cache_key, cache_ttl, json.dumps(retrieved, ensure_ascii=False))
        print(f"💾 Cached retrieval result for {cache_ttl}s")

    # Validate retrieval quality
    validation = validate_retrieval_results(retrieved, brief)
    retrieval_time = round(time.time() - start_time, 3)
    
    # Enhanced logging with quality metrics
    quality_info = f"Quality: {validation['quality_score']:.1f}/100"
    if validation['issues']:
        quality_info += f" (Issues: {len(validation['issues'])})"
    
    print(f"🔍 Retrieval completed in {retrieval_time}s - Found {len(retrieved)} docs (FAISS: {len(faiss_results)}, BM25: {len(bm25_results)}) - Domain: {detected_domain} - {quality_info}")
    
    # Add validation metadata to results
    for result in retrieved:
        result["validation"] = {
            "quality_score": validation["quality_score"],
            "avg_score": validation["quality_metrics"]["avg_score"],
            "retrieval_time": retrieval_time
        }
    
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

    # Enhanced cache check with immediate return
    cache_key = f"generate:{brief}"
    if rdb and (cached := rdb.get(cache_key)):
        print("⚡ Returning cached generation result")
        return json.loads(cached)

    start = time.time()
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    model = os.getenv("MODEL_GENERATION", os.getenv("OPENAI_GEN_MODEL", "gpt-4o-mini"))

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
            top_p=0.95,
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
            
            # Strict validation with automatic re-prompt if needed
            def validate_and_fix_segments(segments):
                """Strict validation with automatic re-prompt for incomplete segments."""
                required_fields = ["name", "reason", "keywords", "headlines", "description"]
                needs_reprompt = False
                
                for i, segment in enumerate(segments):
                    # Check for required fields and empty content
                    for field in required_fields:
                        if field not in segment or not segment[field] or segment[field] == "（未入力）":
                            segment[field] = "（未入力）"
                            needs_reprompt = True
                    
                    # Check for placeholder text that indicates incomplete generation
                    problematic_phrases = ["説明がありません", "No description", "未入力", "（未入力）"]
                    for field in ["reason", "description"]:
                        if field in segment and any(phrase in str(segment[field]) for phrase in problematic_phrases):
                            needs_reprompt = True
                
                return segments, needs_reprompt
            
            # Enhanced validation and quality assurance
            parsed, needs_reprompt = validate_and_fix_segments(parsed)
            
            # Automatic re-prompt if validation fails
            if needs_reprompt:
                print("🔄 Incomplete generation detected, re-prompting...")
                reprompt = f"""上記のJSONを完全に埋め直してください。空欄を残さず、日本語で自然に書いてください。

以下の形式で必ず全てのフィールドを埋めてください：
[
  {{
    "name": "セグメント名（必須）",
    "reason": "適している理由を1-2文で（必須）",
    "keywords": ["キーワード1", "キーワード2", "キーワード3"]（必須）,
    "headlines": ["見出し1", "見出し2"]（必須）,
    "description": "詳細説明を2-3文で（必須）"
  }}
]

キャンペーン概要: {brief}
参照情報: {docs_text[:1000]}

JSON形式のみで出力してください："""

                try:
                    retry_resp = client.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": "日本語で完全なJSONを生成してください。空欄は絶対に残さないでください。"},
                            {"role": "user", "content": reprompt}
                        ],
                        temperature=0.4,
                        top_p=0.95,
                        max_completion_tokens=800,
                    )
                    
                    retry_text = retry_resp.choices[0].message.content.strip()
                    if retry_text.startswith("```"):
                        lines = retry_text.split("\n")
                        retry_text = "\n".join(lines[1:-1]) if len(lines) > 2 else retry_text
                        retry_text = retry_text.strip()
                    
                    retry_parsed = json.loads(retry_text)
                    if isinstance(retry_parsed, list) and len(retry_parsed) > 0:
                        parsed = retry_parsed
                        print("✅ Re-prompt successful")
                    
                except Exception as retry_e:
                    print(f"⚠️ Re-prompt failed: {retry_e}")
            
            # Final validation and cleanup
            for i, segment in enumerate(parsed):
                # Ensure all required fields exist with quality content
                if "name" not in segment or not segment["name"].strip():
                    segment["name"] = f"セグメント{i+1}"
                
                if "reason" not in segment or not segment["reason"].strip():
                    segment["reason"] = "このセグメントは対象キャンペーンに適しています。"
                
                if "keywords" not in segment or not isinstance(segment["keywords"], list) or len(segment["keywords"]) == 0:
                    brief_words = brief.replace('、', ' ').replace('。', ' ').split()
                    name_words = segment["name"].replace('>', ' ').replace('＆', ' ').split()
                    segment["keywords"] = (brief_words[:2] + name_words[:2])[:3] if brief_words or name_words else ["商品", "サービス", "おすすめ"]
                
                if "headlines" not in segment or not isinstance(segment["headlines"], list) or len(segment["headlines"]) == 0:
                    name_clean = segment["name"].split('>')[-1].strip() if '>' in segment["name"] else segment["name"]
                    segment["headlines"] = [f"{name_clean}特集", f"人気の{name_clean}"]
                
                if "description" not in segment or not segment["description"].strip():
                    segment["description"] = f"{segment['reason']} {config['context']}、このセグメントの商品やサービスがお客様のニーズにお応えします。"
                
                # Remove problematic placeholder text
                replacements = {
                    "最適": "ぴったり",
                    "豊富に揃う": "多彩な選択肢", 
                    "充実した": "幅広い",
                    "説明がありません": "魅力的な商品をご提案",
                    "No description": "お客様のニーズにお応えする商品",
                    "（未入力）": "お客様に最適な商品"
                }
                
                for field in ["reason", "description"]:
                    if field in segment:
                        content = str(segment[field])
                        for old, new in replacements.items():
                            content = content.replace(old, new)
                        segment[field] = content
                
                # Ensure headlines are within character limit
                if "headlines" in segment:
                    segment["headlines"] = [h[:20] for h in segment["headlines"][:2]]
                    
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
