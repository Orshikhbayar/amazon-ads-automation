#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
utils/embedding.py
English-only local embeddings via FastEmbed (no torch), or OpenAI if requested.
Choose backend with EMBEDDING_BACKEND=local|openai (default: local).

Recommended English model (fast & accurate):
  EMBEDDING_MODEL="BAAI/bge-small-en-v1.5"
"""

from __future__ import annotations
import os, time
from typing import Iterable, List
import numpy as np

# -------------------------------
# Config
# -------------------------------
BACKEND = os.getenv("EMBEDDING_BACKEND", "local").lower()  # local | openai
LOCAL_MODEL = os.getenv("EMBEDDING_MODEL") or "BAAI/bge-small-en-v1.5"
OPENAI_MODEL = os.getenv("EMBEDDING_MODEL") or "text-embedding-3-small"

# -------------------------------
# Lazy loaders
# -------------------------------
_fastembed_model = None
def _load_fastembed(model_name: str = LOCAL_MODEL):
    global _fastembed_model
    if _fastembed_model is None:
        from fastembed import TextEmbedding
        try:
            _fastembed_model = TextEmbedding(model_name=model_name)
        except ValueError:
            # Show supported models for quick debug
            from fastembed import TextEmbedding as _TE
            available = [m.get("model", m) for m in _TE.list_supported_models()]
            raise RuntimeError(
                f"FastEmbed model '{model_name}' not supported.\n"
                f"Try one of: {available}"
            )
    return _fastembed_model

_openai_client = None
def _load_openai():
    global _openai_client
    if _openai_client is None:
        from openai import OpenAI
        key = os.getenv("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("OPENAI_API_KEY is not set (required for EMBEDDING_BACKEND=openai).")
        _openai_client = OpenAI(api_key=key)
    return _openai_client

# -------------------------------
# Public API
# -------------------------------
def get_embedding(text: str) -> np.ndarray:
    return get_embeddings_batch([text])[0]

def get_embeddings_batch(texts: Iterable[str], batch_size: int = 128) -> np.ndarray:
    texts = [t if isinstance(t, str) else str(t) for t in texts]

    if BACKEND == "openai":
        client = _load_openai()
        out: List[np.ndarray] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            for attempt in range(3):
                try:
                    r = client.embeddings.create(model=OPENAI_MODEL, input=batch)
                    out.extend([np.array(d.embedding, dtype="float32") for d in r.data])
                    break
                except Exception:
                    time.sleep(1.0 * (attempt+1))
                    if attempt == 2:
                        raise
        return np.vstack(out).astype("float32")

    # local (FastEmbed)
    emb = _load_fastembed()
    vecs: List[np.ndarray] = []
    for batch_vecs in emb.embed(texts, batch_size=batch_size):
        arr = np.array(batch_vecs, dtype="float32")
        vecs.append(arr)
    return np.vstack(vecs).astype("float32")
