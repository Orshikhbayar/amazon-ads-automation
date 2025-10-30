#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, numpy as np
from openai import OpenAI

_client = None

def _get_client():
    """Lazy-load OpenAI client to ensure env vars are loaded."""
    global _client
    if _client is None:
        _client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return _client

def get_embeddings_batch(texts: list[str]) -> list[np.ndarray]:
    """Return embeddings for a batch of texts."""
    client = _get_client()
    resp = client.embeddings.create(
        model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-large"),
        input=texts
    )
    return [np.array(e.embedding, dtype="float32") for e in resp.data]
