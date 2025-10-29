#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, numpy as np
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def get_embeddings_batch(texts: list[str]) -> list[np.ndarray]:
    """Return embeddings for a batch of texts."""
    resp = client.embeddings.create(
        model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-large"),
        input=texts
    )
    return [np.array(e.embedding, dtype="float32") for e in resp.data]
