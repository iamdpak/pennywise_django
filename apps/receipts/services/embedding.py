from __future__ import annotations

from typing import Iterable, List, Tuple

import logging
import threading

import faiss
import numpy as np
import os
import requests
from django.conf import settings
from openai import OpenAI


logger = logging.getLogger(__name__)


LLM_CONFIG = settings.LLM_CONFIG
LLM_PROVIDER = LLM_CONFIG.get("provider", "ollama").lower()
OLLAMA_CFG = LLM_CONFIG.get("ollama", {})
OPENAI_CFG = LLM_CONFIG.get("openai", {})
LLM_TIMEOUT = int(LLM_CONFIG.get("timeout", 60))

# Choose embedding model: prefer explicit EMBEDDING_MODEL env, fallback to provider model.
# Select embedding model: prefer explicit env; else provider-specific embedding default.
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
if not EMBEDDING_MODEL:
    if LLM_PROVIDER == "openai":
        # Do not fall back to a chat/completions model; use an embedding model.
        EMBEDDING_MODEL = OPENAI_CFG.get("embedding_model") or "text-embedding-3-small"
    else:
        EMBEDDING_MODEL = OLLAMA_CFG.get("model") or "llama3.2-vision"

_openai_client = None
if LLM_PROVIDER == "openai":
    api_key = OPENAI_CFG.get("api_key")
    if api_key:
        _openai_client = OpenAI(api_key=api_key)


class EmbeddingIndex:
    _lock = threading.Lock()
    _index: faiss.IndexFlatL2 | None = None
    _metadata: List[Tuple[int, str]] = []

    def upsert_receipt(self, receipt_id: int, texts: Iterable[str]) -> None:
        cleaned = [text.strip() for text in texts if text and text.strip()]
        if not cleaned:
            logger.debug("EmbeddingIndex: no text to index for receipt %s", receipt_id)
            return

        embeddings = self._embed(cleaned)
        if embeddings.size == 0:
            logger.warning("EmbeddingIndex: embedding model returned no vectors")
            return

        cls = type(self)

        with cls._lock:
            if cls._index is None:
                cls._index = faiss.IndexFlatL2(embeddings.shape[1])
            elif embeddings.shape[1] != cls._index.d:
                logger.error(
                    "EmbeddingIndex dimension mismatch (existing=%s incoming=%s)",
                    cls._index.d,
                    embeddings.shape[1],
                )
                return

            cls._index.add(embeddings)
            cls._metadata.extend((receipt_id, text) for text in cleaned)
            logger.debug(
                "EmbeddingIndex: indexed %s vector(s) for receipt %s (total=%s)",
                len(cleaned),
                receipt_id,
                len(cls._metadata),
            )

    def search(self, query: str, k: int = 10) -> List[dict]:
        query = (query or "").strip()
        if not query:
            return []

        vector = self._embed([query])
        if vector.size == 0:
            return []

        cls = type(self)

        with cls._lock:
            if cls._index is None or not cls._metadata:
                return []
            k = min(k, len(cls._metadata))
            distances, indices = cls._index.search(vector, k)

        results: List[dict] = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx < 0 or idx >= len(cls._metadata):
                continue
            receipt_id, text = cls._metadata[idx]
            results.append({"receipt_id": receipt_id, "text": text, "distance": float(distance)})
        return results

    def _embed(self, texts: List[str]) -> np.ndarray:
        if LLM_PROVIDER == "openai":
            return _embed_openai(texts)
        return _embed_ollama(texts)


def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Embed a batch of texts using the configured embedding model.
    Returns a list of vectors (or empty lists for failed items).
    """
    index = EmbeddingIndex()
    matrix = index._embed(texts)
    return matrix.tolist() if matrix.size else []


def _embed_ollama(texts: List[str]) -> np.ndarray:
    vectors = []
    endpoint = f"{OLLAMA_CFG.get('url', 'http://ollama:11434').rstrip('/')}/api/embeddings"
    for text in texts:
        payload = {"model": EMBEDDING_MODEL, "prompt": text}
        try:
            response = requests.post(endpoint, json=payload, timeout=LLM_TIMEOUT)
            response.raise_for_status()
            embedding = response.json().get("embedding")
        except requests.RequestException as exc:
            logger.error("EmbeddingIndex: embedding request failed: %s", exc)
            continue

        if not embedding:
            logger.warning("EmbeddingIndex: empty embedding for text %r", text)
            continue
        vectors.append(np.array(embedding, dtype=np.float32))

    if not vectors:
        return np.empty((0,), dtype=np.float32)
    return np.vstack(vectors)


def _embed_openai(texts: List[str]) -> np.ndarray:
    if _openai_client is None:
        logger.error("EmbeddingIndex: OpenAI client not configured; set OPENAI_API_KEY")
        return np.empty((0,), dtype=np.float32)
    try:
        resp = _openai_client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
        data = resp.data
    except Exception as exc:
        logger.error("EmbeddingIndex: OpenAI embedding request failed: %s", exc)
        return np.empty((0,), dtype=np.float32)

    vectors = []
    for item in data:
        vec = getattr(item, "embedding", None)
        if not vec:
            continue
        vectors.append(np.array(vec, dtype=np.float32))
    if not vectors:
        return np.empty((0,), dtype=np.float32)
    return np.vstack(vectors)
