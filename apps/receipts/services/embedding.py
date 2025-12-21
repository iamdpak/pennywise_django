from __future__ import annotations

from typing import Iterable, List, Tuple

import logging
import os
import threading

import faiss
import numpy as np
import requests


logger = logging.getLogger(__name__)


LLM_PROVIDER_URL = os.getenv("LLM_PROVIDER_URL", "http://ollama:11434")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", os.getenv("LLM_MODEL", "llama3.2-vision"))
LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT", "60"))
EMBEDDING_ENDPOINT = f"{LLM_PROVIDER_URL.rstrip('/')}/api/embeddings"


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
        vectors = []
        for text in texts:
            payload = {"model": EMBEDDING_MODEL, "prompt": text}
            try:
                response = requests.post(EMBEDDING_ENDPOINT, json=payload, timeout=LLM_TIMEOUT)
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


def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Embed a batch of texts using the configured embedding model.
    Returns a list of vectors (or empty lists for failed items).
    """
    index = EmbeddingIndex()
    matrix = index._embed(texts)
    return matrix.tolist() if matrix.size else []
