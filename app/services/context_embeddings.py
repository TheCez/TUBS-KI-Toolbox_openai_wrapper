from __future__ import annotations

import hashlib
import math
import os
import re


def embedding_dimensions() -> int:
    return max(16, int(os.getenv("TUBS_CONTEXT_EMBEDDING_DIMENSIONS", "64")))


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9_./:-]+", (text or "").lower())


def embed_text(text: str) -> list[float]:
    dims = embedding_dimensions()
    vector = [0.0] * dims
    tokens = _tokens(text)
    if not tokens:
        return vector

    for token in tokens:
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:2], "big") % dims
        sign = 1.0 if digest[2] % 2 == 0 else -1.0
        vector[index] += sign

    norm = math.sqrt(sum(value * value for value in vector))
    if norm <= 0:
        return vector
    return [value / norm for value in vector]


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right:
        return 0.0
    size = min(len(left), len(right))
    return sum(left[idx] * right[idx] for idx in range(size))
