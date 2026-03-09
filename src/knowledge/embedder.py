from __future__ import annotations

import hashlib
from dataclasses import dataclass
import re
from time import perf_counter

import numpy as np


@dataclass(frozen=True)
class SimpleEmbedder:
    model_name: str = "simple-hash-embedder"
    dim: int = 384


def load_model(model_name: str = "simple-hash-embedder", dim: int = 384, device: str = "cpu") -> SimpleEmbedder:
    # `device` kept for API compatibility with sentence-transformers style usage.
    if dim <= 0:
        raise ValueError("dim must be positive")
    return SimpleEmbedder(model_name=model_name, dim=dim)


def _tokenize(text: str) -> list[str]:
    chunks = re.findall(r"[\u4e00-\u9fa5]+|[A-Za-z0-9_]+", text.lower())
    out: list[str] = []
    for c in chunks:
        if re.fullmatch(r"[\u4e00-\u9fa5]+", c):
            # Chinese fallback tokenization: char unigrams + bigrams.
            out.extend(list(c))
            out.extend([c[i : i + 2] for i in range(len(c) - 1)])
        else:
            out.append(c)
    return [t for t in out if t.strip()]


def normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    arr = np.asarray(vectors, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError("vectors must be a 2D array")
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    return arr / norms


def _token_hash(token: str, dim: int) -> tuple[int, float]:
    h = hashlib.md5(token.encode("utf-8")).digest()
    idx = int.from_bytes(h[:4], byteorder="little", signed=False) % dim
    sign = 1.0 if (h[4] % 2 == 0) else -1.0
    return idx, sign


def embed_texts(
    texts: list[str],
    model: SimpleEmbedder,
    batch_size: int = 64,
    normalize: bool = True,
    return_log: bool = False,
):
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    start = perf_counter()
    dim = model.dim
    vecs = np.zeros((len(texts), dim), dtype=np.float32)

    empty_count = 0
    total_len = 0
    for i, raw in enumerate(texts):
        text = raw.strip() if isinstance(raw, str) else ""
        if not text:
            empty_count += 1
            continue
        total_len += len(text)
        toks = _tokenize(text)
        if not toks:
            continue
        for tok in toks:
            idx, sign = _token_hash(tok, dim)
            vecs[i, idx] += sign

    if normalize:
        vecs = normalize_vectors(vecs)

    if not return_log:
        return vecs

    elapsed = perf_counter() - start
    log = {
        "n_texts": len(texts),
        "empty_ratio": empty_count / max(1, len(texts)),
        "avg_text_len": total_len / max(1, len(texts) - empty_count),
        "elapsed_sec": elapsed,
        "dim": dim,
    }
    return vecs, log


def fine_tune(model: SimpleEmbedder, train_pairs: list[tuple[str, str]], epochs: int = 1) -> dict[str, float | int]:
    # Placeholder for course pipeline compatibility in offline environment.
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    return {
        "status": "skipped",
        "pairs": len(train_pairs),
        "epochs": epochs,
        "note": "SimpleEmbedder has no trainable parameters in this demo implementation.",
    }
