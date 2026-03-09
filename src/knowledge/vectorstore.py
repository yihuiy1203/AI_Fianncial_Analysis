from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class LocalVectorStore:
    def __init__(self, persist_dir: str | Path):
        self.persist_dir = Path(persist_dir)
        self.ids: list[str] = []
        self.embeddings = np.empty((0, 0), dtype=np.float32)
        self.metadatas: list[dict] = []
        self.documents: list[str] = []

    def add(self, ids: list[str], embeddings: list[list[float]] | np.ndarray, metadatas: list[dict], documents: list[str]) -> None:
        if not (len(ids) == len(metadatas) == len(documents)):
            raise ValueError("ids/metadatas/documents length mismatch")

        emb = np.asarray(embeddings, dtype=np.float32)
        if emb.ndim == 1:
            emb = emb.reshape(1, -1)
        if emb.shape[0] != len(ids):
            raise ValueError("embeddings row count mismatch ids")

        if self.embeddings.size == 0:
            self.embeddings = emb
        else:
            if emb.shape[1] != self.embeddings.shape[1]:
                raise ValueError("embedding dimension mismatch")
            self.embeddings = np.vstack([self.embeddings, emb])

        self.ids.extend([str(i) for i in ids])
        self.metadatas.extend(metadatas)
        self.documents.extend([str(d) for d in documents])

    def search(self, query_vector: np.ndarray, top_k: int = 5, filters: dict | None = None) -> list[dict]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        if self.embeddings.size == 0:
            return []

        q = np.asarray(query_vector, dtype=np.float32).reshape(-1)
        if q.shape[0] != self.embeddings.shape[1]:
            raise ValueError("query dimension mismatch")

        idxs = np.arange(len(self.ids))
        if filters:
            keep = []
            for i in idxs:
                meta = self.metadatas[i]
                ok = True
                for k, v in filters.items():
                    if meta.get(k) != v:
                        ok = False
                        break
                if ok:
                    keep.append(i)
            idxs = np.array(keep, dtype=int)
            if idxs.size == 0:
                return []

        mat = self.embeddings[idxs]
        # assume embeddings are normalized; cosine ~= dot-product
        scores = mat @ q
        order = np.argsort(-scores)[:top_k]

        out = []
        for pos in order:
            i = int(idxs[pos])
            out.append(
                {
                    "id": self.ids[i],
                    "score": float(scores[pos]),
                    "metadata": self.metadatas[i],
                    "document": self.documents[i],
                }
            )
        return out


def create_store(store_type: str = "local", persist_dir: str | Path = "data/embeddings/") -> LocalVectorStore:
    if store_type not in {"local", "chromadb", "faiss"}:
        raise ValueError("store_type must be one of: local/chromadb/faiss")
    # For offline portability, all store_type options map to LocalVectorStore.
    return LocalVectorStore(persist_dir=persist_dir)


def add_documents(store: LocalVectorStore, vectors: np.ndarray, metadatas: list[dict], texts: list[str], ids: list[str] | None = None) -> None:
    if len(vectors) != len(metadatas) or len(vectors) != len(texts):
        raise ValueError("vectors/metadatas/texts length mismatch")

    if ids is None:
        ids = [f"doc_{len(store.ids) + i}" for i in range(len(texts))]

    for meta in metadatas:
        if "stock_code" not in meta or "year" not in meta:
            raise ValueError("metadata missing required keys: stock_code/year")

    store.add(ids=ids, embeddings=vectors, metadatas=metadatas, documents=texts)


def save(store: LocalVectorStore) -> None:
    store.persist_dir.mkdir(parents=True, exist_ok=True)
    np.save(store.persist_dir / "embeddings.npy", store.embeddings)
    payload = {
        "ids": store.ids,
        "metadatas": store.metadatas,
        "documents": store.documents,
    }
    (store.persist_dir / "store.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def load(persist_dir: str | Path) -> LocalVectorStore:
    p = Path(persist_dir)
    emb_path = p / "embeddings.npy"
    json_path = p / "store.json"
    if not emb_path.exists() or not json_path.exists():
        raise FileNotFoundError("persisted store not found")

    store = LocalVectorStore(persist_dir=p)
    store.embeddings = np.load(emb_path).astype(np.float32)
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    store.ids = [str(x) for x in payload.get("ids", [])]
    store.metadatas = list(payload.get("metadatas", []))
    store.documents = [str(x) for x in payload.get("documents", [])]
    return store


def get_stats(store: LocalVectorStore) -> dict[str, object]:
    n_docs = len(store.documents)
    dim = int(store.embeddings.shape[1]) if store.embeddings.ndim == 2 and store.embeddings.size else 0
    years = sorted({m.get("year") for m in store.metadatas if "year" in m})
    stocks = sorted({m.get("stock_code") for m in store.metadatas if "stock_code" in m})
    return {
        "n_docs": n_docs,
        "dim": dim,
        "n_years": len(years),
        "n_stocks": len(stocks),
        "years": years,
        "stocks": stocks,
    }
