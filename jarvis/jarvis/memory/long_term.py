"""Long-term vector memory backed by Chroma + sentence-transformers."""
from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

from jarvis.logging_setup import get_logger

log = get_logger(__name__)


class LongTermMemory:
    """Persistent vector store. Lazy-loads heavy deps."""

    def __init__(
        self,
        chroma_path: str,
        collection: str,
        embedding_model: str = "all-MiniLM-L6-v2",
    ) -> None:
        self.chroma_path = chroma_path
        self.collection_name = collection
        self.embedding_model_name = embedding_model
        self._client = None
        self._collection = None
        self._embedder = None
        Path(chroma_path).mkdir(parents=True, exist_ok=True)

    def _ensure(self) -> None:
        if self._collection is not None:
            return
        try:
            import chromadb  # type: ignore
            from sentence_transformers import SentenceTransformer  # type: ignore
        except ImportError as e:  # pragma: no cover
            raise RuntimeError(
                "chromadb and sentence-transformers are required for long-term memory."
            ) from e

        log.info(
            "Initializing long-term memory: model=%s path=%s",
            self.embedding_model_name,
            self.chroma_path,
        )
        self._embedder = SentenceTransformer(self.embedding_model_name)
        self._client = chromadb.PersistentClient(path=self.chroma_path)
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def _embed(self, texts: list[str]) -> list[list[float]]:
        self._ensure()
        return self._embedder.encode(texts, normalize_embeddings=True).tolist()  # type: ignore

    def add(self, text: str, metadata: dict[str, Any] | None = None) -> str:
        self._ensure()
        doc_id = str(uuid.uuid4())
        emb = self._embed([text])
        meta = {k: v for k, v in (metadata or {}).items() if isinstance(v, (str, int, float, bool))}
        self._collection.add(  # type: ignore
            ids=[doc_id],
            documents=[text],
            embeddings=emb,
            metadatas=[meta],
        )
        return doc_id

    def query(
        self,
        query_text: str,
        k: int = 5,
        where: dict[str, Any] | None = None,
    ) -> list[tuple[str, dict[str, Any], float]]:
        """Return list of (document, metadata, distance)."""
        self._ensure()
        if self._collection.count() == 0:  # type: ignore
            return []
        emb = self._embed([query_text])
        res = self._collection.query(  # type: ignore
            query_embeddings=emb,
            n_results=min(k, self._collection.count()),  # type: ignore
            where=where,
        )
        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        dists = res.get("distances", [[]])[0]
        out: list[tuple[str, dict[str, Any], float]] = []
        for d, m, di in zip(docs, metas, dists):
            out.append((d, dict(m or {}), float(di)))
        return out
