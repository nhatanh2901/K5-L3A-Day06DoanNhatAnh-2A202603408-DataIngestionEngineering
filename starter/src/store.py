"""Persistent Chroma vector storage with metadata filtering."""

import math
import re
from pathlib import Path
from typing import Any

from .models import Chunk


class ChromaVectorStore:
    def __init__(self, persist_directory: Path, dimensions: int) -> None:
        if dimensions <= 0:
            raise ValueError("Embedding dimensions must be greater than zero.")
        try:
            import chromadb
        except ImportError as exc:
            raise RuntimeError(
                "Vector-store dependency is missing; install starter/requirements.txt."
            ) from exc

        self.dimensions = dimensions
        self._client = chromadb.PersistentClient(path=str(persist_directory))

    def create_schema(self) -> None:
        """PersistentClient creates the local database on first use."""

    def close(self) -> None:
        """Close the persistent client and release its file handles."""
        self._client.close()

    def _collection(self, strategy: str) -> Any:
        safe_name = re.sub(r"[^a-zA-Z0-9_-]+", "_", strategy.lower()).strip("_-")
        if not safe_name:
            raise ValueError("Strategy name must contain a letter or digit.")
        name = f"day6_{safe_name}"[:63]
        return self._client.get_or_create_collection(
            name=name,
            metadata={"hnsw:space": "cosine"},
        )

    def upsert(
        self,
        strategy: str,
        chunks: list[Chunk],
        vectors: list[list[float]],
    ) -> None:
        """Replace prior chunks for the ingested documents, then write current chunks."""
        if not chunks:
            raise ValueError("Cannot upsert an empty chunk list.")
        if len(chunks) != len(vectors):
            raise ValueError("Each chunk must have exactly one embedding vector.")

        ids = [chunk.chunk_id for chunk in chunks]
        if len(ids) != len(set(ids)):
            raise ValueError("Chunk IDs must be unique within an upsert.")
        for vector in vectors:
            if len(vector) != self.dimensions:
                raise ValueError(
                    f"Expected {self.dimensions} dimensions; got {len(vector)}."
                )
            if not all(math.isfinite(value) for value in vector):
                raise ValueError("Embedding vectors must contain only finite numbers.")

        collection = self._collection(strategy)
        document_ids = sorted({chunk.document_id for chunk in chunks})
        for document_id in document_ids:
            collection.delete(where={"document_id": document_id})

        batch_size = 500
        for start in range(0, len(chunks), batch_size):
            batch_chunks = chunks[start : start + batch_size]
            batch_vectors = vectors[start : start + batch_size]
            metadatas = [
                {
                    "document_id": chunk.document_id,
                    "source": chunk.source,
                    "page": chunk.page,
                    **chunk.metadata,
                }
                for chunk in batch_chunks
            ]
            collection.upsert(
                ids=[chunk.chunk_id for chunk in batch_chunks],
                documents=[chunk.text for chunk in batch_chunks],
                metadatas=metadatas,
                embeddings=batch_vectors,
            )

    def search(
        self,
        strategy: str,
        query_vector: list[float],
        top_k: int = 3,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Return nearest chunks, cosine similarity, and stored metadata."""
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")
        if len(query_vector) != self.dimensions:
            raise ValueError(
                f"Expected {self.dimensions} query dimensions; got {len(query_vector)}."
            )
        if not all(math.isfinite(value) for value in query_vector):
            raise ValueError("Query vectors must contain only finite numbers.")

        collection = self._collection(strategy)
        count = collection.count()
        if count == 0:
            return []

        where: dict[str, Any] | None = None
        if filters:
            for key, value in filters.items():
                if not isinstance(value, (str, int, float, bool)):
                    raise ValueError(f"Filter {key!r} must have a scalar value.")
            if len(filters) == 1:
                where = dict(filters)
            else:
                where = {"$and": [{key: value} for key, value in filters.items()]}

        result = collection.query(
            query_embeddings=[query_vector],
            n_results=min(top_k, count),
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        ids = result["ids"][0]
        documents = result["documents"][0]
        metadatas = result["metadatas"][0]
        distances = result["distances"][0]
        return [
            {
                "chunk_id": chunk_id,
                "text": text,
                "score": 1.0 - float(distance),
                "metadata": metadata or {},
            }
            for chunk_id, text, metadata, distance in zip(
                ids, documents, metadatas, distances
            )
        ]
