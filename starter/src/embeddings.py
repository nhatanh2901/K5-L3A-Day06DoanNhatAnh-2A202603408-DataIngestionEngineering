"""Sentence Transformers adapter using one real multilingual embedding model."""

from typing import Protocol


DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_MODEL_REVISION = "e8f8c211226b894fcb81acc59f3b34ba3efd5f42"


class Embedder(Protocol):
    model_name: str
    dimensions: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Return one vector per input text, in the same order."""
        ...

    def embed_query(self, question: str) -> list[float]:
        """Return a query vector compatible with stored document vectors."""
        ...


class SentenceTransformerEmbedder:
    """Local CPU/GPU model adapter; model download errors are surfaced to the caller."""

    def __init__(
        self, model_name: str = DEFAULT_MODEL, revision: str | None = None
    ) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "Embedding dependency is missing; install starter/requirements.txt."
            ) from exc

        if revision is None and model_name == DEFAULT_MODEL:
            revision = DEFAULT_MODEL_REVISION
        self.model_name = model_name
        self._model = SentenceTransformer(model_name, revision=revision)
        dimensions = self._model.get_embedding_dimension()
        if dimensions is None:
            raise RuntimeError(f"Model {model_name!r} did not report its dimensions.")
        self.dimensions = dimensions
        self.max_sequence_length = int(self._model.max_seq_length)
        first_module = self._model[0]
        model_config = getattr(first_module, "auto_model", None)
        config = getattr(model_config, "config", None)
        self.revision = (
            revision
            or getattr(config, "_commit_hash", None)
            or "not exposed by model loader"
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        token_counts = self.count_tokens(texts)
        if any(count > self.max_sequence_length for count in token_counts):
            raise ValueError(
                f"An input exceeds the model limit of {self.max_sequence_length} "
                "tokens; reduce chunk size rather than relying on truncation."
            )
        vectors = self._model.encode(
            texts,
            batch_size=32,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=True,
        )
        return vectors.tolist()

    def embed_query(self, question: str) -> list[float]:
        if self.count_tokens([question])[0] > self.max_sequence_length:
            raise ValueError(
                f"Query exceeds the model limit of {self.max_sequence_length} tokens."
            )
        vectors = self._model.encode(
            [question],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return vectors[0].tolist()

    def count_tokens(self, texts: list[str]) -> list[int]:
        tokenized = self._model.tokenizer(
            texts,
            add_special_tokens=True,
            padding=False,
            truncation=False,
        )
        return [len(token_ids) for token_ids in tokenized["input_ids"]]
