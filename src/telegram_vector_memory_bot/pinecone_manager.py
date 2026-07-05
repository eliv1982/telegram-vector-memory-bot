"""Pinecone-backed vector storage and OpenAI-compatible embedding generation.

``PineconeManager`` is a thin, typed infrastructure adapter: it validates the
configured Pinecone index, generates embeddings, and performs namespace-scoped
vector CRUD operations. It has no opinion on deduplication, memory policy, or
conversational behavior -- those belong to higher-level services built on top
of this adapter. External clients are created only when a ``PineconeManager``
is instantiated, never at import time.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping, Sequence
from typing import Any, Final

from openai import OpenAI
from pinecone import NotFoundError, Pinecone
from pydantic import ValidationError

from .config import Settings
from .models import IndexInfo, VectorMatch

logger = logging.getLogger(__name__)

_MIN_TOP_K: Final = 1
_MAX_TOP_K: Final = 20
_EXPECTED_METRIC: Final = "cosine"
_NOT_FOUND_STATUS: Final = 404
_COSINE_SCORE_EPSILON: Final = 1e-6

_MISSING: Final = object()


class VectorMemoryError(Exception):
    """Base exception for the vector memory infrastructure layer."""


class IndexConfigurationError(VectorMemoryError):
    """The configured Pinecone index is missing, misconfigured, or not ready."""


class EmbeddingGenerationError(VectorMemoryError):
    """An embedding could not be generated or parsed."""


class VectorStorageError(VectorMemoryError):
    """Writing to or deleting from the vector store failed."""


class VectorQueryError(VectorMemoryError):
    """Reading from the vector store failed or returned a malformed response."""


def _get_field(source: Any, name: str) -> Any:
    """Read *name* from *source*, supporting mapping, attribute, and to_dict() forms."""
    if isinstance(source, Mapping):
        return source.get(name, _MISSING)

    to_dict = getattr(source, "to_dict", None)
    if callable(to_dict):
        try:
            as_dict = to_dict()
        except Exception:
            as_dict = None
        if isinstance(as_dict, Mapping) and name in as_dict:
            return as_dict[name]

    return getattr(source, name, _MISSING)


def _to_plain_dict(source: Any) -> dict[str, Any]:
    """Best-effort conversion of an SDK response object to a new, plain dict."""
    if isinstance(source, Mapping):
        return dict(source)

    to_dict = getattr(source, "to_dict", None)
    if callable(to_dict):
        result = to_dict()
        if isinstance(result, Mapping):
            return dict(result)

    raise TypeError(f"cannot convert {type(source).__name__} to a plain dict")


def _is_not_found_error(exc: Exception) -> bool:
    """Detect a Pinecone not-found response without parsing exception text.

    Prefers the official ``NotFoundError`` type; falls back to a safely
    inspected ``status_code`` (current SDK) or ``status`` (legacy SDK
    variants) attribute equal to 404, for compatibility across SDK versions.
    """
    if isinstance(exc, NotFoundError):
        return True

    status_code = getattr(exc, "status_code", None)
    if isinstance(status_code, int) and not isinstance(status_code, bool):
        if status_code == _NOT_FOUND_STATUS:
            return True

    status = getattr(exc, "status", None)
    if isinstance(status, int) and not isinstance(status, bool):
        if status == _NOT_FOUND_STATUS:
            return True

    return False


def _require_non_empty_str(value: Any, field_name: str, error_cls: type[VectorMemoryError]) -> str:
    if not isinstance(value, str) or not value.strip():
        raise error_cls(f"{field_name} must not be empty")
    return value


def _normalize_vector(
    values: Any,
    *,
    field_name: str,
    error_cls: type[VectorMemoryError],
) -> list[float]:
    """Validate and normalize a vector to ``list[float]``, rejecting bools and non-numerics."""
    if isinstance(values, str | bytes) or not isinstance(values, Sequence):
        raise error_cls(f"{field_name} must be a sequence of numbers")
    if len(values) == 0:
        raise error_cls(f"{field_name} must not be empty")

    normalized: list[float] = []
    for item in values:
        if isinstance(item, bool) or not isinstance(item, int | float):
            raise error_cls(f"{field_name} must contain only numeric, non-boolean values")
        normalized.append(float(item))
    return normalized


def _require_dimension(
    values: list[float],
    expected: int,
    error_cls: type[VectorMemoryError],
) -> None:
    if len(values) != expected:
        raise error_cls(
            f"vector dimension {len(values)} does not match index dimension {expected}"
        )


def _build_openai_client(settings: Settings) -> OpenAI:
    kwargs: dict[str, Any] = {"api_key": settings.OPENAI_API_KEY.get_secret_value()}
    base_url = settings.OPENAI_BASE_URL
    if base_url is not None and base_url.strip():
        kwargs["base_url"] = base_url
    return OpenAI(**kwargs)


def _normalize_cosine_score(score: Any, error_cls: type[VectorMemoryError]) -> float:
    """Normalize a raw Pinecone query-match score to the canonical [-1, 1] range.

    Pinecone cosine scores are conceptually normalized to [-1, 1], but the SDK
    returns ordinary floating-point values that can microscopically overshoot
    that boundary (e.g. ``1.0000001``) due to floating-point arithmetic on the
    external service. Only that microscopic drift -- within
    ``_COSINE_SCORE_EPSILON`` of +-1 -- is clamped to the exact boundary;
    values materially outside [-1, 1], and non-finite values, are still
    rejected. This tolerance is specific to this external-response boundary
    and is never applied to user input or configured thresholds.
    """
    if isinstance(score, bool) or not isinstance(score, int | float):
        raise error_cls("query match score must be numeric")

    value = float(score)
    if not math.isfinite(value):
        raise error_cls("query match score must be a finite number")

    if -1.0 <= value <= 1.0:
        return value
    if 1.0 < value <= 1.0 + _COSINE_SCORE_EPSILON:
        logger.debug("Normalized cosine score at floating-point boundary")
        return 1.0
    if -1.0 - _COSINE_SCORE_EPSILON <= value < -1.0:
        logger.debug("Normalized cosine score at floating-point boundary")
        return -1.0

    raise error_cls("query match score is out of the valid [-1, 1] range")


def _parse_vector_match(raw_match: Any) -> VectorMatch:
    match_id = _get_field(raw_match, "id")
    if not isinstance(match_id, str) or not match_id.strip():
        raise VectorQueryError("query match is missing a valid id")

    score = _normalize_cosine_score(_get_field(raw_match, "score"), VectorQueryError)

    metadata = _get_field(raw_match, "metadata")
    if metadata is _MISSING or metadata is None:
        metadata = {}
    elif not isinstance(metadata, Mapping):
        raise VectorQueryError("query match metadata must be a mapping")
    else:
        metadata = dict(metadata)

    try:
        return VectorMatch(vector_id=match_id, score=score, metadata=metadata)
    except ValidationError as exc:
        raise VectorQueryError("query match failed validation") from exc


class PineconeManager:
    """Typed infrastructure adapter over Pinecone and OpenAI.

    This class does not decide whether text is a duplicate and never returns
    a ``MemoryWriteResult`` -- it only performs infrastructure-level vector
    operations against a single, pre-validated Pinecone index.
    """

    def __init__(
        self,
        settings: Settings,
        *,
        pinecone_client: Pinecone | None = None,
        openai_client: OpenAI | None = None,
    ) -> None:
        self._settings = settings
        self._pinecone = pinecone_client or Pinecone(
            api_key=settings.PINECONE_API_KEY.get_secret_value()
        )
        self._openai = openai_client or _build_openai_client(settings)
        self._index_info = self._discover_index()
        self._index = self._pinecone.Index(host=self._index_info.host)

    @property
    def index_info(self) -> IndexInfo:
        """The validated, resolved description of the configured Pinecone index."""
        return self._index_info

    def _discover_index(self) -> IndexInfo:
        index_name = self._settings.PINECONE_INDEX_NAME
        try:
            raw = self._pinecone.describe_index(index_name)
        except Exception as exc:
            raise IndexConfigurationError(
                f"failed to describe Pinecone index {index_name!r}"
            ) from exc

        name = _get_field(raw, "name")
        host = _get_field(raw, "host")
        dimension = _get_field(raw, "dimension")
        metric = _get_field(raw, "metric")
        status = _get_field(raw, "status")
        ready = _get_field(status, "ready")
        state = _get_field(status, "state")

        if not isinstance(name, str) or not name.strip():
            raise IndexConfigurationError("index description is missing a valid 'name'")
        if not isinstance(host, str) or not host.strip():
            raise IndexConfigurationError("index description is missing a valid 'host'")
        if not isinstance(dimension, int) or isinstance(dimension, bool) or dimension <= 0:
            raise IndexConfigurationError(
                "index description is missing a valid positive 'dimension'"
            )
        if not isinstance(metric, str) or metric.strip().lower() != _EXPECTED_METRIC:
            raise IndexConfigurationError(
                f"index metric must be {_EXPECTED_METRIC!r} (case-insensitive)"
            )
        if ready is not True:
            raise IndexConfigurationError("index is not ready")

        return IndexInfo(
            name=name,
            host=host,
            dimension=dimension,
            metric=metric,
            ready=True,
            state=state if isinstance(state, str) else None,
        )

    def create_embedding(self, text: str) -> list[float]:
        """Generate and validate an embedding for *text* using the configured model."""
        if not text.strip():
            raise EmbeddingGenerationError("text must not be empty or whitespace-only")

        try:
            response = self._openai.embeddings.create(
                input=text,
                model=self._settings.OPENAI_EMBEDDING_MODEL,
            )
        except Exception as exc:
            raise EmbeddingGenerationError("embedding request failed") from exc

        data = _get_field(response, "data")
        if not isinstance(data, list) or not data:
            raise EmbeddingGenerationError("embedding response contained no data")

        raw_values = _get_field(data[0], "embedding")
        values = _normalize_vector(
            raw_values, field_name="embedding", error_cls=EmbeddingGenerationError
        )

        if len(values) != self._index_info.dimension:
            raise IndexConfigurationError(
                f"embedding dimension {len(values)} does not match index dimension "
                f"{self._index_info.dimension}"
            )

        return values

    def upsert_vector(
        self,
        *,
        vector_id: str,
        values: Sequence[float],
        metadata: Mapping[str, Any],
        namespace: str,
    ) -> None:
        """Insert or overwrite a single vector in *namespace*."""
        vector_id = _require_non_empty_str(vector_id, "vector_id", VectorStorageError)
        namespace = _require_non_empty_str(namespace, "namespace", VectorStorageError)
        normalized_values = _normalize_vector(
            values, field_name="values", error_cls=VectorStorageError
        )
        _require_dimension(normalized_values, self._index_info.dimension, VectorStorageError)
        metadata_copy = dict(metadata)

        try:
            response = self._index.upsert(
                vectors=[
                    {"id": vector_id, "values": normalized_values, "metadata": metadata_copy}
                ],
                namespace=namespace,
            )
        except Exception as exc:
            raise VectorStorageError("upsert request failed") from exc

        count = _get_field(response, "upserted_count")
        if count is not _MISSING and count != 1:
            raise VectorStorageError(f"unexpected upserted vector count: {count!r}")

    def query_by_vector(
        self,
        *,
        values: Sequence[float],
        namespace: str,
        top_k: int,
        metadata_filter: Mapping[str, Any] | None = None,
    ) -> list[VectorMatch]:
        """Query the index for the nearest matches to *values* within *namespace*."""
        namespace = _require_non_empty_str(namespace, "namespace", VectorQueryError)
        if (
            not isinstance(top_k, int)
            or isinstance(top_k, bool)
            or not (_MIN_TOP_K <= top_k <= _MAX_TOP_K)
        ):
            raise VectorQueryError(
                f"top_k must be an integer between {_MIN_TOP_K} and {_MAX_TOP_K}"
            )
        normalized_values = _normalize_vector(
            values, field_name="values", error_cls=VectorQueryError
        )
        _require_dimension(normalized_values, self._index_info.dimension, VectorQueryError)

        query_kwargs: dict[str, Any] = {
            "vector": normalized_values,
            "namespace": namespace,
            "top_k": top_k,
            "include_metadata": True,
        }
        if metadata_filter is not None:
            query_kwargs["filter"] = dict(metadata_filter)

        try:
            response = self._index.query(**query_kwargs)
        except Exception as exc:
            raise VectorQueryError("query request failed") from exc

        raw_matches = _get_field(response, "matches")
        if not isinstance(raw_matches, list):
            raise VectorQueryError("query response is missing a 'matches' list")

        return [_parse_vector_match(raw_match) for raw_match in raw_matches]

    def query_by_text(
        self,
        *,
        text: str,
        namespace: str,
        top_k: int,
        metadata_filter: Mapping[str, Any] | None = None,
    ) -> list[VectorMatch]:
        """Embed *text* and delegate to :meth:`query_by_vector`."""
        values = self.create_embedding(text)
        return self.query_by_vector(
            values=values,
            namespace=namespace,
            top_k=top_k,
            metadata_filter=metadata_filter,
        )

    def fetch_vectors(
        self,
        *,
        vector_ids: Sequence[str],
        namespace: str,
    ) -> dict[str, dict[str, Any]]:
        """Fetch vectors by ID from *namespace*, keyed by vector ID."""
        if not vector_ids:
            raise VectorQueryError("vector_ids must not be empty")

        seen: set[str] = set()
        for vector_id in vector_ids:
            if not isinstance(vector_id, str) or not vector_id.strip():
                raise VectorQueryError("vector_ids must not contain empty ids")
            if vector_id in seen:
                raise VectorQueryError(f"duplicate vector id: {vector_id!r}")
            seen.add(vector_id)

        namespace = _require_non_empty_str(namespace, "namespace", VectorQueryError)

        try:
            response = self._index.fetch(ids=list(vector_ids), namespace=namespace)
        except Exception as exc:
            raise VectorQueryError("fetch request failed") from exc

        raw_vectors = _get_field(response, "vectors")
        if not isinstance(raw_vectors, Mapping):
            raise VectorQueryError("fetch response is missing a 'vectors' mapping")

        result: dict[str, dict[str, Any]] = {}
        for vector_id, raw_vector in raw_vectors.items():
            raw_values = _get_field(raw_vector, "values")
            values = _normalize_vector(
                raw_values, field_name="values", error_cls=VectorQueryError
            )

            metadata = _get_field(raw_vector, "metadata")
            if metadata is _MISSING or metadata is None:
                metadata = {}
            elif not isinstance(metadata, Mapping):
                raise VectorQueryError("fetched vector metadata must be a mapping")
            else:
                metadata = dict(metadata)

            result[vector_id] = {"values": values, "metadata": metadata}

        return result

    def delete_namespace(self, namespace: str) -> None:
        """Delete all vectors in *namespace*. Never deletes the index itself.

        Idempotent: a namespace that is already absent is treated as a
        successfully deleted namespace, since Pinecone reports a not-found
        response for a namespace with no data rather than a no-op success.
        """
        namespace = _require_non_empty_str(namespace, "namespace", VectorStorageError)
        try:
            self._index.delete(delete_all=True, namespace=namespace)
        except Exception as exc:
            if _is_not_found_error(exc):
                logger.info("Namespace already absent; deletion treated as successful")
                return
            raise VectorStorageError("Namespace deletion failed") from exc

    def describe_index_stats(self) -> dict[str, Any]:
        """Return the index-wide stats as a plain, unshared dict."""
        try:
            response = self._index.describe_index_stats()
        except Exception as exc:
            raise VectorQueryError("describe_index_stats request failed") from exc

        try:
            return _to_plain_dict(response)
        except TypeError as exc:
            raise VectorQueryError("describe_index_stats response could not be parsed") from exc
