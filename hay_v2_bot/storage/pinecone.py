"""Pinecone preflight and document-store factory helpers for Stage 5."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from haystack.utils import Secret
from haystack_integrations.document_stores.pinecone import PineconeDocumentStore
from pinecone import NotFoundError, Pinecone

from hay_v2_bot.config import DocumentRagSettings
from hay_v2_bot.storage.namespaces import document_namespace_for_user

_EXPECTED_METRIC = "cosine"
_NOT_FOUND_STATUS = 404
_MISSING = object()


class DocumentStoreError(Exception):
    """Base class for document-store setup and cleanup failures."""


class DocumentIndexUnavailableError(DocumentStoreError):
    """Raised when the configured Pinecone index cannot be reached safely."""


class DocumentIndexContractError(DocumentStoreError):
    """Raised when the configured Pinecone index violates the required contract."""


class DocumentCleanupError(DocumentStoreError):
    """Raised when deterministic cleanup of document IDs fails."""


@dataclass(frozen=True)
class DocumentIndexInfo:
    """Validated public information about the configured Pinecone index."""

    name: str
    host: str
    dimension: int
    metric: str
    ready: bool
    state: str | None = None


def _get_field(source: Any, name: str) -> Any:
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


def _is_not_found_error(exc: Exception) -> bool:
    if isinstance(exc, NotFoundError):
        return True

    status_code = getattr(exc, "status_code", None)
    if isinstance(status_code, int) and not isinstance(status_code, bool):
        return status_code == _NOT_FOUND_STATUS

    status = getattr(exc, "status", None)
    if isinstance(status, int) and not isinstance(status, bool):
        return status == _NOT_FOUND_STATUS

    return False


def _normalize_document_ids(document_ids: Sequence[str]) -> tuple[str, ...]:
    if isinstance(document_ids, str | bytes) or not isinstance(document_ids, Sequence):
        raise DocumentCleanupError("document_ids must be a non-empty sequence")

    normalized: list[str] = []
    seen: set[str] = set()
    for document_id in document_ids:
        if not isinstance(document_id, str) or not document_id.strip():
            raise DocumentCleanupError("document_ids must contain only non-empty string ids")
        if document_id in seen:
            continue
        normalized.append(document_id)
        seen.add(document_id)
    if not normalized:
        raise DocumentCleanupError("document_ids must not be empty")
    return tuple(normalized)


class PineconeDocumentStoreFactory:
    """Validate one existing Pinecone index and build per-user document stores."""

    def __init__(
        self,
        settings: DocumentRagSettings,
        *,
        pinecone_client: Any | None = None,
        pinecone_client_factory: Callable[[str], Any] | None = None,
    ) -> None:
        self._settings = settings
        self._pinecone_client = pinecone_client
        self._pinecone_client_factory = pinecone_client_factory
        self._index_info: DocumentIndexInfo | None = None
        self._index_handle: Any | None = None

    def describe_index(self, *, refresh: bool = False) -> DocumentIndexInfo:
        """Return the validated Pinecone index description without creating anything."""
        if self._index_info is not None and not refresh:
            return self._index_info

        index_name = self._settings.PINECONE_INDEX_NAME
        try:
            raw_description = self._client.describe_index(index_name)
        except Exception as exc:
            if _is_not_found_error(exc):
                raise DocumentIndexUnavailableError(
                    "Configured Pinecone index does not exist"
                ) from exc
            raise DocumentIndexUnavailableError(
                "Failed to describe the configured Pinecone index"
            ) from exc

        name = _get_field(raw_description, "name")
        host = _get_field(raw_description, "host")
        dimension = _get_field(raw_description, "dimension")
        metric = _get_field(raw_description, "metric")
        status = _get_field(raw_description, "status")
        ready = _get_field(status, "ready")
        state = _get_field(status, "state")

        if not isinstance(name, str) or not name.strip():
            raise DocumentIndexContractError("Pinecone index description is missing a valid name")
        if not isinstance(host, str) or not host.strip():
            raise DocumentIndexContractError("Pinecone index description is missing a valid host")
        if not isinstance(dimension, int) or isinstance(dimension, bool) or dimension <= 0:
            raise DocumentIndexContractError(
                "Pinecone index description is missing a valid positive dimension"
            )
        if not isinstance(metric, str) or metric.strip().lower() != _EXPECTED_METRIC:
            raise DocumentIndexContractError("Pinecone index metric must be cosine")
        if dimension != self._settings.embedding_dimensions:
            raise DocumentIndexContractError(
                "Pinecone index dimension does not match the configured embedding dimensions"
            )
        if ready is not True:
            raise DocumentIndexContractError("Pinecone index is not ready")

        self._index_info = DocumentIndexInfo(
            name=name,
            host=host,
            dimension=dimension,
            metric=metric,
            ready=True,
            state=state if isinstance(state, str) else None,
        )
        self._index_handle = None
        return self._index_info

    def create_document_store(self, user_id: int) -> PineconeDocumentStore:
        """Return a PineconeDocumentStore bound to the user's document namespace."""
        namespace = document_namespace_for_user(user_id)
        index_info = self.describe_index()
        document_store = PineconeDocumentStore(
            api_key=Secret.from_token(self._settings.PINECONE_API_KEY.get_secret_value()),
            index=index_info.name,
            namespace=namespace,
            dimension=index_info.dimension,
            metric=_EXPECTED_METRIC,
            show_progress=False,
        )
        document_store._index = self._get_index_handle(index_info)  # type: ignore[attr-defined]
        document_store._dummy_vector = [-10.0] * index_info.dimension  # type: ignore[attr-defined]
        document_store.dimension = index_info.dimension
        document_store.metric = _EXPECTED_METRIC
        return document_store

    def delete_documents(self, user_id: int, document_ids: Sequence[str]) -> None:
        """Delete only the specified document IDs from one user namespace."""
        namespace = document_namespace_for_user(user_id)
        normalized_ids = _normalize_document_ids(document_ids)
        try:
            self._get_index_handle(self.describe_index()).delete(
                ids=list(normalized_ids),
                namespace=namespace,
            )
        except Exception as exc:
            if _is_not_found_error(exc):
                return
            raise DocumentCleanupError("Document cleanup failed") from exc

    def fetch_existing_document_ids(
        self,
        user_id: int,
        document_ids: Sequence[str],
    ) -> tuple[str, ...]:
        """Return the subset of requested IDs that still exist in the user namespace."""
        namespace = document_namespace_for_user(user_id)
        normalized_ids = _normalize_document_ids(document_ids)
        try:
            response = self._get_index_handle(self.describe_index()).fetch(
                ids=list(normalized_ids),
                namespace=namespace,
            )
        except Exception as exc:
            raise DocumentCleanupError("Document cleanup verification failed") from exc

        vectors = _get_field(response, "vectors")
        if not isinstance(vectors, Mapping):
            raise DocumentCleanupError("Document cleanup verification failed")
        return tuple(document_id for document_id in normalized_ids if document_id in vectors)

    @property
    def _client(self) -> Any:
        if self._pinecone_client is None:
            api_key = self._settings.PINECONE_API_KEY.get_secret_value()
            if self._pinecone_client_factory is None:
                self._pinecone_client = Pinecone(api_key=api_key)
            else:
                self._pinecone_client = self._pinecone_client_factory(api_key)
        return self._pinecone_client

    def _get_index_handle(self, index_info: DocumentIndexInfo) -> Any:
        if self._index_handle is None:
            try:
                self._index_handle = self._client.Index(host=index_info.host)
            except Exception as exc:
                raise DocumentIndexUnavailableError(
                    "Failed to connect to the configured Pinecone index"
                ) from exc
        return self._index_handle
