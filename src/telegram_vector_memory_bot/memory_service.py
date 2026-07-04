"""Application-level memory service built on top of PineconeManager.

``MemoryService`` orchestrates :class:`~telegram_vector_memory_bot.
memory_policy.MemoryPolicy` (pure deduplication rules) and
:class:`~telegram_vector_memory_bot.pinecone_manager.PineconeManager`
(infrastructure) to implement ``remember`` / ``recall`` / ``forget_user``.
It creates no external clients itself and never queries or deletes across
namespace boundaries.

For this educational MVP, every valid user message passed to
:meth:`MemoryService.remember` is considered eligible for memory -- there is
no separate classifier deciding whether a message is "worth remembering".
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from pydantic import ValidationError

from .config import Settings
from .memory_policy import MemoryPolicy
from .models import (
    MemoryAction,
    MemoryReason,
    MemoryRecord,
    MemoryWriteResult,
    RecalledMemory,
    VectorMatch,
)
from .pinecone_manager import PineconeManager

_RECORD_TYPE = "user_memory"
_RECORD_TYPE_FILTER = {"record_type": {"$eq": _RECORD_TYPE}}


class MemoryServiceError(Exception):
    """Base exception for the memory service application layer."""


class StoredMemoryFormatError(MemoryServiceError):
    """A stored memory's metadata is malformed and cannot be safely parsed."""


def _utc_now() -> datetime:
    return datetime.now(UTC)


class MemoryService:
    """Orchestrates deduplication policy and Pinecone-backed storage.

    Behavioral boundaries: this service never saves bot responses, never
    updates or overwrites an existing vector because of similarity, never
    queries or deletes across namespaces, and never uses a Telegram
    username as identity -- the namespace is derived only from the
    numeric user ID.
    """

    def __init__(
        self,
        *,
        manager: PineconeManager,
        settings: Settings,
        policy: MemoryPolicy | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._manager = manager
        self._settings = settings
        self._policy = policy or MemoryPolicy(settings.MEMORY_SIMILARITY_THRESHOLD)
        self._clock = clock or _utc_now

    def namespace_for_user(self, user_id: int) -> str:
        """Return the Pinecone namespace for *user_id*."""
        return self._policy.namespace_for_user(
            prefix=self._settings.MEMORY_NAMESPACE_PREFIX, user_id=user_id
        )

    def remember(
        self,
        *,
        user_id: int,
        text: str,
        username: str | None = None,
        first_name: str | None = None,
        last_name: str | None = None,
    ) -> MemoryWriteResult:
        """Store *text* as a memory for *user_id*, deduplicating along the way."""
        if user_id <= 0:
            raise ValueError("user_id must be positive")
        if not text.strip():
            raise ValueError("text must not be empty or whitespace-only")

        namespace = self.namespace_for_user(user_id)
        content_hash = self._policy.content_hash(text)
        memory_id = self._policy.memory_id_for_text(text)

        existing = self._manager.fetch_vectors(vector_ids=[memory_id], namespace=namespace)
        if memory_id in existing:
            return MemoryWriteResult(
                action=MemoryAction.SKIPPED,
                reason=MemoryReason.EXACT_DUPLICATE,
                memory_id=None,
                existing_id=memory_id,
                similarity_score=None,
            )

        embedding = self._manager.create_embedding(text)

        matches = self._manager.query_by_vector(
            values=embedding,
            namespace=namespace,
            top_k=1,
            metadata_filter=_RECORD_TYPE_FILTER,
        )

        if matches:
            duplicate_result = self._check_semantic_duplicate(new_text=text, candidate=matches[0])
            if duplicate_result is not None:
                return duplicate_result

        created_at = self._clock()
        if created_at.tzinfo is None:
            raise MemoryServiceError("clock must return a timezone-aware datetime")

        record = MemoryRecord(
            memory_id=memory_id,
            user_id=user_id,
            text=text,
            content_hash=content_hash,
            created_at=created_at,
            username=username,
            first_name=first_name,
            last_name=last_name,
        )

        metadata = _build_metadata(record)

        self._manager.upsert_vector(
            vector_id=memory_id,
            values=embedding,
            metadata=metadata,
            namespace=namespace,
        )

        return MemoryWriteResult(
            action=MemoryAction.INSERTED,
            reason=MemoryReason.NEW_MEMORY,
            memory_id=memory_id,
            existing_id=None,
            similarity_score=None,
        )

    def _check_semantic_duplicate(
        self, *, new_text: str, candidate: VectorMatch
    ) -> MemoryWriteResult | None:
        candidate_text = candidate.metadata.get("text")
        if not isinstance(candidate_text, str) or not candidate_text.strip():
            # Conservative: if we can't read the candidate's original text,
            # never skip -- always insert the new memory instead.
            return None

        if not self._policy.is_semantic_duplicate(
            new_text=new_text,
            existing_text=candidate_text,
            similarity_score=candidate.score,
        ):
            return None

        return MemoryWriteResult(
            action=MemoryAction.SKIPPED,
            reason=MemoryReason.SEMANTIC_DUPLICATE,
            memory_id=None,
            existing_id=candidate.vector_id,
            similarity_score=candidate.score,
        )

    def recall(
        self,
        *,
        user_id: int,
        query: str,
        top_k: int | None = None,
    ) -> list[RecalledMemory]:
        """Return memories for *user_id* relevant to *query*, most similar first."""
        if user_id <= 0:
            raise ValueError("user_id must be positive")
        if not query.strip():
            raise ValueError("query must not be empty or whitespace-only")

        resolved_top_k = top_k if top_k is not None else self._settings.MEMORY_TOP_K
        namespace = self.namespace_for_user(user_id)

        matches = self._manager.query_by_text(
            text=query,
            namespace=namespace,
            top_k=resolved_top_k,
            metadata_filter=_RECORD_TYPE_FILTER,
        )

        return [_parse_recalled_memory(match) for match in matches]

    def forget_user(self, *, user_id: int) -> None:
        """Delete all memories for *user_id*. Never touches other namespaces or the index."""
        if user_id <= 0:
            raise ValueError("user_id must be positive")
        namespace = self.namespace_for_user(user_id)
        self._manager.delete_namespace(namespace)


def _build_metadata(record: MemoryRecord) -> dict[str, Any]:
    """Build the safe, storage-ready metadata for a memory record.

    Only a fixed, known-safe set of fields is stored: no bot responses, API
    keys, tokens, full Telegram update objects, prompts, or chat history.
    """
    metadata: dict[str, Any] = {
        "user_id": record.user_id,
        "text": record.text,
        "content_hash": record.content_hash,
        "created_at": record.created_at.astimezone(UTC).isoformat(),
        "source": record.source,
        "record_type": _RECORD_TYPE,
    }
    if record.username is not None:
        metadata["username"] = record.username
    if record.first_name is not None:
        metadata["first_name"] = record.first_name
    if record.last_name is not None:
        metadata["last_name"] = record.last_name
    return metadata


def _require_metadata_str(metadata: dict[str, Any], field_name: str) -> str:
    value = metadata.get(field_name)
    if not isinstance(value, str) or not value.strip():
        raise StoredMemoryFormatError(f"stored memory is missing valid '{field_name}' metadata")
    return value


def _parse_recalled_memory(match: VectorMatch) -> RecalledMemory:
    metadata = match.metadata

    text = _require_metadata_str(metadata, "text")
    content_hash = _require_metadata_str(metadata, "content_hash")
    source = _require_metadata_str(metadata, "source")
    created_at_raw = metadata.get("created_at")

    if not isinstance(created_at_raw, str) or not created_at_raw.strip():
        raise StoredMemoryFormatError("stored memory is missing valid 'created_at' metadata")

    try:
        created_at = datetime.fromisoformat(created_at_raw)
    except ValueError as exc:
        raise StoredMemoryFormatError(
            "stored memory has a malformed 'created_at' timestamp"
        ) from exc

    if created_at.tzinfo is None:
        raise StoredMemoryFormatError("stored memory 'created_at' must be timezone-aware")

    username = metadata.get("username")
    first_name = metadata.get("first_name")
    last_name = metadata.get("last_name")

    try:
        return RecalledMemory(
            memory_id=match.vector_id,
            text=text,
            score=match.score,
            created_at=created_at,
            source=source,
            content_hash=content_hash,
            username=username if isinstance(username, str) else None,
            first_name=first_name if isinstance(first_name, str) else None,
            last_name=last_name if isinstance(last_name, str) else None,
        )
    except ValidationError as exc:
        raise StoredMemoryFormatError("stored memory failed validation") from exc
