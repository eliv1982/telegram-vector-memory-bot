"""Pure helpers for bounded document context and safe public sources."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence

from haystack import Document
from haystack.dataclasses import ChatMessage

from hay_v2_bot.models.rag import DocumentSource

_MAX_SUMMARY_SENTENCE_CHARS = 500
_SURROUNDING_QUOTES_PATTERN = re.compile(r'^(?P<quote>["\'“”‘’«»])(?P<body>.*)(?P=quote)$')
_FIRST_SENTENCE_PATTERN = re.compile(r"^(.+?[.!?])(?:\s|$)")


def _require_positive_int(value: int, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer")
    return value


def _normalize_text_block(text: str) -> str:
    return " ".join(text.split())


def _chunk_label(document: Document, position: int) -> int:
    if isinstance(document.meta, Mapping):
        chunk_index = document.meta.get("chunk_index")
        if isinstance(chunk_index, int) and not isinstance(chunk_index, bool) and chunk_index >= 0:
            return chunk_index
    return position


def build_summary_context(documents: Sequence[Document], max_chars: int) -> str:
    """Build a bounded summary context from normalized Haystack documents."""
    _require_positive_int(max_chars, "max_chars")
    if isinstance(documents, str | bytes) or not isinstance(documents, Sequence) or not documents:
        raise ValueError("documents must be a non-empty sequence of Haystack Document objects")

    sections: list[str] = []
    usable_chunk_count = 0
    for position, document in enumerate(documents):
        if not isinstance(document, Document):
            raise TypeError("documents must contain only Haystack Document objects")
        if not isinstance(document.content, str):
            continue
        content = _normalize_text_block(document.content)
        if not content:
            continue
        usable_chunk_count += 1
        label = _chunk_label(document, position)
        section = f"[Chunk {label}]\n{content}"
        if sections:
            section = f"\n\n{section}"
        sections.append(section)

    if usable_chunk_count == 0:
        raise ValueError("documents must contain at least one non-empty textual chunk")

    context = "".join(sections)
    return context[:max_chars]


def normalize_single_sentence_summary(text: str) -> str:
    """Normalize model output to one concise sentence."""
    if not isinstance(text, str):
        raise TypeError("summary text must be a string")

    normalized = _normalize_text_block(text)
    if not normalized:
        raise ValueError("summary text must not be empty")

    match = _SURROUNDING_QUOTES_PATTERN.match(normalized)
    if match is not None:
        normalized = match.group("body").strip()
    if not normalized:
        raise ValueError("summary text must not be empty")

    first_sentence_match = _FIRST_SENTENCE_PATTERN.match(normalized)
    sentence = first_sentence_match.group(1) if first_sentence_match is not None else normalized
    sentence = sentence[:_MAX_SUMMARY_SENTENCE_CHARS].strip()
    sentence = sentence.strip("\"'“”‘’«» ")
    sentence = sentence.rstrip(",;:")
    if not sentence:
        raise ValueError("summary text must not be empty")
    if sentence[-1] not in ".!?":
        sentence = f"{sentence}."
    return sentence


def extract_chat_reply_text(replies: Sequence[ChatMessage]) -> str:
    """Extract exactly one usable text reply from Haystack chat messages."""
    if isinstance(replies, str | bytes) or not isinstance(replies, Sequence) or not replies:
        raise ValueError("chat replies must be a non-empty sequence")

    texts: list[str] = []
    for reply in replies:
        if not isinstance(reply, ChatMessage):
            raise TypeError("chat replies must contain Haystack ChatMessage objects")
        for text_part in reply.texts:
            if isinstance(text_part, str):
                stripped = text_part.strip()
                if stripped:
                    texts.append(stripped)

    if len(texts) != 1:
        raise ValueError("chat replies must contain exactly one usable text reply")
    return texts[0]


def build_sources(documents: Sequence[Document]) -> tuple[DocumentSource, ...]:
    """Convert retrieved Haystack documents into public source metadata."""
    if isinstance(documents, str | bytes) or not isinstance(documents, Sequence):
        raise TypeError("documents must be a sequence of Haystack Document objects")

    sources: list[DocumentSource] = []
    seen_ids: set[str] = set()
    for document in documents:
        if not isinstance(document, Document):
            raise TypeError("documents must contain only Haystack Document objects")
        document_id = document.id
        if not isinstance(document_id, str) or not document_id.strip():
            raise ValueError("retrieved document is missing a valid id")
        if document_id in seen_ids:
            continue
        seen_ids.add(document_id)

        metadata = document.meta
        if not isinstance(metadata, Mapping):
            raise ValueError("retrieved document metadata must be a mapping")

        payload: dict[str, object] = {
            "document_id": document_id,
            "file_name": metadata.get("file_name"),
            "chunk_index": metadata.get("chunk_index"),
        }
        if "page_number" in metadata:
            payload["page_number"] = metadata.get("page_number")

        score = document.score
        if isinstance(score, int | float) and not isinstance(score, bool):
            normalized_score = float(score)
            if math.isfinite(normalized_score):
                payload["score"] = normalized_score

        sources.append(DocumentSource(**payload))

    return tuple(sources)
