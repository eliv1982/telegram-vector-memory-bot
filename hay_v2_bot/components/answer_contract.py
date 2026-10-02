"""Machine-readable contract between the document-answer prompt and the service.

The generator is asked to reply with one JSON object::

    {"answerable": true, "answer": "...", "source_ids": ["DOC_1"]}
    {"answerable": false, "answer": null, "source_ids": []}

Routing between a document answer and the Agent fallback is decided *only* by this
parsed contract -- never by comparing the model's prose with a localized sentence.
Parsing is deterministic and fails closed: anything that is not exactly the contract
above (invalid JSON, prose, missing/extra/mistyped fields, an empty answer, no
sources, or a source id that was not among the retrieved chunks) is rejected.

Source ids (``DOC_1``, ``DOC_2``, ...) are created per request from the position of
each retrieved chunk. They are internal to the prompt/result contract and are mapped
back to the retrieved documents by the caller; the model's labels are never trusted
beyond that exact mapping.
"""

from __future__ import annotations

import json
import re
from collections.abc import Collection
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Final

DOCUMENT_LABEL_PREFIX: Final = "DOC_"

_ANSWERABLE_KEY: Final = "answerable"
_ANSWER_KEY: Final = "answer"
_SOURCE_IDS_KEY: Final = "source_ids"
_CONTRACT_KEYS: Final = frozenset({_ANSWERABLE_KEY, _ANSWER_KEY, _SOURCE_IDS_KEY})

# A single surrounding Markdown code fence is the one deterministic deviation from
# "bare JSON" that is tolerated: models commonly add it even when told not to, and
# it carries no ambiguity. Any other text around the object is rejected.
_CODE_FENCE_PATTERN: Final = re.compile(
    r"\A```(?:json)?[ \t]*\r?\n(?P<body>.*)\r?\n```\Z", re.DOTALL | re.IGNORECASE
)


class RejectionReason(StrEnum):
    """Fixed, log-safe codes for why a generator reply cannot be a document answer."""

    NOT_ANSWERABLE = "not_answerable"
    INVALID_FORMAT = "invalid_format"
    INVALID_SCHEMA = "invalid_schema"
    EMPTY_ANSWER = "empty_answer"
    NO_SOURCES = "no_sources"
    UNKNOWN_SOURCE = "unknown_source"


class DocumentAnswerRejected(Exception):
    """The generator reply is not a usable, evidence-backed document answer.

    ``NOT_ANSWERABLE`` is the legitimate "the documents do not contain the answer"
    outcome; every other reason is a malformed or unsafe reply. Either way the
    caller must fall back rather than present a document answer.
    """

    def __init__(self, reason: RejectionReason) -> None:
        super().__init__(reason.value)
        self.reason = reason


@dataclass(frozen=True)
class ParsedDocumentAnswer:
    """A validated answer plus the distinct, supplied source labels it cites."""

    answer: str
    source_labels: tuple[str, ...]


def document_label(position: int) -> str:
    """Return the per-request label for the retrieved chunk at 1-based *position*."""
    if isinstance(position, bool) or not isinstance(position, int) or position < 1:
        raise ValueError("position must be a positive integer")
    return f"{DOCUMENT_LABEL_PREFIX}{position}"


def parse_document_answer(raw_text: str, valid_labels: Collection[str]) -> ParsedDocumentAnswer:
    """Parse and validate one generator reply against the answer contract.

    Returns the answer only when the model reports ``answerable: true``, the answer
    is non-blank, at least one source id is given, and *every* source id is one of
    *valid_labels*. Raises :class:`DocumentAnswerRejected` otherwise -- including
    when a valid id is mixed with an unknown one, since a partly fabricated
    citation cannot be trusted.
    """
    payload = _load_contract_object(raw_text)

    if set(payload) != _CONTRACT_KEYS:
        raise DocumentAnswerRejected(RejectionReason.INVALID_SCHEMA)

    answerable = payload[_ANSWERABLE_KEY]
    answer = payload[_ANSWER_KEY]
    source_ids = payload[_SOURCE_IDS_KEY]

    if type(answerable) is not bool:
        raise DocumentAnswerRejected(RejectionReason.INVALID_SCHEMA)
    if answer is not None and not isinstance(answer, str):
        raise DocumentAnswerRejected(RejectionReason.INVALID_SCHEMA)
    if not isinstance(source_ids, list) or not all(isinstance(item, str) for item in source_ids):
        raise DocumentAnswerRejected(RejectionReason.INVALID_SCHEMA)

    if not answerable:
        # The only well-formed negative reply is exactly {answer: null, source_ids: []};
        # a "no" that still carries an answer or sources is self-contradictory.
        if answer is None and not source_ids:
            raise DocumentAnswerRejected(RejectionReason.NOT_ANSWERABLE)
        raise DocumentAnswerRejected(RejectionReason.INVALID_SCHEMA)

    if answer is None or not answer.strip():
        raise DocumentAnswerRejected(RejectionReason.EMPTY_ANSWER)
    if not source_ids:
        raise DocumentAnswerRejected(RejectionReason.NO_SOURCES)

    allowed = frozenset(valid_labels)
    if any(source_id not in allowed for source_id in source_ids):
        raise DocumentAnswerRejected(RejectionReason.UNKNOWN_SOURCE)

    distinct_labels = tuple(dict.fromkeys(source_ids))
    return ParsedDocumentAnswer(answer=answer.strip(), source_labels=distinct_labels)


def _load_contract_object(raw_text: str) -> dict[str, Any]:
    if not isinstance(raw_text, str):
        raise DocumentAnswerRejected(RejectionReason.INVALID_FORMAT)

    text = raw_text.strip()
    fenced = _CODE_FENCE_PATTERN.match(text)
    if fenced is not None:
        text = fenced.group("body").strip()

    try:
        payload = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except (ValueError, RecursionError) as exc:
        raise DocumentAnswerRejected(RejectionReason.INVALID_FORMAT) from exc

    if not isinstance(payload, dict):
        raise DocumentAnswerRejected(RejectionReason.INVALID_FORMAT)
    return payload


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate key in generator reply")
        result[key] = value
    return result


def _reject_constant(name: str) -> Any:
    # json.loads accepts NaN/Infinity by default; they are not valid JSON.
    raise ValueError(f"invalid JSON constant: {name}")
