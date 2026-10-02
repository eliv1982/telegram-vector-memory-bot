"""Offline tests for the machine-readable document-answer contract parser."""

from __future__ import annotations

import json
from typing import Any

import pytest
from hay_v2_bot.components import (
    DOCUMENT_LABEL_PREFIX,
    DocumentAnswerRejected,
    ParsedDocumentAnswer,
    RejectionReason,
    document_label,
    parse_document_answer,
)
from hay_v2_bot.models import INSUFFICIENT_DOCUMENT_ANSWER

LABELS = ("DOC_1", "DOC_2", "DOC_3")
ANSWER = "Утверждённый бюджет составляет 4,2 млн евро."

_NOT_ANSWERABLE_JSON = '{"answerable": false, "answer": null, "source_ids": []}'


def _raw(**overrides: Any) -> str:
    payload: dict[str, Any] = {"answerable": True, "answer": ANSWER, "source_ids": ["DOC_1"]}
    payload.update(overrides)
    return json.dumps(payload, ensure_ascii=False)


def _raw_without(key: str) -> str:
    payload: dict[str, Any] = {"answerable": True, "answer": ANSWER, "source_ids": ["DOC_1"]}
    del payload[key]
    return json.dumps(payload, ensure_ascii=False)


def _rejection(raw: Any, labels: Any = LABELS) -> RejectionReason:
    with pytest.raises(DocumentAnswerRejected) as exc_info:
        parse_document_answer(raw, labels)
    return exc_info.value.reason


# ---------------------------------------------------------------------------
# Accepted replies
# ---------------------------------------------------------------------------


def test_valid_reply_is_parsed() -> None:
    parsed = parse_document_answer(_raw(), LABELS)

    assert parsed == ParsedDocumentAnswer(answer=ANSWER, source_labels=("DOC_1",))


@pytest.mark.parametrize("fence", ["```json", "```JSON", "```"])
def test_single_surrounding_code_fence_is_tolerated(fence: str) -> None:
    parsed = parse_document_answer(f"{fence}\n{_raw()}\n```", LABELS)

    assert parsed.answer == ANSWER
    assert parsed.source_labels == ("DOC_1",)


def test_surrounding_whitespace_is_tolerated() -> None:
    parsed = parse_document_answer(f"\n\n  {_raw()}  \n", LABELS)

    assert parsed.source_labels == ("DOC_1",)


def test_answer_text_is_trimmed() -> None:
    parsed = parse_document_answer(_raw(answer=f"  {ANSWER}\n"), LABELS)

    assert parsed.answer == ANSWER


def test_source_labels_are_distinct_and_keep_first_seen_order() -> None:
    parsed = parse_document_answer(_raw(source_ids=["DOC_3", "DOC_1", "DOC_3"]), LABELS)

    assert parsed.source_labels == ("DOC_3", "DOC_1")


def test_valid_labels_may_be_any_collection_of_strings() -> None:
    assert parse_document_answer(_raw(), {"DOC_1": object()}).source_labels == ("DOC_1",)
    assert parse_document_answer(_raw(), frozenset({"DOC_1"})).source_labels == ("DOC_1",)


# ---------------------------------------------------------------------------
# Legitimate "not answerable"
# ---------------------------------------------------------------------------


def test_explicit_not_answerable_is_the_legitimate_fallback_signal() -> None:
    assert _rejection(_NOT_ANSWERABLE_JSON) is RejectionReason.NOT_ANSWERABLE
    assert _rejection(f"```json\n{_NOT_ANSWERABLE_JSON}\n```") is RejectionReason.NOT_ANSWERABLE


def test_not_answerable_even_when_the_supplied_labels_are_empty() -> None:
    assert _rejection(_NOT_ANSWERABLE_JSON, labels=()) is RejectionReason.NOT_ANSWERABLE


# ---------------------------------------------------------------------------
# Fail closed: malformed machine output
# ---------------------------------------------------------------------------

_INVALID_FORMAT_REPLIES = [
    pytest.param("", id="empty"),
    pytest.param("   \n ", id="blank"),
    pytest.param("not json", id="prose"),
    pytest.param("Бюджет составляет 4,2 млн евро.", id="plain-answer-prose"),
    pytest.param("```", id="bare-fence"),
    pytest.param("```json\n```", id="empty-fence"),
    pytest.param(f"```json\n{_raw()}\n``` thanks", id="text-after-fence"),
    pytest.param(f"Here you go: {_raw()}", id="text-before-object"),
    pytest.param(f"{_raw()} Hope this helps.", id="text-after-object"),
    pytest.param(f"{_raw()}{_raw()}", id="two-objects"),
    pytest.param('{"answerable": true', id="truncated"),
    pytest.param("{'answerable': true}", id="single-quotes"),
    pytest.param("[]", id="array"),
    pytest.param('"text"', id="json-string"),
    pytest.param("null", id="json-null"),
    pytest.param("123", id="json-number"),
    pytest.param("true", id="json-bool"),
    pytest.param(
        '{"answerable": true, "answerable": false, "answer": null, "source_ids": []}',
        id="duplicate-key",
    ),
    pytest.param('{"answerable": NaN, "answer": null, "source_ids": []}', id="nan-constant"),
    pytest.param("[" * 200_000, id="pathologically-nested"),
]


@pytest.mark.parametrize("raw", _INVALID_FORMAT_REPLIES)
def test_malformed_machine_output_is_rejected_as_invalid_format(raw: str) -> None:
    assert _rejection(raw) is RejectionReason.INVALID_FORMAT


@pytest.mark.parametrize("raw", [None, 123, b'{"answerable": false}', ["x"]])
def test_non_string_reply_is_rejected_as_invalid_format(raw: Any) -> None:
    assert _rejection(raw) is RejectionReason.INVALID_FORMAT


_INVALID_SCHEMA_REPLIES = [
    pytest.param("{}", id="empty-object"),
    pytest.param(_raw_without("answerable"), id="missing-answerable"),
    pytest.param(_raw_without("answer"), id="missing-answer"),
    pytest.param(_raw_without("source_ids"), id="missing-source_ids"),
    pytest.param(_raw(reasoning="because"), id="extra-key"),
    pytest.param(_raw(answerable="true"), id="answerable-string"),
    pytest.param(_raw(answerable=1), id="answerable-int"),
    pytest.param(_raw(answerable=None), id="answerable-null"),
    pytest.param(_raw(answerable=[True]), id="answerable-list"),
    pytest.param(_raw(answer=123), id="answer-number"),
    pytest.param(_raw(answer=[ANSWER]), id="answer-list"),
    pytest.param(_raw(answer={"text": ANSWER}), id="answer-object"),
    pytest.param(_raw(source_ids="DOC_1"), id="source_ids-string"),
    pytest.param(_raw(source_ids=None), id="source_ids-null"),
    pytest.param(_raw(source_ids=[1]), id="source_ids-int-item"),
    pytest.param(_raw(source_ids=["DOC_1", None]), id="source_ids-null-item"),
    pytest.param(_raw(source_ids={"0": "DOC_1"}), id="source_ids-object"),
    pytest.param(_raw(answerable=False, answer=ANSWER, source_ids=[]), id="no-with-answer"),
    pytest.param(_raw(answerable=False, answer=None, source_ids=["DOC_1"]), id="no-with-sources"),
    pytest.param(_raw(answerable=False, answer="", source_ids=[]), id="no-with-empty-string"),
]


@pytest.mark.parametrize("raw", _INVALID_SCHEMA_REPLIES)
def test_wrong_or_contradictory_fields_are_rejected_as_invalid_schema(raw: str) -> None:
    assert _rejection(raw) is RejectionReason.INVALID_SCHEMA


# ---------------------------------------------------------------------------
# Fail closed: answerable=true without evidence
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("answer", ["", "   ", "\n\t "])
def test_answerable_with_blank_answer_is_rejected(answer: str) -> None:
    assert _rejection(_raw(answer=answer)) is RejectionReason.EMPTY_ANSWER


def test_answerable_with_null_answer_is_rejected() -> None:
    assert _rejection(_raw(answer=None)) is RejectionReason.EMPTY_ANSWER


def test_answerable_without_any_source_is_rejected() -> None:
    assert _rejection(_raw(source_ids=[])) is RejectionReason.NO_SOURCES


@pytest.mark.parametrize(
    "source_ids",
    [
        pytest.param(["DOC_9"], id="unknown"),
        pytest.param(["DOC_4"], id="one-past-the-end"),
        pytest.param(["DOC_0"], id="zero"),
        pytest.param(["doc_1"], id="wrong-case"),
        pytest.param([" DOC_1"], id="leading-space"),
        pytest.param(["DOC_01"], id="zero-padded"),
        pytest.param(["doc-1"], id="storage-style-id"),
        pytest.param([""], id="empty-string"),
        pytest.param(["DOC_1", "DOC_9"], id="valid-mixed-with-fabricated"),
        pytest.param(["DOC_9", "DOC_1"], id="fabricated-first"),
    ],
)
def test_any_unknown_source_id_rejects_the_whole_answer(source_ids: list[str]) -> None:
    assert _rejection(_raw(source_ids=source_ids)) is RejectionReason.UNKNOWN_SOURCE


def test_labels_are_checked_against_the_supplied_collection_not_a_fixed_pattern() -> None:
    assert _rejection(_raw(source_ids=["DOC_3"]), labels=("DOC_1", "DOC_2")) is (
        RejectionReason.UNKNOWN_SOURCE
    )


# ---------------------------------------------------------------------------
# The old natural-language sentinel has no control-flow authority
# ---------------------------------------------------------------------------

_SENTINEL_PROSE_VARIANTS = [
    pytest.param(INSUFFICIENT_DOCUMENT_ANSWER, id="exact"),
    pytest.param(f'"{INSUFFICIENT_DOCUMENT_ANSWER}"', id="quoted"),
    pytest.param(f"«{INSUFFICIENT_DOCUMENT_ANSWER}»", id="guillemets"),
    pytest.param(INSUFFICIENT_DOCUMENT_ANSWER + "!", id="plus-punctuation"),
    pytest.param(INSUFFICIENT_DOCUMENT_ANSWER.rstrip("."), id="no-final-period"),
    pytest.param(INSUFFICIENT_DOCUMENT_ANSWER + " Попробуйте уточнить вопрос.", id="plus-sentence"),
    pytest.param("Увы. " + INSUFFICIENT_DOCUMENT_ANSWER, id="sentence-before"),
    pytest.param("В документах нет данных, чтобы ответить на этот вопрос.", id="paraphrase"),
    pytest.param("NO_ANSWER", id="internal-token"),
]


@pytest.mark.parametrize("prose", _SENTINEL_PROSE_VARIANTS)
def test_sentinel_prose_is_just_malformed_output_never_a_document_answer(prose: str) -> None:
    # Whatever the prose says, it is not the machine contract, so it cannot be an answer.
    assert _rejection(prose) is RejectionReason.INVALID_FORMAT


@pytest.mark.parametrize("prose", _SENTINEL_PROSE_VARIANTS)
def test_sentinel_text_inside_the_answer_field_does_not_change_routing(prose: str) -> None:
    # Routing follows the machine fields only: an answerable reply with valid sources is a
    # document answer even if its prose happens to equal the old sentinel...
    parsed = parse_document_answer(_raw(answer=prose), LABELS)
    assert parsed.answer == prose.strip()
    # ...and an explicit "not answerable" falls back regardless of what prose is around.
    assert _rejection(_NOT_ANSWERABLE_JSON) is RejectionReason.NOT_ANSWERABLE


# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------


def test_document_label_is_one_based_and_prefixed() -> None:
    assert DOCUMENT_LABEL_PREFIX == "DOC_"
    assert [document_label(position) for position in (1, 2, 10)] == ["DOC_1", "DOC_2", "DOC_10"]


@pytest.mark.parametrize("position", [0, -1, True, 1.0, "1", None])
def test_document_label_rejects_non_positive_or_non_integer_positions(position: Any) -> None:
    with pytest.raises(ValueError):
        document_label(position)


def test_rejection_exposes_only_a_fixed_reason_code() -> None:
    with pytest.raises(DocumentAnswerRejected) as exc_info:
        parse_document_answer('{"answerable": true, "secret": "TOP-SECRET-USER-DATA"}', LABELS)

    assert str(exc_info.value) == RejectionReason.INVALID_SCHEMA.value
    assert "TOP-SECRET-USER-DATA" not in str(exc_info.value)
