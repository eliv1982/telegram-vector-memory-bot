"""Unit tests for telegram_vector_memory_bot.memory_policy.

MemoryPolicy is pure and deterministic: these tests never touch Pinecone,
OpenAI, Settings, or the network.
"""

from __future__ import annotations

import hashlib

import pytest

from telegram_vector_memory_bot.memory_policy import MemoryPolicy

SHORT_ANSWERS_RU = "Я предпочитаю короткие ответы."
SHORT_ANSWERS_RU_ALT = "  Я   предпочитаю короткие ответы. "
BRIEF_RU = "Пиши мне кратко и по существу."
NO_MORE_SHORT_RU = "Я больше не хочу коротких ответов."


@pytest.fixture
def policy() -> MemoryPolicy:
    return MemoryPolicy(0.90)


# ---------------------------------------------------------------------------
# Constructor
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("threshold", [0.0, 0.5, 1.0])
def test_valid_threshold_accepted(threshold: float) -> None:
    policy = MemoryPolicy(threshold)

    assert policy.similarity_threshold == threshold


@pytest.mark.parametrize("threshold", [-0.01, 1.01])
def test_invalid_threshold_rejected(threshold: float) -> None:
    with pytest.raises(ValueError):
        MemoryPolicy(threshold)


# ---------------------------------------------------------------------------
# normalize_text
# ---------------------------------------------------------------------------


def test_normalize_text_collapses_whitespace_and_casefolds(policy: MemoryPolicy) -> None:
    assert policy.normalize_text(SHORT_ANSWERS_RU_ALT) == "я предпочитаю короткие ответы."


def test_normalize_text_casefold_normalization(policy: MemoryPolicy) -> None:
    assert policy.normalize_text("ПИШИ КРАТКО") == "пиши кратко"


def test_normalize_text_unicode_nfkc_normalization(policy: MemoryPolicy) -> None:
    # U+FB01 LATIN SMALL LIGATURE FI decomposes under NFKC to "fi".
    ligature = "ﬁle"

    assert policy.normalize_text(ligature) == "file"


def test_normalize_text_preserves_punctuation(policy: MemoryPolicy) -> None:
    assert policy.normalize_text(SHORT_ANSWERS_RU).endswith(".")


@pytest.mark.parametrize("text", ["", "   ", "\t\n"])
def test_normalize_text_empty_rejected(policy: MemoryPolicy, text: str) -> None:
    with pytest.raises(ValueError):
        policy.normalize_text(text)


# ---------------------------------------------------------------------------
# content_hash
# ---------------------------------------------------------------------------


def test_content_hash_is_sha256_hex_of_normalized_text(policy: MemoryPolicy) -> None:
    normalized = policy.normalize_text(SHORT_ANSWERS_RU)
    expected = hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    assert policy.content_hash(SHORT_ANSWERS_RU) == expected


def test_content_hash_is_deterministic(policy: MemoryPolicy) -> None:
    assert policy.content_hash(SHORT_ANSWERS_RU) == policy.content_hash(SHORT_ANSWERS_RU)


def test_content_hash_same_for_semantically_equivalent_normalized_inputs(
    policy: MemoryPolicy,
) -> None:
    assert policy.content_hash(SHORT_ANSWERS_RU) == policy.content_hash(SHORT_ANSWERS_RU_ALT)


def test_content_hash_differs_for_different_text(policy: MemoryPolicy) -> None:
    assert policy.content_hash(SHORT_ANSWERS_RU) != policy.content_hash(BRIEF_RU)


# ---------------------------------------------------------------------------
# memory_id_for_text
# ---------------------------------------------------------------------------


def test_memory_id_for_text_is_deterministic_and_prefixed(policy: MemoryPolicy) -> None:
    memory_id = policy.memory_id_for_text(SHORT_ANSWERS_RU)

    assert memory_id == f"mem-{policy.content_hash(SHORT_ANSWERS_RU)}"
    assert policy.memory_id_for_text(SHORT_ANSWERS_RU) == memory_id


# ---------------------------------------------------------------------------
# namespace_for_user
# ---------------------------------------------------------------------------


def test_namespace_for_user_builds_expected_namespace(policy: MemoryPolicy) -> None:
    assert policy.namespace_for_user(prefix="telegram-user", user_id=42) == "telegram-user-42"


@pytest.mark.parametrize("user_id", [0, -1])
def test_namespace_for_user_invalid_user_id_rejected(policy: MemoryPolicy, user_id: int) -> None:
    with pytest.raises(ValueError):
        policy.namespace_for_user(prefix="telegram-user", user_id=user_id)


@pytest.mark.parametrize("prefix", ["", "   ", "bad prefix", "bad/prefix"])
def test_namespace_for_user_invalid_prefix_rejected(policy: MemoryPolicy, prefix: str) -> None:
    with pytest.raises(ValueError):
        policy.namespace_for_user(prefix=prefix, user_id=42)


# ---------------------------------------------------------------------------
# has_explicit_negation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "Я больше не хочу коротких ответов.",
        "Никогда не соглашайся на это.",
        "Нет, мне это не нужно.",
    ],
)
def test_russian_standalone_negation_detected(policy: MemoryPolicy, text: str) -> None:
    assert policy.has_explicit_negation(text) is True


@pytest.mark.parametrize(
    "text",
    [
        "I do not want short answers.",
        "I don't want short answers.",
        "No, that's not what I meant.",
        "I will never do that again.",
    ],
)
def test_english_standalone_negation_detected(policy: MemoryPolicy, text: str) -> None:
    assert policy.has_explicit_negation(text) is True


@pytest.mark.parametrize(
    "text",
    [
        "I can't stand short answers.",
        "She doesn't like coffee.",
        "He didn't call me back.",
        "I won't be there on Monday.",
        "It isn't ready.",
        "I haven't decided yet.",
        "They aren't coming.",
        "It wasn't me.",
        "I wouldn't do that.",
        "I shouldn't say.",
        "DON'T shout.",
    ],
)
def test_english_nt_contractions_detected(policy: MemoryPolicy, text: str) -> None:
    assert policy.has_explicit_negation(text) is True


@pytest.mark.parametrize(
    "text",
    [
        "I can’t stand short answers.",  # U+2019 right single quotation mark
        "She doesn’t like coffee.",
        "I won‘t be there.",  # U+2018 left single quotation mark
        "I don´t know.",  # U+00B4 acute accent
        "I don`t know.",  # backtick
        "I donʼt know.",  # U+02BC modifier letter apostrophe
        "I don′t know.",  # U+2032 prime
        "I don＇t know.",  # U+FF07 fullwidth apostrophe (folded by NFKC)
        "DON’T shout.",
    ],
)
def test_typographic_apostrophes_are_normalized_before_matching(
    policy: MemoryPolicy, text: str
) -> None:
    assert policy.has_explicit_negation(text) is True


@pytest.mark.parametrize(
    "text",
    [
        "I cannot attend.",
        "You CANNOT do that.",
        "Так нельзя делать.",
        "Нельзя опаздывать на встречи.",
    ],
)
def test_cannot_and_russian_nelzya_detected(policy: MemoryPolicy, text: str) -> None:
    assert policy.has_explicit_negation(text) is True


@pytest.mark.parametrize(
    "text",
    [
        "недоделал проект вовремя",
        "nothing to see here",
        "know your limits",
        SHORT_ANSWERS_RU,
        BRIEF_RU,
        "I live in Paris.",
        "Мы можем встретиться завтра.",
        "It's fine, she's here and they're ready.",  # apostrophes without a negation
        "Beyond the cannon's range.",  # "cannon" must not match "cannot"
        "О'Брайен любит чай.",
    ],
)
def test_negation_substrings_are_not_false_positives(policy: MemoryPolicy, text: str) -> None:
    assert policy.has_explicit_negation(text) is False


def test_nt_contraction_mismatch_blocks_semantic_duplicate(policy: MemoryPolicy) -> None:
    # "can’t" (typographic apostrophe) must register as negation so the guard still fires.
    result = policy.is_semantic_duplicate(
        new_text="I can’t eat peanuts.",
        existing_text="I can eat peanuts.",
        similarity_score=0.99,
    )

    assert result is False


# ---------------------------------------------------------------------------
# is_semantic_duplicate
# ---------------------------------------------------------------------------


def test_score_below_threshold_is_not_duplicate(policy: MemoryPolicy) -> None:
    result = policy.is_semantic_duplicate(
        new_text=SHORT_ANSWERS_RU,
        existing_text=BRIEF_RU,
        similarity_score=0.89,
    )

    assert result is False


def test_score_at_threshold_is_duplicate(policy: MemoryPolicy) -> None:
    result = policy.is_semantic_duplicate(
        new_text=SHORT_ANSWERS_RU,
        existing_text=BRIEF_RU,
        similarity_score=0.90,
    )

    assert result is True


def test_score_above_threshold_is_duplicate(policy: MemoryPolicy) -> None:
    result = policy.is_semantic_duplicate(
        new_text=SHORT_ANSWERS_RU,
        existing_text=BRIEF_RU,
        similarity_score=0.98,
    )

    assert result is True


@pytest.mark.parametrize("score", [-1.01, 1.01])
def test_is_semantic_duplicate_score_out_of_range_rejected(
    policy: MemoryPolicy, score: float
) -> None:
    with pytest.raises(ValueError):
        policy.is_semantic_duplicate(
            new_text=SHORT_ANSWERS_RU,
            existing_text=BRIEF_RU,
            similarity_score=score,
        )


def test_one_negated_one_plain_text_is_not_duplicate_even_at_high_score(
    policy: MemoryPolicy,
) -> None:
    result = policy.is_semantic_duplicate(
        new_text=NO_MORE_SHORT_RU,
        existing_text=SHORT_ANSWERS_RU,
        similarity_score=0.99,
    )

    assert result is False


def test_both_negated_texts_may_still_be_duplicates_at_high_score(policy: MemoryPolicy) -> None:
    existing_negated = "Я больше не хочу коротких ответов."
    new_negated = "Больше не хочу, чтобы ты отвечал коротко."

    result = policy.is_semantic_duplicate(
        new_text=new_negated,
        existing_text=existing_negated,
        similarity_score=0.99,
    )

    assert result is True
