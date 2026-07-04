"""Pure, deterministic memory deduplication policy.

``MemoryPolicy`` contains no I/O: it never talks to Pinecone or OpenAI,
never reads application ``Settings``, and never creates external clients
or mutates external state. It only transforms text and makes
duplicate-detection decisions from the inputs it is given.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata

# Mirrors Settings.MEMORY_NAMESPACE_PREFIX's validator in config.py. Duplicated
# here (rather than imported) so this module has zero dependency on Settings.
_NAMESPACE_PREFIX_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")

_WHITESPACE_PATTERN = re.compile(r"\s+")

# Small, fixed list of standalone RU/EN negation tokens and phrases. Matched at
# word boundaries so substrings inside unrelated words never count (e.g. the
# "не" in "недоделал" or the "not" in "nothing" do not match).
_NEGATION_PATTERN = re.compile(
    r"\b(?:не|нет|никогда|больше\s+не|not|no|never|do\s+not|don't)\b",
    re.IGNORECASE,
)


class MemoryPolicy:
    """Deterministic text normalization, hashing, and duplicate-detection rules."""

    def __init__(self, similarity_threshold: float) -> None:
        if not (0.0 <= similarity_threshold <= 1.0):
            raise ValueError("similarity_threshold must be between 0 and 1")
        self.similarity_threshold = similarity_threshold

    def normalize_text(self, text: str) -> str:
        """Normalize *text* for hashing and duplicate comparison.

        This does not mutate or replace the original text stored as a
        memory -- callers keep the raw user text and only use this
        normalized form for hashing and duplicate detection.
        """
        if not text.strip():
            raise ValueError("text must not be empty or whitespace-only")

        normalized = unicodedata.normalize("NFKC", text).strip()
        normalized = _WHITESPACE_PATTERN.sub(" ", normalized)
        return normalized.casefold()

    def content_hash(self, text: str) -> str:
        """Deterministic, unsalted SHA-256 hash of the normalized text.

        Deliberately excludes the user ID: users are already isolated by
        Pinecone namespace, so the hash only needs to identify the content.
        """
        normalized = self.normalize_text(text)
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    def memory_id_for_text(self, text: str) -> str:
        """Deterministic memory ID used for exact-duplicate detection."""
        return f"mem-{self.content_hash(text)}"

    def namespace_for_user(self, *, prefix: str, user_id: int) -> str:
        """Build the per-user Pinecone namespace.

        Never derived from username, first name, last name, or message
        text -- only the configured prefix and the numeric Telegram user ID.
        """
        if user_id <= 0:
            raise ValueError("user_id must be positive")
        if not _NAMESPACE_PREFIX_PATTERN.fullmatch(prefix):
            raise ValueError(
                "prefix must be non-empty and contain only letters, digits, "
                "hyphens, and underscores"
            )
        return f"{prefix}-{user_id}"

    def has_explicit_negation(self, text: str) -> bool:
        """Detect a standalone RU/EN negation token or phrase in *text*.

        This is a small, deliberately limited heuristic, not a full
        contradiction detector: it only recognizes a fixed list of common
        negation words/phrases at word boundaries (RU: "не", "нет",
        "никогда", "больше не"; EN: "not", "no", "never", "don't",
        "do not"). It will miss implicit or indirect negation such as
        "I changed my mind" or "that's no longer true", and it does not
        reason about sentence structure or grammatical scope.
        """
        normalized = unicodedata.normalize("NFKC", text)
        return _NEGATION_PATTERN.search(normalized) is not None

    def is_semantic_duplicate(
        self,
        *,
        new_text: str,
        existing_text: str,
        similarity_score: float,
    ) -> bool:
        """Decide whether a high-similarity candidate should be treated as a duplicate.

        Returns False below the configured threshold, and False whenever
        exactly one of the two texts contains an explicit negation (so a
        preference and its later negation are never conflated even at a
        high similarity score). Never triggers an automatic update -- the
        caller is expected to skip insertion rather than overwrite.
        """
        if not (-1.0 <= similarity_score <= 1.0):
            raise ValueError("similarity_score must be between -1 and 1")

        if similarity_score < self.similarity_threshold:
            return False

        if self.has_explicit_negation(new_text) != self.has_explicit_negation(existing_text):
            return False

        return True
