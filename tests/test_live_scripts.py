"""Unit tests for scripts/calibrate_similarity.py and scripts/smoke_test_memory.py.

Both scripts make real network calls when run directly, but neither creates
any client at import time. These tests load each script with ``runpy`` (which
executes top-level code but never triggers the ``if __name__ == "__main__"``
guard), then monkeypatch/replace the script's ``get_settings`` and
``PineconeManager`` references with fully offline test doubles. No test
performs a real network request or reads the user's real ``.env`` file.
"""

from __future__ import annotations

import json
import logging
import math
import runpy
import time
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from telegram_vector_memory_bot.config import Settings
from telegram_vector_memory_bot.memory_policy import MemoryPolicy
from telegram_vector_memory_bot.memory_service import MemoryService
from telegram_vector_memory_bot.models import IndexInfo, VectorMatch
from telegram_vector_memory_bot.pinecone_manager import VectorStorageError

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
DIMENSION = 4


def _load_script(name: str) -> dict[str, Any]:
    """Execute a script's top-level code without triggering its __main__ guard.

    Returns the script's *actual* globals dict (via an already-defined
    function's ``__globals__``) rather than ``runpy.run_path``'s returned
    copy, so that reassigning e.g. ``ns["get_settings"]`` is visible to
    ``main()`` when it looks up that name at call time.
    """
    namespace = runpy.run_path(str(SCRIPTS_DIR / name))
    return namespace["main"].__globals__


def _build_settings(**overrides: Any) -> Settings:
    data: dict[str, Any] = {
        "PINECONE_API_KEY": "test-pinecone-key",
        "PINECONE_INDEX_NAME": "test-index",
        "OPENAI_API_KEY": "test-openai-key",
        "OPENAI_CHAT_MODEL": "gpt-4o-mini",
        "TELEGRAM_BOT_TOKEN": "test-telegram-token",
    }
    data.update(overrides)
    return Settings(_env_file=None, **data)


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _matches_filter(metadata: dict[str, Any], metadata_filter: dict[str, Any] | None) -> bool:
    if metadata_filter is None:
        return True
    for field, condition in metadata_filter.items():
        if isinstance(condition, dict) and "$eq" in condition:
            if metadata.get(field) != condition["$eq"]:
                return False
        elif metadata.get(field) != condition:
            return False
    return True


class InMemoryFakeManager:
    """Tiny in-memory stand-in for PineconeManager, used only in these tests.

    Real cosine similarity is computed over configurable per-text fake
    embeddings, so the scripts' real orchestration logic (and, for the smoke
    test, the real MemoryService/MemoryPolicy) can be exercised end-to-end
    without any network access.
    """

    def __init__(self, embeddings: dict[str, list[float]]) -> None:
        self._embeddings = embeddings
        self._store: dict[str, dict[str, dict[str, Any]]] = {}
        self.create_embedding_calls: list[str] = []
        self.upsert_calls: list[dict[str, Any]] = []
        self.query_calls: list[dict[str, Any]] = []
        self.fetch_calls: list[dict[str, Any]] = []
        self.delete_calls: list[str] = []
        self.index_info = IndexInfo(
            name="test-index",
            host="test-index.svc.pinecone.io",
            dimension=DIMENSION,
            metric="cosine",
            ready=True,
            state="Ready",
        )

    def seed(
        self, *, namespace: str, vector_id: str, values: list[float], metadata: dict[str, Any]
    ) -> None:
        """Pre-populate stored data, e.g. to simulate a previous interrupted run."""
        self._store.setdefault(namespace, {})[vector_id] = {
            "values": list(values),
            "metadata": dict(metadata),
        }

    def create_embedding(self, text: str) -> list[float]:
        self.create_embedding_calls.append(text)
        if text not in self._embeddings:
            raise AssertionError(f"no fake embedding configured for text: {text!r}")
        return list(self._embeddings[text])

    def upsert_vector(
        self,
        *,
        vector_id: str,
        values: list[float],
        metadata: dict[str, Any],
        namespace: str,
    ) -> None:
        self.upsert_calls.append(
            {"vector_id": vector_id, "namespace": namespace, "metadata": dict(metadata)}
        )
        self._store.setdefault(namespace, {})[vector_id] = {
            "values": list(values),
            "metadata": dict(metadata),
        }

    def query_by_vector(
        self,
        *,
        values: list[float],
        namespace: str,
        top_k: int,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[VectorMatch]:
        self.query_calls.append(
            {"namespace": namespace, "top_k": top_k, "metadata_filter": metadata_filter}
        )
        candidates = []
        for vector_id, entry in self._store.get(namespace, {}).items():
            if not _matches_filter(entry["metadata"], metadata_filter):
                continue
            score = _cosine(values, entry["values"])
            candidates.append(
                VectorMatch(vector_id=vector_id, score=score, metadata=dict(entry["metadata"]))
            )
        candidates.sort(key=lambda m: m.score, reverse=True)
        return candidates[:top_k]

    def query_by_text(
        self,
        *,
        text: str,
        namespace: str,
        top_k: int,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[VectorMatch]:
        values = self.create_embedding(text)
        return self.query_by_vector(
            values=values, namespace=namespace, top_k=top_k, metadata_filter=metadata_filter
        )

    def fetch_vectors(
        self, *, vector_ids: list[str], namespace: str
    ) -> dict[str, dict[str, Any]]:
        self.fetch_calls.append({"vector_ids": list(vector_ids), "namespace": namespace})
        bucket = self._store.get(namespace, {})
        return {vid: bucket[vid] for vid in vector_ids if vid in bucket}

    def delete_namespace(self, namespace: str) -> None:
        self.delete_calls.append(namespace)
        self._store.pop(namespace, None)

    def __getattr__(self, name: str) -> Any:
        if name == "delete_index":
            raise AssertionError("scripts must never call PineconeManager.delete_index")
        raise AttributeError(name)


class AlwaysEmptyQueryManager(InMemoryFakeManager):
    """Forces every query to return no matches, to exercise failure/cleanup paths."""

    def query_by_vector(
        self,
        *,
        values: list[float],
        namespace: str,
        top_k: int,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[VectorMatch]:
        self.query_calls.append(
            {"namespace": namespace, "top_k": top_k, "metadata_filter": metadata_filter}
        )
        return []


class DelayedVisibilityManager(InMemoryFakeManager):
    """Forces the first *empty_responses* ``query_by_vector`` calls to return zero
    matches, then defers to the real in-memory store. Used to simulate Pinecone's
    eventual consistency: an upsert is acknowledged but not immediately visible to
    a subsequent filtered query.
    """

    def __init__(self, embeddings: dict[str, list[float]], *, empty_responses: int) -> None:
        super().__init__(embeddings)
        self._empty_responses_remaining = empty_responses

    def query_by_vector(
        self,
        *,
        values: list[float],
        namespace: str,
        top_k: int,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[VectorMatch]:
        if self._empty_responses_remaining > 0:
            self._empty_responses_remaining -= 1
            self.query_calls.append(
                {"namespace": namespace, "top_k": top_k, "metadata_filter": metadata_filter}
            )
            return []
        return super().query_by_vector(
            values=values, namespace=namespace, top_k=top_k, metadata_filter=metadata_filter
        )


class DuplicateMatchQueryManager(InMemoryFakeManager):
    """Always returns two matches, to exercise the malformed-calibration-state guard."""

    def query_by_vector(
        self,
        *,
        values: list[float],
        namespace: str,
        top_k: int,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[VectorMatch]:
        self.query_calls.append(
            {"namespace": namespace, "top_k": top_k, "metadata_filter": metadata_filter}
        )
        return [
            VectorMatch(vector_id="dup-1", score=0.9, metadata={}),
            VectorMatch(vector_id="dup-2", score=0.8, metadata={}),
        ]


class DelayedFetchVisibilityManager(InMemoryFakeManager):
    """Forces the first *missing_fetches* ``fetch_vectors`` calls to report a miss
    (as if the id were not yet visible), then defers to the real in-memory store.
    """

    def __init__(self, embeddings: dict[str, list[float]], *, missing_fetches: int) -> None:
        super().__init__(embeddings)
        self._missing_fetches_remaining = missing_fetches

    def fetch_vectors(
        self, *, vector_ids: list[str], namespace: str
    ) -> dict[str, dict[str, Any]]:
        if self._missing_fetches_remaining > 0:
            self._missing_fetches_remaining -= 1
            self.fetch_calls.append({"vector_ids": list(vector_ids), "namespace": namespace})
            return {}
        return super().fetch_vectors(vector_ids=vector_ids, namespace=namespace)


class RecallVisibilityManager(InMemoryFakeManager):
    """Test double controlling only recall()-shaped ``query_by_vector`` calls
    (``top_k != 1``), which distinguishes them from ``remember()``'s ``top_k=1``
    duplicate lookups. The first *skip_recall_calls* recall-shaped calls behave
    normally; the next *forced_responses* are overridden by *forced_result*; all
    calls after that defer to the real in-memory store.
    """

    def __init__(
        self,
        embeddings: dict[str, list[float]],
        *,
        skip_recall_calls: int = 0,
        forced_responses: int = 0,
        forced_result: Callable[[], list[VectorMatch]] | None = None,
    ) -> None:
        super().__init__(embeddings)
        self._skip_recall_calls_remaining = skip_recall_calls
        self._forced_responses_remaining = forced_responses
        self._forced_result = forced_result or (lambda: [])

    def query_by_vector(
        self,
        *,
        values: list[float],
        namespace: str,
        top_k: int,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[VectorMatch]:
        if top_k != 1:
            if self._skip_recall_calls_remaining > 0:
                self._skip_recall_calls_remaining -= 1
            elif self._forced_responses_remaining > 0:
                self._forced_responses_remaining -= 1
                self.query_calls.append(
                    {"namespace": namespace, "top_k": top_k, "metadata_filter": metadata_filter}
                )
                return self._forced_result()
        return super().query_by_vector(
            values=values, namespace=namespace, top_k=top_k, metadata_filter=metadata_filter
        )


class RaisingDeleteManager(InMemoryFakeManager):
    """Delete-namespace double mirroring PineconeManager's idempotent contract.

    Deleting an already-absent namespace always succeeds silently here, just
    as the real ``PineconeManager.delete_namespace`` treats a Pinecone
    not-found response as a successful no-op. A configured failure instead
    raises ``VectorStorageError`` -- the same exception type the real manager
    raises for genuine (non-404) infrastructure errors -- so these tests
    exercise the scripts' and MemoryService's behavior at that seam without
    re-implementing any Pinecone-specific exception inspection here.
    """

    def __init__(
        self,
        embeddings: dict[str, list[float]],
        *,
        fail_on_call_number: int | None = None,
        failure: Exception | None = None,
    ) -> None:
        super().__init__(embeddings)
        self._fail_on_call_number = fail_on_call_number
        self._failure = failure or VectorStorageError("Namespace deletion failed")
        self._delete_namespace_calls = 0

    def delete_namespace(self, namespace: str) -> None:
        self._delete_namespace_calls += 1
        if self._delete_namespace_calls == self._fail_on_call_number:
            raise self._failure
        super().delete_namespace(namespace)


def _stale_match() -> VectorMatch:
    """A syntactically valid but logically stale recall match for delete-consistency tests."""
    return VectorMatch(
        vector_id="stale-memory-id",
        score=0.5,
        metadata={
            "text": "stale memory from before deletion",
            "content_hash": "stale-hash",
            "created_at": "2020-01-01T00:00:00+00:00",
            "source": "telegram",
            "record_type": "user_memory",
        },
    )


def _irrelevant_match() -> VectorMatch:
    """The exact false positive the live smoke test hit: an older, unrelated
    memory that is already query-visible and non-empty, but is not the newly
    inserted training memory recall acceptance actually requires."""
    return VectorMatch(
        vector_id="irrelevant-older-memory-id",
        score=0.55,
        metadata={
            "text": "Я предпочитаю короткие ответы без лишних подробностей.",
            "content_hash": "irrelevant-hash",
            "created_at": "2020-01-01T00:00:00+00:00",
            "source": "telegram",
            "record_type": "user_memory",
        },
    )


def _patch_time(monkeypatch: pytest.MonkeyPatch, *, step: float = 0.01) -> list[float]:
    """Monkeypatch ``time.monotonic``/``time.sleep`` for instant, deterministic polling.

    ``time.monotonic()`` advances by *step* on every call; ``time.sleep()`` never
    actually sleeps but records its argument in the returned list.
    """
    state = {"value": 0.0}

    def _fake_monotonic() -> float:
        value = state["value"]
        state["value"] += step
        return value

    sleep_calls: list[float] = []

    monkeypatch.setattr(time, "monotonic", _fake_monotonic)
    monkeypatch.setattr(time, "sleep", sleep_calls.append)
    return sleep_calls


def _calibration_embeddings(pairs: list[dict[str, str]]) -> dict[str, list[float]]:
    embeddings: dict[str, list[float]] = {}
    for i, pair in enumerate(pairs):
        vector = [0.0] * DIMENSION
        vector[i % DIMENSION] = 1.0
        embeddings[pair["reference"]] = list(vector)
        embeddings[pair["candidate"]] = list(vector)
    return embeddings


def _default_smoke_embeddings(ns: dict[str, Any]) -> dict[str, list[float]]:
    return {
        ns["SHORT_ANSWERS_TEXT"]: [1.0, 0.0, 0.0, 0.0],
        ns["PARAPHRASE_TEXT"]: [0.0, 1.0, 0.0, 0.0],
        ns["DIFFERENT_MEMORY_TEXT"]: [0.0, 0.0, 1.0, 0.0],
        ns["RECALL_QUERY_TEXT"]: [0.0, 0.0, 0.9, 0.1],
    }


# ---------------------------------------------------------------------------
# Import safety
# ---------------------------------------------------------------------------


def test_calibrate_script_creates_no_clients_on_import(monkeypatch: pytest.MonkeyPatch) -> None:
    import openai
    import pinecone

    def _explode(self: Any, *args: Any, **kwargs: Any) -> None:
        raise AssertionError("client constructor must not be called at import time")

    monkeypatch.setattr(pinecone.Pinecone, "__init__", _explode)
    monkeypatch.setattr(openai.OpenAI, "__init__", _explode)

    ns = _load_script("calibrate_similarity.py")

    assert callable(ns["main"])


def test_smoke_test_script_creates_no_clients_on_import(monkeypatch: pytest.MonkeyPatch) -> None:
    import openai
    import pinecone

    def _explode(self: Any, *args: Any, **kwargs: Any) -> None:
        raise AssertionError("client constructor must not be called at import time")

    monkeypatch.setattr(pinecone.Pinecone, "__init__", _explode)
    monkeypatch.setattr(openai.OpenAI, "__init__", _explode)

    ns = _load_script("smoke_test_memory.py")

    assert callable(ns["main"])


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def test_calibrate_parser_accepts_valid_arguments() -> None:
    ns = _load_script("calibrate_similarity.py")
    parser = ns["build_arg_parser"]()

    args = parser.parse_args([])

    assert args is not None


def test_smoke_test_parser_accepts_valid_arguments() -> None:
    ns = _load_script("smoke_test_memory.py")
    parser = ns["build_arg_parser"]()

    default_args = parser.parse_args([])
    assert default_args.user_id == ns["DEFAULT_SYNTHETIC_USER_ID"]
    assert default_args.require_semantic_skip is False

    custom_args = parser.parse_args(["--user-id", "900000042", "--require-semantic-skip"])
    assert custom_args.user_id == 900000042
    assert custom_args.require_semantic_skip is True


@pytest.mark.parametrize("user_id", [0, -1, 123456, 12345678901])
def test_smoke_test_validate_user_id_rejects_invalid(user_id: int) -> None:
    ns = _load_script("smoke_test_memory.py")

    with pytest.raises(ns["SmokeTestError"]):
        ns["validate_user_id"](user_id)


def test_smoke_test_validate_user_id_accepts_default() -> None:
    ns = _load_script("smoke_test_memory.py")

    ns["validate_user_id"](ns["DEFAULT_SYNTHETIC_USER_ID"])


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------


def test_calibrate_new_namespace_is_unique() -> None:
    ns = _load_script("calibrate_similarity.py")

    first = ns["new_calibration_namespace"]()
    second = ns["new_calibration_namespace"]()

    assert first != second
    assert first.startswith("calibration-")
    assert second.startswith("calibration-")


def test_calibration_passes_exact_pair_id_metadata_filter() -> None:
    ns = _load_script("calibrate_similarity.py")
    pairs = ns["CALIBRATION_PAIRS"]
    manager = InMemoryFakeManager(_calibration_embeddings(pairs))

    ns["run_calibration"](manager, "text-embedding-3-small")

    assert len(manager.query_calls) == len(pairs)
    for call, pair in zip(manager.query_calls, pairs, strict=True):
        assert call["metadata_filter"] == {"pair_id": {"$eq": pair["pair_id"]}}
        # top_k=2 (not 1) lets a malformed multi-match state be detected instead
        # of silently truncated.
        assert call["top_k"] == 2


def test_calibration_preserves_returned_score() -> None:
    ns = _load_script("calibrate_similarity.py")
    pairs = ns["CALIBRATION_PAIRS"]
    embeddings = _calibration_embeddings(pairs)
    # Override pair A's candidate so the cosine score is a known, non-trivial value.
    embeddings[pairs[0]["candidate"]] = [0.6, 0.8, 0.0, 0.0]
    manager = InMemoryFakeManager(embeddings)

    summary = ns["run_calibration"](manager, "text-embedding-3-small")

    assert summary["pairs"][0]["score"] == pytest.approx(0.6)
    assert summary["embedding_model"] == "text-embedding-3-small"
    assert summary["index_name"] == "test-index"


def _separated_calibration_embeddings(
    pairs: list[dict[str, str]], scores_by_pair_id: dict[str, float]
) -> dict[str, list[float]]:
    """Build fake embeddings giving each pair's candidate an exact,
    independently controllable cosine score against its own reference --
    even though pairs C and D share an identical reference phrase (and
    therefore, realistically, an identical embedding).
    """
    dimension = 4
    axis_by_reference_text: dict[str, int] = {}
    embeddings: dict[str, list[float]] = {}
    for pair in pairs:
        text = pair["reference"]
        if text not in axis_by_reference_text:
            axis = len(axis_by_reference_text)
            axis_by_reference_text[text] = axis
            vector = [0.0] * dimension
            vector[axis] = 1.0
            embeddings[text] = vector

    other_axis = dimension - 1
    for pair in pairs:
        axis = axis_by_reference_text[pair["reference"]]
        score = scores_by_pair_id[pair["pair_id"]]
        vector = [0.0] * dimension
        vector[axis] = score
        vector[other_axis] = math.sqrt(max(0.0, 1.0 - score * score))
        embeddings[pair["candidate"]] = vector
    return embeddings


def test_calibration_interpretation_reports_separation_and_selected_threshold() -> None:
    ns = _load_script("calibrate_similarity.py")
    pairs = ns["CALIBRATION_PAIRS"]
    # Structurally mirrors an observed live run: both positive-labeled pairs
    # (A, B) scored higher than both negative-labeled pairs (C, D).
    scores_by_pair_id = {
        "A_likely_paraphrase": 0.55,
        "B_related_preference": 0.60,
        "C_potential_conflict": 0.30,
        "D_unrelated": 0.20,
    }
    manager = InMemoryFakeManager(_separated_calibration_embeddings(pairs, scores_by_pair_id))

    summary = ns["run_calibration"](
        manager, "text-embedding-3-small", similarity_threshold=0.50
    )
    interpretation = summary["calibration_interpretation"]

    assert interpretation["minimum_positive_score"] == pytest.approx(0.55)
    assert interpretation["maximum_negative_score"] == pytest.approx(0.30)
    assert interpretation["separation_exists"] is True
    assert interpretation["candidate_threshold_interval"] == {
        "greater_than": pytest.approx(0.30),
        "less_than_or_equal_to": pytest.approx(0.55),
    }
    assert interpretation["selected_project_threshold"] == pytest.approx(0.50)


def test_calibration_interpretation_reports_no_separation_without_fabricating() -> None:
    ns = _load_script("calibrate_similarity.py")
    pairs = ns["CALIBRATION_PAIRS"]
    # The lowest positive (A) now scores below the highest negative (C) --
    # no single cosine threshold can separate this (hypothetical future) run.
    scores_by_pair_id = {
        "A_likely_paraphrase": 0.35,
        "B_related_preference": 0.60,
        "C_potential_conflict": 0.45,
        "D_unrelated": 0.20,
    }
    manager = InMemoryFakeManager(_separated_calibration_embeddings(pairs, scores_by_pair_id))

    summary = ns["run_calibration"](manager, "text-embedding-3-small")
    interpretation = summary["calibration_interpretation"]

    assert interpretation["minimum_positive_score"] == pytest.approx(0.35)
    assert interpretation["maximum_negative_score"] == pytest.approx(0.45)
    assert interpretation["separation_exists"] is False
    assert interpretation["candidate_threshold_interval"] is None
    assert "no single" in interpretation["note"].lower()
    # It must still report the currently selected threshold as a plain fact,
    # never as a recommendation derived from this (unseparated) run.
    assert interpretation["selected_project_threshold"] == pytest.approx(
        ns["DEFAULT_SELECTED_THRESHOLD"]
    )


def test_calibration_main_reports_live_settings_threshold_not_a_hardcoded_default() -> None:
    ns = _load_script("calibrate_similarity.py")
    pairs = ns["CALIBRATION_PAIRS"]
    manager = InMemoryFakeManager(_calibration_embeddings(pairs))
    settings = _build_settings(MEMORY_SIMILARITY_THRESHOLD="0.42")

    ns["get_settings"] = lambda: settings
    ns["PineconeManager"] = lambda settings: manager

    captured_summary: dict[str, Any] = {}
    original_print_report = ns["print_report"]

    def _capture_and_print(summary: dict[str, Any]) -> None:
        captured_summary.update(summary)
        original_print_report(summary)

    ns["print_report"] = _capture_and_print

    exit_code = ns["main"]([])

    assert exit_code == 0
    assert captured_summary["calibration_interpretation"][
        "selected_project_threshold"
    ] == pytest.approx(0.42)


def test_calibration_never_touches_env_file(monkeypatch: pytest.MonkeyPatch) -> None:
    """Calibration must never write to (or read a real) ``.env`` automatically."""
    ns = _load_script("calibrate_similarity.py")
    pairs = ns["CALIBRATION_PAIRS"]
    manager = InMemoryFakeManager(_calibration_embeddings(pairs))

    import builtins

    real_open = builtins.open

    def _guarded_open(file: Any, *args: Any, **kwargs: Any) -> Any:
        if isinstance(file, str | Path) and ".env" in str(file):
            raise AssertionError("calibration must never open a .env file")
        return real_open(file, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", _guarded_open)

    summary = ns["run_calibration"](manager, "text-embedding-3-small")

    assert summary["calibration_interpretation"]["selected_project_threshold"] == pytest.approx(
        ns["DEFAULT_SELECTED_THRESHOLD"]
    )


def test_calibration_cleanup_runs_after_success() -> None:
    ns = _load_script("calibrate_similarity.py")
    pairs = ns["CALIBRATION_PAIRS"]
    manager = InMemoryFakeManager(_calibration_embeddings(pairs))

    ns["run_calibration"](manager, "text-embedding-3-small")

    assert len(manager.delete_calls) == 1
    namespace_used = manager.upsert_calls[0]["namespace"]
    assert manager.delete_calls[0] == namespace_used


def test_calibration_cleanup_runs_after_failure() -> None:
    ns = _load_script("calibrate_similarity.py")
    pairs = ns["CALIBRATION_PAIRS"]
    manager = AlwaysEmptyQueryManager(_calibration_embeddings(pairs))

    with pytest.raises(ns["CalibrationError"]):
        ns["run_calibration"](manager, "text-embedding-3-small")

    assert len(manager.delete_calls) == 1


def test_calibrate_never_calls_delete_index() -> None:
    ns = _load_script("calibrate_similarity.py")
    pairs = ns["CALIBRATION_PAIRS"]
    manager = InMemoryFakeManager(_calibration_embeddings(pairs))

    # InMemoryFakeManager raises AssertionError if delete_index is ever
    # accessed; reaching this point without error proves calibration only
    # ever deletes the temporary namespace, never the index.
    summary = ns["run_calibration"](manager, "text-embedding-3-small")

    assert len(summary["pairs"]) == len(pairs)


def test_calibrate_main_returns_nonzero_on_failure() -> None:
    ns = _load_script("calibrate_similarity.py")
    pairs = ns["CALIBRATION_PAIRS"]
    manager = AlwaysEmptyQueryManager(_calibration_embeddings(pairs))
    settings = _build_settings()

    ns["get_settings"] = lambda: settings
    ns["PineconeManager"] = lambda settings: manager

    exit_code = ns["main"]([])

    assert exit_code != 0
    assert len(manager.delete_calls) == 1


def test_calibrate_main_output_never_contains_injected_fake_secret(
    capsys: pytest.CaptureFixture[str],
) -> None:
    ns = _load_script("calibrate_similarity.py")
    pairs = ns["CALIBRATION_PAIRS"]
    fake_secret = "sk-FAKE-INJECTED-SECRET-VALUE"
    manager = InMemoryFakeManager(_calibration_embeddings(pairs))
    settings = _build_settings(OPENAI_API_KEY=fake_secret, PINECONE_API_KEY=fake_secret)

    ns["get_settings"] = lambda: settings
    ns["PineconeManager"] = lambda settings: manager

    exit_code = ns["main"]([])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert fake_secret not in captured.out
    assert fake_secret not in captured.err


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------


def test_smoke_test_cleans_stale_namespace_before_execution() -> None:
    ns = _load_script("smoke_test_memory.py")
    settings = _build_settings()
    manager = InMemoryFakeManager(_default_smoke_embeddings(ns))
    service = MemoryService(manager=manager, settings=settings)

    user_id = 900000001
    namespace = service.namespace_for_user(user_id)
    policy = MemoryPolicy(settings.MEMORY_SIMILARITY_THRESHOLD)
    stale_id = policy.memory_id_for_text(ns["SHORT_ANSWERS_TEXT"])
    manager.seed(
        namespace=namespace,
        vector_id=stale_id,
        values=[1.0, 0.0, 0.0, 0.0],
        metadata={"text": "stale data from a previous run", "record_type": "user_memory"},
    )

    summary = ns["run_smoke_test"](
        service, manager, settings=settings, user_id=user_id, require_semantic_skip=False
    )

    # If pre-cleanup hadn't run, this exact deterministic ID would already
    # exist and the first write would be reported as an exact duplicate.
    assert summary["first_write"]["action"] == "inserted"


def test_smoke_test_initial_stale_cleanup_succeeds_on_absent_namespace_and_continues() -> None:
    """A manager reporting an absent namespace on initial cleanup must not
    fail the smoke test -- it behaves as a successful idempotent no-op, and
    the scenario continues on to the first write."""
    ns = _load_script("smoke_test_memory.py")
    settings = _build_settings()
    manager = RaisingDeleteManager(_default_smoke_embeddings(ns))
    service = MemoryService(manager=manager, settings=settings)

    summary = ns["run_smoke_test"](
        service, manager, settings=settings, user_id=900000030, require_semantic_skip=False
    )

    assert manager._delete_namespace_calls >= 1
    assert summary["first_write"]["action"] == "inserted"
    assert summary["cleanup_verified"] is True


def test_smoke_test_non_404_initial_cleanup_failure_aborts_safely() -> None:
    """A genuine (non-404) infrastructure failure during the initial cleanup
    must propagate rather than being swallowed, and must abort before any
    write is attempted."""
    ns = _load_script("smoke_test_memory.py")
    settings = _build_settings()
    manager = RaisingDeleteManager(_default_smoke_embeddings(ns), fail_on_call_number=1)
    service = MemoryService(manager=manager, settings=settings)

    with pytest.raises(VectorStorageError):
        ns["run_smoke_test"](
            service, manager, settings=settings, user_id=900000031, require_semantic_skip=False
        )

    assert manager.upsert_calls == []


def test_smoke_test_main_reports_non_404_cleanup_failure_and_returns_nonzero() -> None:
    ns = _load_script("smoke_test_memory.py")
    settings = _build_settings()
    manager = RaisingDeleteManager(_default_smoke_embeddings(ns), fail_on_call_number=1)

    ns["get_settings"] = lambda: settings
    ns["PineconeManager"] = lambda settings: manager

    exit_code = ns["main"](["--user-id", "900000032"])

    assert exit_code != 0
    assert manager.upsert_calls == []


def test_smoke_test_final_cleanup_remains_idempotent_on_already_absent_namespace() -> None:
    """The safety-net final cleanup, run after the scenario's own successful
    cleanup has already emptied the namespace, must also succeed idempotently
    rather than treating the now-absent namespace as an error."""
    ns = _load_script("smoke_test_memory.py")
    settings = _build_settings()
    manager = RaisingDeleteManager(_default_smoke_embeddings(ns))
    service = MemoryService(manager=manager, settings=settings)

    summary = ns["run_smoke_test"](
        service, manager, settings=settings, user_id=900000033, require_semantic_skip=False
    )

    # forget_user is called at least three times: initial cleanup, the
    # scenario's own end-of-run cleanup, and the finally-block safety net --
    # every one of them idempotent against an already-empty namespace.
    assert manager._delete_namespace_calls >= 3
    assert summary["cleanup_verified"] is True


def test_smoke_test_cleanup_failure_output_never_contains_injected_secret(
    caplog: pytest.LogCaptureFixture,
) -> None:
    ns = _load_script("smoke_test_memory.py")
    fake_secret = "sk-FAKE-INJECTED-SECRET-VALUE"
    settings = _build_settings(OPENAI_API_KEY=fake_secret, PINECONE_API_KEY=fake_secret)
    # Matches PineconeManager.delete_namespace's real contract: a genuine
    # (non-404) failure always carries a fixed, secret-free message -- never
    # the underlying exception's body, headers, or credentials.
    manager = RaisingDeleteManager(_default_smoke_embeddings(ns), fail_on_call_number=1)

    ns["get_settings"] = lambda: settings
    ns["PineconeManager"] = lambda settings: manager

    with caplog.at_level(logging.INFO):
        exit_code = ns["main"](["--user-id", "900000034"])

    assert exit_code != 0
    assert fake_secret not in caplog.text


def test_smoke_test_happy_path_end_to_end() -> None:
    ns = _load_script("smoke_test_memory.py")
    settings = _build_settings()
    manager = InMemoryFakeManager(_default_smoke_embeddings(ns))
    service = MemoryService(manager=manager, settings=settings)

    summary = ns["run_smoke_test"](
        service, manager, settings=settings, user_id=900000005, require_semantic_skip=False
    )

    assert summary["first_write"]["action"] == "inserted"
    assert summary["first_write"]["reason"] == "new_memory"
    assert summary["exact_duplicate"]["action"] == "skipped"
    assert summary["exact_duplicate"]["reason"] == "exact_duplicate"
    assert summary["different_memory"]["action"] == "inserted"
    assert summary["different_memory"]["reason"] == "new_memory"
    assert summary["recall_count"] >= 1
    assert summary["cleanup_verified"] is True


def test_smoke_test_semantic_paraphrase_reported_without_mandatory_failure_by_default() -> None:
    ns = _load_script("smoke_test_memory.py")
    settings = _build_settings()
    # Orthogonal (low-similarity) paraphrase embedding -> gets inserted, not skipped.
    manager = InMemoryFakeManager(_default_smoke_embeddings(ns))
    service = MemoryService(manager=manager, settings=settings)

    summary = ns["run_smoke_test"](
        service, manager, settings=settings, user_id=900000006, require_semantic_skip=False
    )

    assert summary["semantic_paraphrase"]["action"] == "inserted"


def test_smoke_test_require_semantic_skip_succeeds_when_skipped() -> None:
    ns = _load_script("smoke_test_memory.py")
    settings = _build_settings()
    embeddings = _default_smoke_embeddings(ns)
    embeddings[ns["PARAPHRASE_TEXT"]] = [0.99, 0.14, 0.0, 0.0]  # high similarity -> duplicate
    manager = InMemoryFakeManager(embeddings)
    service = MemoryService(manager=manager, settings=settings)

    summary = ns["run_smoke_test"](
        service, manager, settings=settings, user_id=900000007, require_semantic_skip=True
    )

    assert summary["semantic_paraphrase"]["action"] == "skipped"
    assert summary["semantic_paraphrase"]["reason"] == "semantic_duplicate"


def test_smoke_test_require_semantic_skip_fails_when_not_skipped() -> None:
    ns = _load_script("smoke_test_memory.py")
    settings = _build_settings()
    manager = InMemoryFakeManager(_default_smoke_embeddings(ns))  # orthogonal -> inserted
    service = MemoryService(manager=manager, settings=settings)

    with pytest.raises(ns["SmokeTestError"]):
        ns["run_smoke_test"](
            service, manager, settings=settings, user_id=900000008, require_semantic_skip=True
        )


def test_smoke_test_final_cleanup_always_runs_after_failure() -> None:
    ns = _load_script("smoke_test_memory.py")
    settings = _build_settings()
    embeddings = _default_smoke_embeddings(ns)
    # Deliberately identical to the first memory -> triggers a semantic-duplicate
    # failure on the "different memory must be inserted" assertion.
    embeddings[ns["DIFFERENT_MEMORY_TEXT"]] = [1.0, 0.0, 0.0, 0.0]
    manager = InMemoryFakeManager(embeddings)
    service = MemoryService(manager=manager, settings=settings)
    user_id = 900000002
    namespace = service.namespace_for_user(user_id)

    with pytest.raises(ns["SmokeTestError"]):
        ns["run_smoke_test"](
            service, manager, settings=settings, user_id=user_id, require_semantic_skip=False
        )

    assert manager.delete_calls.count(namespace) >= 2


def test_smoke_test_never_calls_delete_index() -> None:
    ns = _load_script("smoke_test_memory.py")
    settings = _build_settings()
    manager = InMemoryFakeManager(_default_smoke_embeddings(ns))
    service = MemoryService(manager=manager, settings=settings)

    # InMemoryFakeManager raises AssertionError if delete_index is ever
    # accessed; reaching this point without error proves the smoke test only
    # ever deletes the synthetic user's namespace, never the index.
    summary = ns["run_smoke_test"](
        service, manager, settings=settings, user_id=900000009, require_semantic_skip=False
    )

    assert summary["cleanup_verified"] is True


def test_smoke_test_main_rejects_invalid_user_id_without_manager_calls() -> None:
    ns = _load_script("smoke_test_memory.py")

    def _explode(settings: Any) -> Any:
        raise AssertionError("must not construct PineconeManager for an invalid --user-id")

    ns["get_settings"] = lambda: _build_settings()
    ns["PineconeManager"] = _explode

    exit_code = ns["main"](["--user-id", "42"])

    assert exit_code != 0


def test_smoke_test_main_returns_nonzero_on_failure() -> None:
    ns = _load_script("smoke_test_memory.py")
    settings = _build_settings()
    embeddings = _default_smoke_embeddings(ns)
    embeddings[ns["DIFFERENT_MEMORY_TEXT"]] = [1.0, 0.0, 0.0, 0.0]
    manager = InMemoryFakeManager(embeddings)

    ns["get_settings"] = lambda: settings
    ns["PineconeManager"] = lambda settings: manager

    exit_code = ns["main"](["--user-id", "900000010"])

    assert exit_code != 0


def test_smoke_test_main_output_never_contains_injected_fake_secret(
    capsys: pytest.CaptureFixture[str],
) -> None:
    ns = _load_script("smoke_test_memory.py")
    fake_secret = "sk-FAKE-INJECTED-SECRET-VALUE"
    settings = _build_settings(OPENAI_API_KEY=fake_secret, PINECONE_API_KEY=fake_secret)
    manager = InMemoryFakeManager(_default_smoke_embeddings(ns))

    ns["get_settings"] = lambda: settings
    ns["PineconeManager"] = lambda settings: manager

    exit_code = ns["main"](["--user-id", "900000011"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert fake_secret not in captured.out


# ---------------------------------------------------------------------------
# Cosine score boundary regression: real PineconeManager behind the smoke test
# ---------------------------------------------------------------------------
#
# These tests wire the *real* PineconeManager (not InMemoryFakeManager) behind
# MemoryService and the smoke script, backed only by a fake raw Pinecone index
# handle and a fake OpenAI client. This exercises PineconeManager's actual
# query-response parsing -- including its cosine-score boundary normalization
# -- rather than reimplementing that logic in a test double.


def _score_boundary_index_description() -> dict[str, Any]:
    return {
        "name": "test-index",
        "host": "test-index.svc.pinecone.io",
        "dimension": DIMENSION,
        "metric": "cosine",
        "status": {"ready": True, "state": "Ready"},
    }


class ScoreInjectingPineconeClient:
    """Fake control-plane client wrapping a single fake raw index handle."""

    def __init__(self, index_handle: ScoreInjectingIndexHandle) -> None:
        self._index_handle = index_handle

    def describe_index(self, name: str) -> dict[str, Any]:
        return _score_boundary_index_description()

    def Index(self, *, host: str = "", name: str = "") -> ScoreInjectingIndexHandle:
        return self._index_handle


class ScoreInjectingIndexHandle:
    """Fake raw Pinecone index: real in-memory storage and real cosine
    similarity, but lets a test override the *raw* score Pinecone would
    return for the top match on the next non-empty query -- e.g. a
    floating-point overshoot like ``1.000000119``, or a materially invalid
    score like ``1.01`` -- to reproduce exactly what a live external
    response boundary looks like. PineconeManager's own response parser
    (unmodified, real) is what turns that raw score into a ``VectorMatch``
    or a ``VectorQueryError``.
    """

    def __init__(self) -> None:
        self._store: dict[str, dict[str, dict[str, Any]]] = {}
        self.score_overrides: list[float] = []

    def upsert(self, *, vectors: list[dict[str, Any]], namespace: str) -> dict[str, Any]:
        bucket = self._store.setdefault(namespace, {})
        for vector in vectors:
            bucket[vector["id"]] = {
                "values": list(vector["values"]),
                "metadata": dict(vector["metadata"]),
            }
        return {"upserted_count": len(vectors)}

    def query(
        self,
        *,
        vector: list[float],
        namespace: str,
        top_k: int,
        include_metadata: bool = True,
        filter: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        bucket = self._store.get(namespace, {})
        candidates = []
        for vector_id, entry in bucket.items():
            if not _matches_filter(entry["metadata"], filter):
                continue
            candidates.append(
                {
                    "id": vector_id,
                    "score": _cosine(vector, entry["values"]),
                    "metadata": entry["metadata"],
                }
            )
        candidates.sort(key=lambda m: m["score"], reverse=True)
        top = candidates[:top_k]
        # Only ever substitute a genuine self-match (natural cosine already
        # ~1.0) -- mirroring reality, where the external service's tiny
        # floating-point drift only ever shows up right at that boundary,
        # never on an arbitrary, mathematically-distant similarity score.
        if top and self.score_overrides and top[0]["score"] >= 1.0 - 1e-9:
            top[0] = {**top[0], "score": self.score_overrides.pop(0)}
        return {"matches": top}

    def fetch(self, *, ids: list[str], namespace: str) -> dict[str, Any]:
        bucket = self._store.get(namespace, {})
        return {"vectors": {vid: bucket[vid] for vid in ids if vid in bucket}}

    def delete(self, *, delete_all: bool, namespace: str) -> None:
        self._store.pop(namespace, None)


class ScoreInjectingOpenAIClient:
    """Fake OpenAI client returning deterministic, pre-configured embeddings."""

    def __init__(self, embeddings: dict[str, list[float]]) -> None:
        self._embeddings = embeddings
        self.embeddings = SimpleNamespace(create=self._create)

    def _create(self, *, input: str, model: str) -> dict[str, Any]:
        if input not in self._embeddings:
            raise AssertionError(f"no fake embedding configured for text: {input!r}")
        return {"data": [{"embedding": list(self._embeddings[input])}]}


def _build_real_manager_with_fake_index(
    embeddings: dict[str, list[float]], settings: Settings
) -> tuple[Any, ScoreInjectingIndexHandle]:
    from telegram_vector_memory_bot.pinecone_manager import PineconeManager

    index_handle = ScoreInjectingIndexHandle()
    pinecone_client = ScoreInjectingPineconeClient(index_handle)
    openai_client = ScoreInjectingOpenAIClient(embeddings)
    manager = PineconeManager(
        settings, pinecone_client=pinecone_client, openai_client=openai_client
    )
    return manager, index_handle


def test_smoke_test_query_visibility_self_match_overshoot_is_canonicalized() -> None:
    """A self-match query score of 1.000000119 -- exactly the kind of
    microscopic floating-point overshoot the live smoke test hit -- must be
    canonicalized to 1.0 by the real PineconeManager, and the query-visibility
    check must see it as a normal, valid match rather than raising."""
    ns = _load_script("smoke_test_memory.py")
    settings = _build_settings()
    assert settings.MEMORY_SIMILARITY_THRESHOLD == pytest.approx(0.50)

    manager, index_handle = _build_real_manager_with_fake_index(
        _default_smoke_embeddings(ns), settings
    )
    service = MemoryService(manager=manager, settings=settings)

    user_id = 900000040
    namespace = service.namespace_for_user(user_id)
    service.forget_user(user_id=user_id)  # idempotent delete on an absent namespace

    first_write = service.remember(user_id=user_id, text=ns["SHORT_ANSWERS_TEXT"])
    assert first_write.action.value == "inserted"
    first_memory_id = first_write.memory_id

    # Force the *next* non-empty query's top match -- the self-match visibility
    # check below -- to return the exact overshoot value observed live.
    index_handle.score_overrides = [1.000000119]

    first_memory_query_embedding = manager.create_embedding(ns["SHORT_ANSWERS_TEXT"])
    matches = manager.query_by_vector(
        values=first_memory_query_embedding,
        namespace=namespace,
        top_k=1,
        metadata_filter={"record_type": {"$eq": "user_memory"}},
    )

    assert matches[0].vector_id == first_memory_id
    assert matches[0].score == 1.0

    # Cleanup polling still works against the real manager afterwards.
    service.forget_user(user_id=user_id)
    assert service.recall(user_id=user_id, query=ns["RECALL_QUERY_TEXT"]) == []


def test_smoke_test_materially_invalid_score_still_aborts_safely() -> None:
    """A materially invalid raw score (1.01, far beyond the epsilon tolerance)
    must still raise -- the fix must not have widened the tolerance -- and
    that failure must propagate rather than being silently treated as a
    passing visibility check."""
    ns = _load_script("smoke_test_memory.py")
    settings = _build_settings()

    manager, index_handle = _build_real_manager_with_fake_index(
        _default_smoke_embeddings(ns), settings
    )
    service = MemoryService(manager=manager, settings=settings)

    user_id = 900000041
    namespace = service.namespace_for_user(user_id)
    service.forget_user(user_id=user_id)

    first_write = service.remember(user_id=user_id, text=ns["SHORT_ANSWERS_TEXT"])
    assert first_write.action.value == "inserted"

    index_handle.score_overrides = [1.01]

    embedding = manager.create_embedding(ns["SHORT_ANSWERS_TEXT"])

    with pytest.raises(ns["VectorMemoryError"]):
        manager.query_by_vector(
            values=embedding,
            namespace=namespace,
            top_k=1,
            metadata_filter={"record_type": {"$eq": "user_memory"}},
        )

    service.forget_user(user_id=user_id)


def test_smoke_test_run_smoke_test_end_to_end_survives_self_match_overshoot() -> None:
    """The full smoke-test scenario, run through the real PineconeManager,
    must complete successfully even when the query-visibility self-match
    carries a microscopic floating-point overshoot -- reproducing the
    original live failure end-to-end rather than only at the unit level."""
    ns = _load_script("smoke_test_memory.py")
    settings = _build_settings()

    manager, index_handle = _build_real_manager_with_fake_index(
        _default_smoke_embeddings(ns), settings
    )
    service = MemoryService(manager=manager, settings=settings)

    # Every non-empty query encountered during the run gets the overshoot
    # applied to its top match; the scenario touches several such queries
    # (duplicate lookups, the explicit visibility check, and recall), and
    # every one of them must still resolve correctly.
    index_handle.score_overrides = [1.000000119] * 10

    summary = ns["run_smoke_test"](
        service, manager, settings=settings, user_id=900000042, require_semantic_skip=False
    )

    assert summary["first_write"]["action"] == "inserted"
    assert summary["different_memory"]["action"] == "inserted"
    assert summary["recall_count"] >= 1
    assert summary["cleanup_verified"] is True


# ---------------------------------------------------------------------------
# Consistency-timeout / poll-interval argument validation (both scripts)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("consistency_timeout", [0.0, -1.0])
def test_calibrate_rejects_non_positive_consistency_timeout(consistency_timeout: float) -> None:
    ns = _load_script("calibrate_similarity.py")

    with pytest.raises(ns["CalibrationError"]):
        ns["_validate_consistency_arguments"](consistency_timeout, 1.0)


@pytest.mark.parametrize("poll_interval", [0.0, -1.0])
def test_calibrate_rejects_non_positive_poll_interval(poll_interval: float) -> None:
    ns = _load_script("calibrate_similarity.py")

    with pytest.raises(ns["CalibrationError"]):
        ns["_validate_consistency_arguments"](20.0, poll_interval)


def test_calibrate_rejects_poll_interval_greater_than_timeout() -> None:
    ns = _load_script("calibrate_similarity.py")

    with pytest.raises(ns["CalibrationError"]):
        ns["_validate_consistency_arguments"](5.0, 10.0)


def test_calibrate_accepts_poll_interval_equal_to_timeout() -> None:
    ns = _load_script("calibrate_similarity.py")

    ns["_validate_consistency_arguments"](5.0, 5.0)


def test_calibrate_main_rejects_invalid_consistency_options_without_manager_calls() -> None:
    ns = _load_script("calibrate_similarity.py")

    def _explode(settings: Any) -> Any:
        raise AssertionError("must not construct PineconeManager for invalid consistency options")

    ns["get_settings"] = lambda: _build_settings()
    ns["PineconeManager"] = _explode

    exit_code = ns["main"](["--consistency-timeout", "0"])

    assert exit_code != 0


@pytest.mark.parametrize("consistency_timeout", [0.0, -1.0])
def test_smoke_test_rejects_non_positive_consistency_timeout(consistency_timeout: float) -> None:
    ns = _load_script("smoke_test_memory.py")

    with pytest.raises(ns["SmokeTestError"]):
        ns["_validate_consistency_arguments"](consistency_timeout, 1.0)


@pytest.mark.parametrize("poll_interval", [0.0, -1.0])
def test_smoke_test_rejects_non_positive_poll_interval(poll_interval: float) -> None:
    ns = _load_script("smoke_test_memory.py")

    with pytest.raises(ns["SmokeTestError"]):
        ns["_validate_consistency_arguments"](20.0, poll_interval)


def test_smoke_test_rejects_poll_interval_greater_than_timeout() -> None:
    ns = _load_script("smoke_test_memory.py")

    with pytest.raises(ns["SmokeTestError"]):
        ns["_validate_consistency_arguments"](5.0, 10.0)


def test_smoke_test_main_rejects_invalid_consistency_options_without_manager_calls() -> None:
    ns = _load_script("smoke_test_memory.py")

    def _explode(settings: Any) -> Any:
        raise AssertionError("must not construct PineconeManager for invalid consistency options")

    ns["get_settings"] = lambda: _build_settings()
    ns["PineconeManager"] = _explode

    exit_code = ns["main"](["--poll-interval", "0"])

    assert exit_code != 0


# ---------------------------------------------------------------------------
# Calibration: bounded visibility polling
# ---------------------------------------------------------------------------


def test_calibration_visibility_retries_then_succeeds_without_repeating_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Covers: 0,0-then-1 match succeeds; no repeated upsert; no repeated
    reference/candidate embedding generation while polling."""
    ns = _load_script("calibrate_similarity.py")
    pairs = ns["CALIBRATION_PAIRS"]
    manager = DelayedVisibilityManager(_calibration_embeddings(pairs), empty_responses=2)
    sleep_calls = _patch_time(monkeypatch, step=1.0)

    summary = ns["run_calibration"](
        manager, "text-embedding-3-small", consistency_timeout=10.0, poll_interval=1.0
    )

    assert len(summary["pairs"]) == len(pairs)
    # Only the first pair (processed first) absorbs the two forced-empty
    # responses; later pairs see the real store immediately.
    assert summary["pairs"][0]["visibility_attempts"] == 3
    assert all(p["visibility_attempts"] == 1 for p in summary["pairs"][1:])
    assert sleep_calls == [pytest.approx(1.0), pytest.approx(1.0)]

    assert len(manager.upsert_calls) == len(pairs)
    # Exactly one reference + one candidate embedding per pair, regardless of
    # how many visibility polling attempts that pair needed. (Two pairs share
    # an identical reference phrase, so counting per pair-invocation -- not by
    # deduplicated text -- is what actually proves no re-generation occurred.)
    assert len(manager.create_embedding_calls) == 2 * len(pairs)

    assert len(manager.delete_calls) == 1


def test_calibration_more_than_one_match_fails_immediately(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ns = _load_script("calibrate_similarity.py")
    pairs = ns["CALIBRATION_PAIRS"]
    manager = DuplicateMatchQueryManager(_calibration_embeddings(pairs))
    sleep_calls = _patch_time(monkeypatch)

    with pytest.raises(ns["CalibrationError"]):
        ns["run_calibration"](manager, "text-embedding-3-small")

    # Fails immediately -- a malformed state is never retried.
    assert sleep_calls == []
    assert len(manager.delete_calls) == 1
    assert len(manager.upsert_calls) == 1


def test_calibration_visibility_timeout_fails_safely_and_cleans_up(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ns = _load_script("calibrate_similarity.py")
    pairs = ns["CALIBRATION_PAIRS"]
    manager = AlwaysEmptyQueryManager(_calibration_embeddings(pairs))
    sleep_calls = _patch_time(monkeypatch, step=1.0)

    with pytest.raises(ns["CalibrationVisibilityTimeoutError"]):
        ns["run_calibration"](
            manager, "text-embedding-3-small", consistency_timeout=3.0, poll_interval=1.0
        )

    assert len(sleep_calls) == 2
    assert len(manager.delete_calls) == 1
    assert len(manager.upsert_calls) == 1
    assert manager.create_embedding_calls.count(pairs[0]["reference"]) == 1
    assert manager.create_embedding_calls.count(pairs[0]["candidate"]) == 1


def test_calibration_visibility_timeout_output_never_contains_injected_secret(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    ns = _load_script("calibrate_similarity.py")
    pairs = ns["CALIBRATION_PAIRS"]
    fake_secret = "sk-FAKE-INJECTED-SECRET-VALUE"
    manager = AlwaysEmptyQueryManager(_calibration_embeddings(pairs))
    settings = _build_settings(OPENAI_API_KEY=fake_secret, PINECONE_API_KEY=fake_secret)

    ns["get_settings"] = lambda: settings
    ns["PineconeManager"] = lambda settings: manager

    _patch_time(monkeypatch, step=1.0)

    with caplog.at_level(logging.INFO):
        exit_code = ns["main"](["--consistency-timeout", "3", "--poll-interval", "1"])

    assert exit_code != 0
    assert fake_secret not in caplog.text


# ---------------------------------------------------------------------------
# Smoke test: bounded visibility polling
# ---------------------------------------------------------------------------


def test_smoke_fetch_visibility_retries_then_succeeds_without_repeating_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Covers: fetch returns missing then visible; no repeated remember()/upsert
    or embedding generation while polling the fetch check itself."""
    ns = _load_script("smoke_test_memory.py")
    settings = _build_settings()
    manager = DelayedFetchVisibilityManager(_default_smoke_embeddings(ns), missing_fetches=2)
    service = MemoryService(manager=manager, settings=settings)
    sleep_calls = _patch_time(monkeypatch, step=0.01)

    summary = ns["run_smoke_test"](
        service,
        manager,
        settings=settings,
        user_id=900000020,
        require_semantic_skip=False,
        consistency_timeout=5.0,
        poll_interval=0.5,
    )

    assert summary["first_write"]["action"] == "inserted"
    assert summary["exact_duplicate"]["action"] == "skipped"
    assert summary["exact_duplicate"]["reason"] == "exact_duplicate"
    assert sleep_calls == [pytest.approx(0.5)]

    policy = MemoryPolicy(settings.MEMORY_SIMILARITY_THRESHOLD)
    first_memory_id = policy.memory_id_for_text(ns["SHORT_ANSWERS_TEXT"])
    assert sum(1 for c in manager.upsert_calls if c["vector_id"] == first_memory_id) == 1
    # One embedding for the actual write, one for this script's own explicit
    # query-visibility check -- never recreated per poll attempt.
    assert manager.create_embedding_calls.count(ns["SHORT_ANSWERS_TEXT"]) == 2


def test_smoke_query_visibility_confirmed_before_paraphrase_processing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ns = _load_script("smoke_test_memory.py")
    settings = _build_settings()
    # 1 call is "free" (remember(first_write)'s own duplicate-check query, which
    # is legitimately empty since nothing is stored yet); the next 2 are the
    # forced-empty attempts this test actually exercises.
    manager = DelayedVisibilityManager(_default_smoke_embeddings(ns), empty_responses=3)
    service = MemoryService(manager=manager, settings=settings)
    sleep_calls = _patch_time(monkeypatch, step=0.01)

    summary = ns["run_smoke_test"](
        service,
        manager,
        settings=settings,
        user_id=900000021,
        require_semantic_skip=False,
        consistency_timeout=5.0,
        poll_interval=0.5,
    )

    assert summary["first_write"]["action"] == "inserted"
    # Orthogonal embedding by default -> correctly inserted once query-visibility
    # of the first memory was confirmed (not misclassified due to a stale read).
    assert summary["semantic_paraphrase"]["action"] == "inserted"
    assert len(sleep_calls) == 2
    assert summary["cleanup_verified"] is True


def test_smoke_recall_retries_until_populated(monkeypatch: pytest.MonkeyPatch) -> None:
    ns = _load_script("smoke_test_memory.py")
    settings = _build_settings()
    manager = RecallVisibilityManager(_default_smoke_embeddings(ns), forced_responses=2)
    service = MemoryService(manager=manager, settings=settings)
    sleep_calls = _patch_time(monkeypatch, step=0.01)

    summary = ns["run_smoke_test"](
        service,
        manager,
        settings=settings,
        user_id=900000022,
        require_semantic_skip=False,
        consistency_timeout=5.0,
        poll_interval=0.5,
    )

    assert summary["recall_count"] >= 1
    assert len(sleep_calls) == 2
    assert summary["cleanup_verified"] is True

    different_memory_id = MemoryPolicy(settings.MEMORY_SIMILARITY_THRESHOLD).memory_id_for_text(
        ns["DIFFERENT_MEMORY_TEXT"]
    )
    assert sum(1 for c in manager.upsert_calls if c["vector_id"] == different_memory_id) == 1


def test_smoke_recall_ignores_irrelevant_nonempty_result_until_expected_id_appears(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reproduces the live false positive directly: an older, unrelated memory
    is already query-visible and returned non-empty, but recall-populated
    polling must not accept that as consistency -- it must keep polling until
    the newly inserted training memory's exact ID appears, then succeed once
    the real store supplies it, ranked first."""
    ns = _load_script("smoke_test_memory.py")
    settings = _build_settings()
    manager = RecallVisibilityManager(
        _default_smoke_embeddings(ns),
        forced_responses=2,
        forced_result=lambda: [_irrelevant_match()],
    )
    service = MemoryService(manager=manager, settings=settings)
    sleep_calls = _patch_time(monkeypatch, step=0.01)

    summary = ns["run_smoke_test"](
        service,
        manager,
        settings=settings,
        user_id=900000026,
        require_semantic_skip=False,
        consistency_timeout=5.0,
        poll_interval=0.5,
    )

    expected_memory_id = MemoryPolicy(settings.MEMORY_SIMILARITY_THRESHOLD).memory_id_for_text(
        ns["DIFFERENT_MEMORY_TEXT"]
    )
    # Two forced, non-empty-but-irrelevant responses must not have ended
    # polling early -- exactly two sleeps occur before the real store (on the
    # third call) supplies the expected memory.
    assert len(sleep_calls) == 2
    assert summary["expected_recall_memory_id"] == expected_memory_id
    assert summary["expected_recall_present"] is True
    assert summary["expected_recall_rank"] == 1
    assert summary["expected_recall_within_top_k"] is True
    assert summary["recall_count"] >= 1
    assert summary["cleanup_verified"] is True
    # remember()/upsert for the expected memory happens exactly once -- never
    # repeated while polling recall.
    assert sum(1 for c in manager.upsert_calls if c["vector_id"] == expected_memory_id) == 1


def test_smoke_recall_expected_memory_present_at_rank_two_succeeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A successful top-k retrieval for the current (non-reranked) architecture:
    the expected memory is present but ranked second behind an older memory.
    Dense vector retrieval alone does not guarantee rank-1 relevance, so this
    must be accepted -- not treated as a failure -- as long as it is within
    the configured top_k. Pinecone's returned order must be preserved as the
    reported rank."""
    ns = _load_script("smoke_test_memory.py")
    settings = _build_settings()
    manager = RecallVisibilityManager(_default_smoke_embeddings(ns), forced_responses=1)
    service = MemoryService(manager=manager, settings=settings)
    user_id = 900000027
    namespace = service.namespace_for_user(user_id)
    policy = MemoryPolicy(settings.MEMORY_SIMILARITY_THRESHOLD)
    first_memory_id = policy.memory_id_for_text(ns["SHORT_ANSWERS_TEXT"])
    expected_memory_id = policy.memory_id_for_text(ns["DIFFERENT_MEMORY_TEXT"])

    def _rank_two_result() -> list[VectorMatch]:
        bucket = manager._store[namespace]
        return [
            VectorMatch(
                vector_id=first_memory_id,
                score=0.99,
                metadata=dict(bucket[first_memory_id]["metadata"]),
            ),
            VectorMatch(
                vector_id=expected_memory_id,
                score=0.98,
                metadata=dict(bucket[expected_memory_id]["metadata"]),
            ),
        ]

    manager._forced_result = _rank_two_result
    sleep_calls = _patch_time(monkeypatch, step=0.01)

    summary = ns["run_smoke_test"](
        service,
        manager,
        settings=settings,
        user_id=user_id,
        require_semantic_skip=False,
        consistency_timeout=5.0,
        poll_interval=0.5,
    )

    # Accepted on the very first check -- a rank-2 presence is not "not yet
    # consistent" and must not be retried.
    assert sleep_calls == []
    assert summary["expected_recall_memory_id"] == expected_memory_id
    assert summary["expected_recall_present"] is True
    assert summary["expected_recall_rank"] == 2
    assert summary["expected_recall_within_top_k"] is True
    assert summary["cleanup_verified"] is True
    # Pinecone's returned order is preserved verbatim in the reported results.
    assert [r["memory_id"] for r in summary["recall_results"]] == [
        first_memory_id,
        expected_memory_id,
    ]
    assert sum(1 for c in manager.upsert_calls if c["vector_id"] == expected_memory_id) == 1


def test_smoke_recall_expected_memory_at_last_allowed_top_k_position_succeeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The expected memory ranked exactly at the last allowed top_k position
    (rank == MEMORY_TOP_K) must still be accepted -- the boundary of the
    configured window, not just ranks 1 or 2."""
    ns = _load_script("smoke_test_memory.py")
    settings = _build_settings()
    top_k = settings.MEMORY_TOP_K
    manager = RecallVisibilityManager(_default_smoke_embeddings(ns), forced_responses=1)
    service = MemoryService(manager=manager, settings=settings)
    user_id = 900000043
    namespace = service.namespace_for_user(user_id)
    policy = MemoryPolicy(settings.MEMORY_SIMILARITY_THRESHOLD)
    expected_memory_id = policy.memory_id_for_text(ns["DIFFERENT_MEMORY_TEXT"])

    def _rank_last_result() -> list[VectorMatch]:
        bucket = manager._store[namespace]
        fillers = [
            VectorMatch(
                vector_id=f"filler-memory-{i}",
                score=0.9 - i * 0.01,
                metadata={
                    "text": f"filler memory number {i}",
                    "content_hash": f"filler-hash-{i}",
                    "created_at": "2020-01-01T00:00:00+00:00",
                    "source": "telegram",
                    "record_type": "user_memory",
                },
            )
            for i in range(1, top_k)
        ]
        expected = VectorMatch(
            vector_id=expected_memory_id,
            score=0.5,
            metadata=dict(bucket[expected_memory_id]["metadata"]),
        )
        return [*fillers, expected]

    manager._forced_result = _rank_last_result
    sleep_calls = _patch_time(monkeypatch, step=0.01)

    summary = ns["run_smoke_test"](
        service,
        manager,
        settings=settings,
        user_id=user_id,
        require_semantic_skip=False,
        consistency_timeout=5.0,
        poll_interval=0.5,
    )

    assert sleep_calls == []
    assert summary["expected_recall_rank"] == top_k
    assert summary["expected_recall_within_top_k"] is True
    assert summary["recall_count"] == top_k
    assert summary["cleanup_verified"] is True
    # Order preserved: the expected memory's reported position is last.
    assert summary["recall_results"][-1]["memory_id"] == expected_memory_id


def test_smoke_recall_expected_memory_never_appears_times_out_safely(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """If the expected memory's exact ID never appears in recall results
    before the consistency timeout, the script must raise a dedicated
    visibility timeout error rather than accepting an irrelevant non-empty
    result or hanging indefinitely -- and the finally-block safety-net
    cleanup must still run despite the failure."""
    ns = _load_script("smoke_test_memory.py")
    settings = _build_settings()
    manager = RecallVisibilityManager(
        _default_smoke_embeddings(ns),
        forced_responses=10_000,
        forced_result=lambda: [_irrelevant_match()],
    )
    service = MemoryService(manager=manager, settings=settings)
    _patch_time(monkeypatch, step=1.0)

    with caplog.at_level(logging.INFO):
        with pytest.raises(ns["SmokeTestVisibilityTimeoutError"]) as exc_info:
            ns["run_smoke_test"](
                service,
                manager,
                settings=settings,
                user_id=900000028,
                require_semantic_skip=False,
                consistency_timeout=3.0,
                poll_interval=1.0,
            )

    assert "newly inserted training memory" in str(exc_info.value)
    # The safety net's own cleanup attempt still ran (and, since this fake
    # manager keeps returning a non-empty irrelevant match forever, it too
    # times out and is logged -- but that never replaces the original error).
    assert "Safety-net cleanup (forget_user) failed" in caplog.text


def test_smoke_test_expected_recall_output_never_contains_full_metadata_or_secrets(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The structured summary must expose only the expected-recall scalars
    (id/present/rank/score) and short labels -- never full stored metadata,
    embeddings, client representations, or configured secrets."""
    ns = _load_script("smoke_test_memory.py")
    fake_secret = "sk-FAKE-INJECTED-SECRET-VALUE"
    settings = _build_settings(OPENAI_API_KEY=fake_secret, PINECONE_API_KEY=fake_secret)
    manager = InMemoryFakeManager(_default_smoke_embeddings(ns))

    ns["get_settings"] = lambda: settings
    ns["PineconeManager"] = lambda settings: manager

    exit_code = ns["main"](["--user-id", "900000029"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert fake_secret not in captured.out
    assert "content_hash" not in captured.out
    assert "embedding" not in captured.out.lower()
    assert "VectorMatch" not in captured.out

    summary = json.loads(captured.out)
    assert summary["expected_recall_present"] is True
    assert summary["expected_recall_rank"] == 1
    assert summary["expected_recall_within_top_k"] is True
    assert isinstance(summary["expected_recall_memory_id"], str)
    assert summary["expected_recall_memory_id"]
    assert isinstance(summary["expected_recall_score"], float)


def test_smoke_deletion_recall_retries_until_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    ns = _load_script("smoke_test_memory.py")
    settings = _build_settings()
    manager = RecallVisibilityManager(
        _default_smoke_embeddings(ns),
        skip_recall_calls=1,
        forced_responses=2,
        forced_result=lambda: [_stale_match()],
    )
    service = MemoryService(manager=manager, settings=settings)
    sleep_calls = _patch_time(monkeypatch, step=0.01)

    summary = ns["run_smoke_test"](
        service,
        manager,
        settings=settings,
        user_id=900000023,
        require_semantic_skip=False,
        consistency_timeout=5.0,
        poll_interval=0.5,
    )

    # A single stale (still non-empty) recall result is never treated as a
    # cleanup failure on its own -- polling continues until it is truly empty.
    assert summary["cleanup_verified"] is True
    assert len(sleep_calls) == 2


def test_smoke_deletion_visibility_timeout_fails_safely_without_masking_original_error(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    ns = _load_script("smoke_test_memory.py")
    settings = _build_settings()
    manager = RecallVisibilityManager(
        _default_smoke_embeddings(ns),
        skip_recall_calls=1,
        forced_responses=10_000,
        forced_result=lambda: [_stale_match()],
    )
    service = MemoryService(manager=manager, settings=settings)
    _patch_time(monkeypatch, step=1.0)

    with caplog.at_level(logging.INFO):
        with pytest.raises(ns["SmokeTestVisibilityTimeoutError"]) as exc_info:
            ns["run_smoke_test"](
                service,
                manager,
                settings=settings,
                user_id=900000024,
                require_semantic_skip=False,
                consistency_timeout=3.0,
                poll_interval=1.0,
            )

    # The exception that escapes is the original scenario failure (from the
    # in-scenario forget_user step), not the finally-block safety net's own
    # timeout -- the safety net's failure is reported but never replaces it.
    assert "after forget_user" in str(exc_info.value)
    assert "safety-net" not in str(exc_info.value)
    assert "Safety-net cleanup (forget_user) failed" in caplog.text


def test_smoke_test_deletion_visibility_timeout_output_never_contains_injected_secret(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    ns = _load_script("smoke_test_memory.py")
    fake_secret = "sk-FAKE-INJECTED-SECRET-VALUE"
    settings = _build_settings(OPENAI_API_KEY=fake_secret, PINECONE_API_KEY=fake_secret)
    manager = RecallVisibilityManager(
        _default_smoke_embeddings(ns),
        skip_recall_calls=1,
        forced_responses=10_000,
        forced_result=lambda: [_stale_match()],
    )

    ns["get_settings"] = lambda: settings
    ns["PineconeManager"] = lambda settings: manager

    _patch_time(monkeypatch, step=1.0)

    with caplog.at_level(logging.INFO):
        exit_code = ns["main"](
            ["--user-id", "900000025", "--consistency-timeout", "3", "--poll-interval", "1"]
        )

    assert exit_code != 0
    assert fake_secret not in caplog.text
