"""Unit tests for scripts/calibrate_similarity.py and scripts/smoke_test_memory.py.

Both scripts make real network calls when run directly, but neither creates
any client at import time. These tests load each script with ``runpy`` (which
executes top-level code but never triggers the ``if __name__ == "__main__"``
guard), then monkeypatch/replace the script's ``get_settings`` and
``PineconeManager`` references with fully offline test doubles. No test
performs a real network request or reads the user's real ``.env`` file.
"""

from __future__ import annotations

import math
import runpy
from pathlib import Path
from typing import Any

import pytest

from telegram_vector_memory_bot.config import Settings
from telegram_vector_memory_bot.memory_policy import MemoryPolicy
from telegram_vector_memory_bot.memory_service import MemoryService
from telegram_vector_memory_bot.models import IndexInfo, VectorMatch

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
        assert call["top_k"] == 1


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

    summary = ns["run_smoke_test"](service, user_id=user_id, require_semantic_skip=False)

    # If pre-cleanup hadn't run, this exact deterministic ID would already
    # exist and the first write would be reported as an exact duplicate.
    assert summary["first_write"]["action"] == "inserted"


def test_smoke_test_happy_path_end_to_end() -> None:
    ns = _load_script("smoke_test_memory.py")
    settings = _build_settings()
    manager = InMemoryFakeManager(_default_smoke_embeddings(ns))
    service = MemoryService(manager=manager, settings=settings)

    summary = ns["run_smoke_test"](service, user_id=900000005, require_semantic_skip=False)

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

    summary = ns["run_smoke_test"](service, user_id=900000006, require_semantic_skip=False)

    assert summary["semantic_paraphrase"]["action"] == "inserted"


def test_smoke_test_require_semantic_skip_succeeds_when_skipped() -> None:
    ns = _load_script("smoke_test_memory.py")
    settings = _build_settings()
    embeddings = _default_smoke_embeddings(ns)
    embeddings[ns["PARAPHRASE_TEXT"]] = [0.99, 0.14, 0.0, 0.0]  # high similarity -> duplicate
    manager = InMemoryFakeManager(embeddings)
    service = MemoryService(manager=manager, settings=settings)

    summary = ns["run_smoke_test"](service, user_id=900000007, require_semantic_skip=True)

    assert summary["semantic_paraphrase"]["action"] == "skipped"
    assert summary["semantic_paraphrase"]["reason"] == "semantic_duplicate"


def test_smoke_test_require_semantic_skip_fails_when_not_skipped() -> None:
    ns = _load_script("smoke_test_memory.py")
    settings = _build_settings()
    manager = InMemoryFakeManager(_default_smoke_embeddings(ns))  # orthogonal -> inserted
    service = MemoryService(manager=manager, settings=settings)

    with pytest.raises(ns["SmokeTestError"]):
        ns["run_smoke_test"](service, user_id=900000008, require_semantic_skip=True)


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
        ns["run_smoke_test"](service, user_id=user_id, require_semantic_skip=False)

    assert manager.delete_calls.count(namespace) >= 2


def test_smoke_test_never_calls_delete_index() -> None:
    ns = _load_script("smoke_test_memory.py")
    settings = _build_settings()
    manager = InMemoryFakeManager(_default_smoke_embeddings(ns))
    service = MemoryService(manager=manager, settings=settings)

    # InMemoryFakeManager raises AssertionError if delete_index is ever
    # accessed; reaching this point without error proves the smoke test only
    # ever deletes the synthetic user's namespace, never the index.
    summary = ns["run_smoke_test"](service, user_id=900000009, require_semantic_skip=False)

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
    assert fake_secret not in captured.err
