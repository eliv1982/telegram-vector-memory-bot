# Telegram Vector Memory Bot

## Purpose

A learning project implementing a Telegram bot with long-term vector memory.
The bot is intended to remember facts shared by users across conversations,
using semantic search over embeddings stored in Pinecone, with deduplication
so that repeated or paraphrased facts do not create redundant memories.

## Planned MVP

- Receive messages from users via Telegram.
- Extract memory-worthy statements from user messages.
- Embed statements with an OpenAI embedding model.
- Deduplicate against existing memories (exact and semantic duplicates).
- Store new memories in Pinecone, isolated per user via namespaces.
- Retrieve relevant memories to give the bot context when answering.

## Architecture

- `src/telegram_vector_memory_bot/config.py` -- typed application settings,
  loaded from environment variables / `.env` via `pydantic-settings`.
- `src/telegram_vector_memory_bot/models.py` -- strict domain models and
  enums describing memory records, index metadata, vector matches, and the
  outcome of memory write attempts.
- `src/telegram_vector_memory_bot/pinecone_manager.py` -- `PineconeManager`,
  a typed infrastructure adapter over the Pinecone and OpenAI SDKs. It
  validates the configured index, generates embeddings, and performs
  namespace-scoped vector CRUD operations (upsert, query, fetch, delete,
  stats).
- `src/telegram_vector_memory_bot/memory_policy.py` -- `MemoryPolicy`, a
  pure, dependency-free set of deduplication rules: text normalization,
  deterministic hashing/IDs, namespace construction, a lightweight RU/EN
  negation guard, and the semantic-duplicate decision. No Pinecone, no
  OpenAI, no `Settings`, no I/O.
- `src/telegram_vector_memory_bot/memory_service.py` -- `MemoryService`,
  the application layer. It orchestrates `MemoryPolicy` and
  `PineconeManager` to implement `remember`, `recall`, and `forget_user`.
- Future stages will add: the Telegram bot entry point and live
  end-to-end validation.

## Project status

**Stage 4A: live calibration and smoke-test tooling, on top of Stage 3's
memory policy and application-level memory service.**

Stages 1-2 established the project skeleton, typed configuration, domain
models, and `PineconeManager`. Stage 3 adds two new layers on top:

### MemoryPolicy (pure, deterministic)

- `normalize_text`: NFKC-normalizes, strips, collapses internal whitespace,
  and casefolds -- used only for hashing/comparison, never overwrites the
  original text stored as a memory.
- `content_hash` / `memory_id_for_text`: an unsalted SHA-256 hash of the
  normalized text, and a deterministic `mem-<hash>` ID derived from it.
  Deliberately excludes the user ID -- users are already isolated by
  namespace, so identical text from two different users hashes the same
  but lives in two different namespaces.
- `namespace_for_user`: builds `<prefix>-<user_id>` and nothing else --
  never derived from username, first name, last name, or message text.
- `has_explicit_negation`: a small, documented RU/EN negation guard
  (`не`, `нет`, `никогда`, `больше не`, `not`, `no`, `never`, `don't`,
  `do not`) matched at word boundaries. **This is not a full contradiction
  detector** -- it will miss implicit negation like "I changed my mind",
  and only exists to stop a clear negation from being merged into an
  earlier, non-negated memory.
- `is_semantic_duplicate`: a candidate is only treated as a duplicate when
  its score is at or above the configured threshold **and** the new text
  and the existing text agree on whether they contain an explicit
  negation. For example, given the threshold is met:
  - "Пиши мне кратко и по существу." and "Я предпочитаю короткие ответы."
    -- may be treated as the same memory (no negation on either side).
  - "Я предпочитаю короткие ответы." and "Я больше не хочу коротких
    ответов." -- are **never** treated as the same memory, even at a very
    high similarity score, because exactly one side is negated.

### MemoryService (application layer, built on PineconeManager)

- `remember`: computes a deterministic ID first and checks for an **exact**
  duplicate via `fetch_vectors` (no embedding call needed for that check).
  If none exists, it creates exactly one embedding and looks for a
  **semantic** duplicate via a `top_k=1` query scoped to the user's
  namespace. A semantic duplicate is only ever **skipped**, never used to
  update or overwrite the existing vector.
  - stored metadata is a fixed, safe set: `user_id`, `text`, `content_hash`,
    `created_at` (UTC ISO-8601), `source`, `record_type`, and the optional
    Telegram fields (only when present). Bot responses, API keys, tokens,
    full Telegram update objects, prompts, and chat history are never
    stored.
- `recall`: queries only the requested user's namespace with a
  `record_type` metadata filter, and parses stored metadata strictly into
  `RecalledMemory` -- malformed stored metadata raises
  `StoredMemoryFormatError` rather than being silently fabricated.
- `forget_user`: deletes only that user's namespace. Never touches another
  user's namespace and never deletes the Pinecone index itself.

**Current limitation:** for this educational MVP, every valid, non-empty
message passed to `remember` is considered eligible for memory -- there is
no separate classifier deciding whether a message is "worth remembering".

### PineconeManager (infrastructure, from Stage 2)

`PineconeManager` is a typed infrastructure adapter responsible for:

- validating the configured Pinecone index (name, host, dimension, metric,
  ready state) exactly once at manager instantiation;
- generating and validating embeddings via an OpenAI-compatible client;
- namespace-scoped vector upsert, query (by vector or by text), fetch, and
  deletion;
- reading index-wide stats.

`PineconeManager` targets the index by its **resolved host** (returned by
`describe_index`), not by index name, because Pinecone's data-plane API is
served per-host; resolving the host once and caching the data client avoids
a `describe_index` round trip on every read or write.

External clients (Pinecone, OpenAI) are created lazily, only when a
`PineconeManager` is instantiated -- never at module import time. The
constructor accepts pre-built clients via dependency injection, so unit
tests run entirely against fakes/mocks and require no live credentials or
network access. The same is true of `MemoryService`: it creates no clients
of its own and only reuses the `PineconeManager` it is given.

`PineconeManager` remains infrastructure only: it has no concept of
duplicate detection or memory-write policy, and never returns a
`MemoryWriteResult` itself -- that decision now lives in `MemoryPolicy` and
`MemoryService`.

Cosine similarity scores returned from queries range from **-1 to 1**
(`VectorMatch.score`, `MemoryWriteResult.similarity_score`,
`RecalledMemory.score`). The configured duplicate *threshold* in `Settings`
(`MEMORY_SIMILARITY_THRESHOLD`) remains constrained to **0 to 1**, since
deduplication only ever cares about positive similarity.

### What's still pending

Telegram integration (handlers, bot commands, chat completion) has **not**
been implemented yet. The bot is not runnable as a Telegram bot at this
stage -- only the storage, policy, and application layers exist.

## Stage 4A: live calibration and smoke-test tooling

Stage 4A adds two small, standalone operator scripts under `scripts/` that
make **real** Pinecone and OpenAI API calls -- unlike every other layer in
this project, which is covered purely by offline unit tests against
fakes/mocks. Both scripts only construct `Settings` / `PineconeManager` /
`MemoryService` inside their `main()` function, never at import time, and
both clean up every piece of data they write, in a `finally` block, even on
partial failure. Neither script can delete the Pinecone index itself.

### Pinecone index requirements

Before running either script, the configured Pinecone index must already
exist with:

- **1536 dimensions** (matching the default embedding model,
  `text-embedding-3-small`);
- **metric: cosine**.

### Local configuration

Create `.env` from `.env.example` (see the [Configuration](#configuration)
section below) and fill in real `PINECONE_API_KEY`, `PINECONE_INDEX_NAME`,
`OPENAI_API_KEY`, and `OPENAI_CHAT_MODEL`. **Never commit `.env`** -- it is
already excluded via `.gitignore`.

### Eventual consistency: bounded visibility polling

Pinecone is **eventually consistent**: an acknowledged upsert or delete does
not guarantee that a subsequent query or fetch immediately reflects it. A
live calibration run once observed exactly this -- a reference upsert was
acknowledged, the immediate filtered query returned zero matches, and
cleanup still completed successfully. That is not a semantic-score failure;
it is a timing gap between write acknowledgment and read visibility.

Both scripts handle every read-after-write and read-after-delete boundary
with **bounded polling** instead of a single immediate check or a fixed
sleep: they re-issue the same read (never a new upsert, and never a
recreated embedding) at a configurable interval until it succeeds or a
configurable timeout elapses.

```bash
python scripts/calibrate_similarity.py --consistency-timeout 30 --poll-interval 2
python scripts/smoke_test_memory.py --consistency-timeout 30 --poll-interval 2
```

- `--consistency-timeout` (default `20.0` seconds): the maximum time to wait
  for visibility before failing with a clear, non-zero exit code.
- `--poll-interval` (default `1.0` second): the pause between polling
  attempts. Must be positive and no greater than `--consistency-timeout`.

This retry handling only affects *when* a read is trusted -- it never
changes a semantic score, the duplicate threshold, or the deduplication
policy itself. A reported score is always the Pinecone-returned score from
the query that first observed the write; nothing here re-derives or
adjusts it.

### Calibration: measuring real cosine scores

`scripts/calibrate_similarity.py` embeds four fixed, realistic
Russian-language phrase pairs (a likely paraphrase, a related-but-distinct
preference, a potentially conflicting preference, and an unrelated
statement), measures the real Pinecone cosine score for each, and prints a
table plus a JSON summary. All calibration data is written to a throwaway
`calibration-<uuid>` namespace and **deleted in a `finally` block** before
the script exits, regardless of success or failure.

```bash
python scripts/calibrate_similarity.py
```

After each reference upsert, the script polls the same exact `pair_id`
filtered query -- bounded by `--consistency-timeout` / `--poll-interval` --
until exactly one match appears, never recreating the reference or
candidate embedding and never repeating the upsert while it waits. More
than one match is treated as a malformed calibration state and fails
immediately, without retrying. If no match ever becomes visible, the script
raises a visibility timeout error and still cleans up the temporary
namespace before exiting.

The script's `calibration_interpretation` output labels each pair
positive/duplicate-like (`A_likely_paraphrase`, `B_related_preference`) or
negative/should-remain-separate (`C_potential_conflict`, `D_unrelated`) by
its fixed category -- never by its own score -- then reports
`minimum_positive_score`, `maximum_negative_score`, and whether
`separation_exists` (every negative score below every positive score) for
*this run*. When it does, `candidate_threshold_interval` is the half-open
interval `(maximum_negative_score, minimum_positive_score]` implied by that
run's numbers. When it doesn't, `candidate_threshold_interval` is `null` and
the note says plainly that no single cosine threshold separates that run's
examples -- the script never fabricates a recommendation from data that
doesn't support one.

One live calibration run against `text-embedding-3-small` did show
separation: both positive-labeled pairs scored higher than both
negative-labeled pairs, producing an approximate working interval of
roughly **0.40 to 0.51**. `MEMORY_SIMILARITY_THRESHOLD` defaults to **0.50**
in `.env.example` for that reason -- it sits inside that observed interval,
close to the boundary shared with the lowest-scoring positive example. The
report's `selected_project_threshold` field always reflects the live,
currently configured `Settings.MEMORY_SIMILARITY_THRESHOLD` value (never a
value hardcoded independently of it), and calibration never edits `.env`
itself -- choosing and applying a threshold stays a manual, human step.

This default is specific to the current embedding model
(`text-embedding-3-small`) and to this project's small, fixed calibration
scenario -- it is not a universal constant. A single cosine threshold also
cannot be expected to distinguish every merely *related* fact from a true
duplicate in every case (see `C_potential_conflict`, which is related but
should never be silently merged with its reference). Recalibrate -- rerun
`calibrate_similarity.py` and re-review its interpretation -- whenever the
embedding model changes, or whenever the memory deduplication policy
changes in a way that could shift what "duplicate enough" means.

### Smoke test: end-to-end MemoryService validation

`scripts/smoke_test_memory.py` runs `remember` / `recall` / `forget_user`
through a synthetic Telegram user ID (default `900000001`, a placeholder
that is never a real Telegram user). It cleans that user's namespace both
before and after the run, so it never leaves data behind and never touches
another namespace.

```bash
python scripts/smoke_test_memory.py
python scripts/smoke_test_memory.py --require-semantic-skip
```

Without `--require-semantic-skip`, the script does not fail just because
the paraphrase step was classified as a new memory instead of a semantic
duplicate under the current threshold -- that classification is exactly
what calibration is for. With the flag, the paraphrase step **must** be
classified as `skipped/semantic_duplicate`, or the script fails.

Every read-after-write and read-after-delete boundary is bounded-polled
before the script trusts it: the first memory must become fetch-visible
before the exact-duplicate check, and query-visible before the paraphrase
step; recall must return at least one result before it is required, and
recall after `forget_user` must become empty (a single stale, non-empty
recall result right after deletion is not treated as a cleanup failure).
The `finally`-block safety net still always runs; if the main scenario
already failed and the safety net's own cleanup also times out, the
original scenario failure is what the script reports -- the safety net's
failure is logged, never substituted in its place.

### Still pending

Telegram integration and chat completion remain unimplemented. These two
scripts validate the storage and policy layers against real Pinecone/OpenAI
credentials; they do not exercise any Telegram code path.

## Requirements

- Python 3.11 or newer.

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate      # Windows
source .venv/bin/activate   # macOS / Linux

pip install -e ".[dev]"
```

## Configuration

Copy `.env.example` to `.env` and fill in real values. Never commit `.env`.

```bash
copy .env.example .env   # Windows
cp .env.example .env     # macOS / Linux
```

See `.env.example` for the full list of supported variables, including
Pinecone, OpenAI, Telegram, and memory-related settings.

## Development checks

```bash
pytest
ruff check .
```

## Privacy principles

- Each Telegram user's memories are isolated in a separate Pinecone
  namespace.
- Bot responses are never stored as user memory -- only user-provided
  statements are eligible for storage.
- No secrets, tokens, or API keys are stored in this repository.
- Deletion of a user's memory will be supported.
- Semantic similarity alone will never silently overwrite an existing
  memory; overwriting requires an explicit, deliberate action.
