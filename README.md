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

The script's `suggested_threshold_range` output is **descriptive only** --
a single observed score never proves semantic equivalence on its own.
Choosing `MEMORY_SIMILARITY_THRESHOLD` requires human judgment across
repeated runs and real usage, not one script's numbers; it is not a
universal constant to copy verbatim into `.env`.

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
