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
- Future stages will add: semantic deduplication policy, a memory service
  layer, and the Telegram bot entry point.

## Project status

**Stage 2: Pinecone infrastructure manager and embedding generation.**

Stage 1 established the project skeleton, dependency management, typed
configuration, and domain models. Stage 2 adds `PineconeManager`, a typed
infrastructure adapter responsible for:

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
network access.

`PineconeManager` is infrastructure only: it has no concept of duplicate
detection or memory-write policy, and never returns a `MemoryWriteResult`.
Deduplication and Telegram integration have not been implemented yet.

Cosine similarity scores returned from queries range from **-1 to 1**
(`VectorMatch.score`, `MemoryWriteResult.similarity_score`). The configured
duplicate *threshold* in `Settings` (`MEMORY_SIMILARITY_THRESHOLD`) remains
constrained to **0 to 1**, since deduplication only ever cares about
positive similarity.

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
