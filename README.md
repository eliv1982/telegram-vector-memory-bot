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
  enums describing memory records and the outcome of memory write attempts.
- Future stages will add: embedding generation, a Pinecone manager,
  deduplication logic, and the Telegram bot entry point.

## Project status

**Stage 1: project foundation and typed configuration.**

This stage only establishes the project skeleton, dependency management,
typed configuration, and domain models. There are no external integrations
yet: no Pinecone client, no OpenAI client, no Telegram bot, no embeddings,
no retrieval, and no live API calls. Those will be introduced in later
stages.

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
