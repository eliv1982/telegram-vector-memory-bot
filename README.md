# Telegram Docling RAG Bot

Telegram-бот на `aiogram 3`, который принимает PDF и DOCX, прогоняет документы через Docling, сохраняет структурированные чанки в Pinecone и отвечает по контексту загруженных файлов. После загрузки бот автоматически возвращает короткое русское резюме одним предложением, а при ответе показывает компактный блок источников.

Основная реализация находится в `hay_v2_bot`. Принятый v1-бот остаётся в `src/telegram_vector_memory_bot`: новая версия переиспользует его память и возможности Agent, когда документного контекста недостаточно.

## Возможности

- загрузка PDF и DOCX через Telegram;
- конвертация документов через Docling и структурированное разбиение на чанки;
- удалённые embeddings через OpenAI-compatible API;
- хранение документных чанков в Pinecone;
- автоматическое русское резюме одним предложением после загрузки;
- ответы по контексту документа с краткими источниками;
- переход к v1 Agent с памятью и инструментами, если документов недостаточно;
- команды `/start`, `/help`, `/memory` и `/forget_me`;
- long polling и безопасная обработка неизвестных команд, нетекстовых сообщений и длинных ответов.

## Архитектура

```text
Telegram document
-> Docling adapter
-> ingestion pipeline
-> Pinecone

document question
-> text embedder
-> Pinecone retriever
-> prompt builder
-> chat generator
-> answer + sources

insufficient document context
-> accepted v1 Haystack Agent
```

`hay_v2_bot` отвечает за документный сценарий: приём файла, конвертацию, индексацию, генерацию резюме и ответы по документам. Если поиск не даёт достаточного контекста, управление один раз передаётся в принятый v1 Agent из `src/telegram_vector_memory_bot`, который использует память и существующие инструменты.

## Структура проекта

```text
hay_v2_bot/
  adapters/
  bot/
  components/
  models/
  pipelines/
  services/
  storage/
  config.py
  main.py
  __main__.py

src/telegram_vector_memory_bot/
scripts/
tests/hay_v2_bot/
```

- `hay_v2_bot/` содержит модульную реализацию Telegram-бота с документным RAG.
- `src/telegram_vector_memory_bot/` хранит принятую v1-реализацию памяти и Agent.
- `scripts/` содержит вспомогательные smoke-test сценарии.
- `tests/hay_v2_bot/` покрывает документный контур, Telegram-слой и конфигурацию новой версии.

## Обработка документов

1. Пользователь отправляет PDF или DOCX в Telegram.
2. Бот валидирует тип и размер файла, затем сохраняет его локально на время обработки.
3. Docling нормализует документ и разбивает его на структурированные текстовые чанки.
4. Для каждого чанка сохраняются полезные метаданные, включая имя файла, индекс фрагмента и номер страницы для PDF, если он доступен.
5. Ingestion pipeline строит embeddings и записывает чанки в Pinecone.
6. После успешной индексации бот отправляет одно русское предложение-резюме.
7. При следующем вопросе бот сначала ищет ответ по документам, а затем при необходимости делает переход к v1 Agent.

Источники в пользовательском ответе выводятся компактно: для PDF показывается страница, для DOCX — номер фрагмента.

## Haystack Pipelines

В проекте используются три реальные Haystack Pipeline:

- `ingestion` — получает нормализованные Docling-чанки, строит embeddings и записывает их в Pinecone;
- `summary` — формирует короткое безопасное русское резюме после загрузки документа;
- `RAG` — извлекает релевантные документные чанки, собирает prompt и генерирует ответ по контексту документа.

## Хранение данных в Pinecone

- используется существующий Pinecone index с метрикой `cosine`;
- ожидаемая размерность index — `1536`;
- v1 memory namespace: `telegram-user-{user_id}`;
- v2 documents namespace: `telegram-documents-user-{user_id}`;
- содержимое документов не дублируется в v1 memory namespace;
- память обычных текстовых сообщений и документные чанки остаются разделёнными.

## Установка

`pyproject.toml` — основной источник зависимостей проекта. Дублировать установку через `requirements.txt` не требуется.

Пример установки из корня репозитория:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

На Windows может быть удобнее использовать короткое внешнее виртуальное окружение, потому что зависимости Docling иногда упираются в длинные пути.

## Настройка окружения

README не содержит секретов. Минимально нужны такие переменные:

- `PINECONE_API_KEY`
- `PINECONE_INDEX_NAME`
- `OPENAI_API_KEY`
- `OPENAI_EMBEDDING_MODEL`
- `OPENAI_CHAT_MODEL`
- `TELEGRAM_BOT_TOKEN`

Опционально:

- `OPENAI_BASE_URL`
- `DOCUSCOPE_MAX_FILE_BYTES`
- `DOCUSCOPE_MAX_CHUNKS_PER_DOCUMENT`
- `DOCUSCOPE_EMBEDDING_DIMENSIONS`
- `DOCUSCOPE_RETRIEVAL_TOP_K`
- `DOCUSCOPE_MAX_SUMMARY_CHARS`
- `DOCUSCOPE_MAX_QUESTION_CHARS`

Настройки читаются из `.env`. Шаблон переменных находится в `.env.example`.

## Запуск

Запускать бота нужно из корня репозитория:

```powershell
.\.venv\Scripts\python.exe -m hay_v2_bot
```

Дополнительно:

- `python -m hay_v2_bot.main` использует тот же runtime;
- бот работает через long polling;
- процесс завершается через `Ctrl+C`;
- нельзя одновременно держать два polling-процесса с одним и тем же токеном Telegram.

## Тестирование

Базовые команды проверки:

```powershell
python -m pytest -p no:cacheprovider
python -m ruff check .
python -m pip check
```

Финально подтверждённые результаты:

| Команда | Результат |
|---|---|
| `python -m pytest -p no:cacheprovider` | `876 passed`, покрытие `97%` |
| `python -m ruff check .` | PASS |
| `python -m pip check` | PASS |

Для интеграционных smoke-test проверок в репозитории есть два сценария:

- `scripts/smoke_test_docling.py`
- `scripts/smoke_test_document_rag.py`

Они проверяют соответственно конвертацию через Docling и полный документный RAG-сценарий без интерфейса Telegram.

## Ручная проверка

| Сценарий | Результат | Статус |
|---|---|---|
| `/help` включает поддержку PDF/DOCX | подтверждено | PASS |
| Загрузка PDF и русское резюме одним предложением | подтверждено | PASS |
| Ответ по бюджету из PDF с источником на странице 1 | подтверждено | PASS |
| Ответ по версии из PDF с источником на странице 2 | подтверждено | PASS |
| Загрузка отдельного DOCX и ответ с источником по DOCX-фрагменту | подтверждено | PASS |
| Переход к v1 Agent через инструмент текущего времени | подтверждено | PASS |

## Ограничения

- поддерживаются только PDF и DOCX;
- первый запуск Docling может инициализировать model artifacts;
- номера страниц для PDF зависят от доступных метаданных;
- для DOCX в источниках используются номера фрагментов;
- бот работает через long polling;
- внешние API могут быть временно недоступны.
