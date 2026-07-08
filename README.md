# Telegram Vector Memory Bot

Telegram-бот на `aiogram 3` с долговременной векторной памятью в Pinecone и текущим live-путем генерации ответа через Haystack Agent. Перед ответом на обычное текстовое сообщение бот извлекает релевантные воспоминания пользователя, передает их модели как недоверенный контекст и при необходимости позволяет агенту вызвать один из пяти практичных инструментов: погода, курс валют, текущее время, государственные праздники и информация о Python-пакетах из PyPI.

## Возможности

- Telegram-бот на `aiogram 3` с long polling.
- Текущий live-путь генерации ответа через `HaystackAgentService` и `haystack.components.agents.Agent`.
- OpenAI-compatible API для chat completion и embeddings с поддержкой `OPENAI_BASE_URL`.
- Pinecone как долговременная память с отдельным namespace для каждого `Telegram user_id`.
- `recall` перед генерацией ответа и `remember` только после успешной отправки всех частей ответа.
- Exact и semantic deduplication для пользовательских текстовых сообщений.
- Русский системный промпт и русские descriptions инструментов; агент сам выбирает, нужен ли tool call.
- Ровно пять активных инструментов: `get_current_weather`, `convert_currency`, `get_current_time`, `get_public_holidays`, `get_pypi_package_info`.
- Безопасная обработка неизвестных команд, некорректного slash-текста и нетекстовых сообщений.
- Автоматическое разбиение длинных ответов под лимит Telegram по UTF-16 code units.
- Offline unit-тесты и отдельные smoke-скрипты для инструментов и памяти.

## Как работает запрос

```text
Telegram message
-> Pinecone recall
-> Haystack Agent
-> optional tool call
-> Telegram reply
-> Pinecone remember
```

`remember` выполняется только после того, как все части ответа успешно отправлены в Telegram. Если отправка ломается, сообщение пользователя не сохраняется в памяти.

## Инструменты агента

`build_default_tools()` регистрирует ровно пять инструментов:

| Инструмент | Назначение | Пример запроса | Источник |
|---|---|---|---|
| `get_current_weather` | Возвращает текущую температуру и скорость ветра по городу. | `Какая погода в Хельсинки?` | Open-Meteo |
| `convert_currency` | Конвертирует сумму между валютами по текущему курсу. | `Сколько будет 100 USD в EUR?` | `open.er-api.com` |
| `get_current_time` | Возвращает текущие дату, время, день недели и часовой пояс. | `Который час в Токио?` | Локальный инструмент на `zoneinfo` / `tzdata` |
| `get_public_holidays` | Возвращает государственные праздники страны за указанный год. | `Какие праздники в Индонезии в 2026 году?` | Nager.Date API v4 |
| `get_pypi_package_info` | Возвращает актуальные метаданные публичного Python-пакета. | `Какая последняя версия haystack-ai?` | PyPI JSON API (`pypi.org`) |

Короткие примечания:

- `get_current_weather` поддерживает небольшой набор русских алиасов распространенных городов, например `Хельсинки`.
- `get_current_time` принимает либо известный алиас города, либо IANA timezone, например `Europe/Helsinki` или `UTC`.
- `get_public_holidays` принимает название страны на русском или английском в именительном падеже либо ISO 3166-1 alpha-2 код. `Babel` используется, чтобы резолвить названия стран в alpha-2 коды. Инструмент работает для стран, доступных в Nager.Date.
- `get_pypi_package_info` возвращает название пакета, последнюю версию, `summary`, требуемую версию Python, лицензию и ссылку. Для нескольких частых вариантов запроса есть нормализация, например `Haystack AI` -> `haystack-ai`, `Pinecone Haystack` -> `pinecone-haystack`, `айограм` -> `aiogram`.

## Архитектура

- `PineconeManager` создает embeddings через OpenAI-compatible API, валидирует Pinecone index и выполняет namespace-scoped операции `query`, `fetch`, `upsert`, `delete` и чтение статистики.
- `MemoryPolicy` содержит чистую детерминированную логику нормализации текста, хеширования и правил deduplication.
- `MemoryService` объединяет `MemoryPolicy` и `PineconeManager`, строит namespace пользователя и реализует `remember`, `recall`, `get_memory_count` и `forget_user`.
- `HaystackAgentService` строит Haystack Agent поверх `OpenAIChatGenerator`, передает ему недоверенный контекст памяти и набор из пяти инструментов.
- `tools.py` содержит реализации и `Tool`-обертки всех пяти активных инструментов, включая HTTP timeout, fallback-ответы и валидацию payload.
- `bot.py` связывает Telegram-слой с памятью и агентом, регистрирует команды, маршрутизирует обычные и нетекстовые сообщения и вызывает `remember` только после успешной отправки ответа.
- `ChatService` сохранен и покрыт тестами как legacy-адаптер над Chat Completions API, но в текущем live-пути не используется.

## Структура проекта

```text
src/
  telegram_vector_memory_bot/
    bot.py
    chat_service.py
    config.py
    haystack_agent.py
    memory_policy.py
    memory_service.py
    models.py
    pinecone_manager.py
    tools.py
scripts/
  calibrate_similarity.py
  smoke_test_memory.py
  smoke_test_tools.py
tests/
  test_bot.py
  test_chat_service.py
  test_config.py
  test_haystack_agent.py
  test_live_scripts.py
  test_memory_policy.py
  test_memory_service.py
  test_models.py
  test_pinecone_manager.py
  test_tools.py
.env.example
pyproject.toml
README.md
```

## Технологии

- Python 3.11+
- `aiogram 3`
- `haystack-ai`
- Pinecone
- OpenAI-compatible API
- `httpx`
- `pydantic-settings`
- `Babel`
- `zoneinfo` / `tzdata`
- `pytest`
- Ruff

## Установка

PowerShell, из корня репозитория:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

`pyproject.toml` является основным источником зависимостей и dev-зависимостей проекта.

## Переменные окружения

Настройки описаны в `src/telegram_vector_memory_bot/config.py` (`Settings`) и читаются из `.env`. Шаблон доступен в `.env.example`.

| Переменная | Обязательна | По умолчанию | Назначение |
|---|---|---|---|
| `PINECONE_API_KEY` | да | нет | ключ доступа к Pinecone |
| `PINECONE_INDEX_NAME` | да | нет | имя заранее созданного Pinecone index |
| `OPENAI_API_KEY` | да | нет | ключ доступа к OpenAI-compatible API |
| `OPENAI_BASE_URL` | нет | `None` | альтернативный OpenAI-compatible endpoint |
| `OPENAI_EMBEDDING_MODEL` | нет | `text-embedding-3-small` | модель для embeddings |
| `OPENAI_CHAT_MODEL` | да | нет | chat-модель для генерации ответа |
| `TELEGRAM_BOT_TOKEN` | да | нет | токен Telegram-бота |
| `MEMORY_SIMILARITY_THRESHOLD` | нет | `0.50` | порог semantic deduplication |
| `MEMORY_TOP_K` | нет | `5` | сколько ближайших записей извлекать при `recall` |
| `MEMORY_NAMESPACE_PREFIX` | нет | `telegram-user` | префикс namespace пользователя |
| `LOG_LEVEL` | нет | `INFO` | уровень логирования |

Реальные секреты в README не приводятся и не должны попадать в Git.

## Настройка Pinecone

- Pinecone index должен быть создан заранее и находиться в состоянии `ready`.
- Метрика index должна быть `cosine`.
- Размерность index должна совпадать с embedding model. Для `text-embedding-3-small` по умолчанию это `1536`.
- Каждый Telegram-пользователь хранится в отдельном namespace вида `telegram-user-{telegram_user_id}`.
- `PineconeManager` валидирует конфигурацию index при запуске и не продолжит работу, если index не готов или настроен неверно.

## Запуск

Из корня репозитория:

```powershell
.\.venv\Scripts\python.exe -m telegram_vector_memory_bot.bot
```

Что важно при запуске:

- Команду нужно выполнять из корня репозитория, чтобы `.env` был найден корректно.
- Бот работает через long polling и не завершится сам по себе, пока не будет остановлен.
- Остановка выполняется через `Ctrl+C`.
- Нельзя держать два polling-процесса с одним и тем же `TELEGRAM_BOT_TOKEN`.

## Команды Telegram

| Команда | Что делает |
|---|---|
| `/start` | отправляет короткое приветствие |
| `/help` | показывает список команд и краткую справку |
| `/memory` | показывает количество сохраненных записей текущего пользователя |
| `/forget_me` | удаляет весь namespace текущего пользователя |

Дополнительно:

- неизвестная, но синтаксически корректная команда получает безопасную подсказку перейти к `/help`;
- некорректный slash-текст вроде `/foo-bar` молча игнорируется;
- нетекстовые сообщения получают фиксированный ответ о том, что бот понимает только текст.

## Память и дедупликация

- В текущем MVP каждое обычное текстовое сообщение пользователя считается кандидатом на сохранение.
- Exact duplicate: одинаковый нормализованный текст дает тот же `content hash` и не вставляется повторно.
- Semantic duplicate: если ближайшая запись в namespace имеет similarity не ниже `MEMORY_SIMILARITY_THRESHOLD`, новая запись пропускается.
- Если только одна из двух фраз содержит явное отрицание, они не считаются semantic duplicate и могут храниться отдельно.
- Новая память создается только когда не найден ни exact, ни semantic duplicate.
- В память сохраняются только обычные текстовые сообщения пользователя.
- Ответы ассистента, команды и нетекстовые сообщения не сохраняются.
- Противоположные утверждения могут сосуществовать как отдельные записи; текущий MVP не выполняет merge или overwrite существующей памяти.

## Безопасность и приватность

- Данные пользователей изолированы по отдельным Pinecone namespace, построенным от числового `user_id`.
- В prompt попадает только текст извлеченных воспоминаний, без `Telegram user_id`, username, vector ID, content hash, score, timestamp и без секретов.
- Retrieved memory явно маркируется как недоверенный контекст, а не инструкции.
- Агенту и legacy `ChatService` запрещено исполнять или трактовать текст из памяти как команду.
- Реальные ключи и токены не сохраняются в Pinecone и не передаются модели.
- Полные пользовательские сообщения не пишутся в обычные логи бота; для инструментов логируются только безопасные аргументы вызова и типы ошибок.
- Все HTTP-инструменты используют `httpx` с timeout `10.0`, безопасными fallback-ответами и проверкой структуры JSON на верхнем и вложенном уровне.
- Если внешний сервис недоступен или прислал malformed payload, инструмент возвращает понятное русское сообщение вместо необработанного исключения.

## Тестирование

Команды:

```powershell
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe scripts\smoke_test_tools.py
.\.venv\Scripts\python.exe scripts\smoke_test_memory.py --require-semantic-skip
```

Назначение smoke-скриптов:

- `scripts\smoke_test_tools.py` вызывает пять инструментов напрямую, без Telegram и без model routing.
- `scripts\smoke_test_memory.py --require-semantic-skip` прогоняет live-сценарий памяти через синтетический `user_id`, с очисткой namespace до и после теста.

Последний локальный запуск в этой рабочей копии 8 июля 2026:

| Команда | Результат |
|---|---|
| `.\.venv\Scripts\python.exe -m pytest` | PASS: `658 passed`, итоговое покрытие `97%` |
| `.\.venv\Scripts\python.exe -m ruff check .` | PASS: `All checks passed!` |
| `.\.venv\Scripts\python.exe -m pip check` | PASS: `No broken requirements found.` |
| `.\.venv\Scripts\python.exe scripts\smoke_test_tools.py` | не запускалось в рамках этой документационной правки; требует live-доступа к публичным API |
| `.\.venv\Scripts\python.exe scripts\smoke_test_memory.py --require-semantic-skip` | не запускалось в рамках этой документационной правки; требует настроенных Pinecone/OpenAI credentials и live-доступа |

## Ручная проверка

| Сценарий | Ожидаемый результат | Статус |
|---|---|---|
| Сохранение факта и последующий `recall` | бот использует ранее сохраненный факт в следующем ответе | PASS |
| `Какая погода в Хельсинки?` | вызывается `get_current_weather`, ответ содержит температуру и скорость ветра | PASS |
| `Сколько будет 100 USD в EUR?` | вызывается `convert_currency`, ответ содержит конвертированную сумму | PASS |
| `Который час в Токио?` | вызывается `get_current_time`, ответ содержит дату, время и день недели для `Asia/Tokyo` | PASS |
| `Какие праздники в Индонезии в 2026 году?` | страна резолвится в `ID`, ответ содержит государственные праздники Индонезии за 2026 год | PASS |
| `Какая последняя версия haystack-ai?` | вызывается `get_pypi_package_info`, ответ содержит актуальную версию и метаданные пакета | PASS |

Скриншоты ручной проверки подготовлены отдельно и не хранятся в репозитории.

## Ограничения MVP

- Сохраняются все обычные текстовые сообщения пользователя, включая вопросы, если они не отфильтрованы deduplication.
- Нет отдельного classifier, который решает, что именно стоит запоминать.
- Нет автоматического разрешения противоречий между разными фактами пользователя.
- Нельзя удалить или отредактировать одну запись памяти; доступно только удаление всего namespace через `/forget_me`.
- Бот работает через long polling, а не через webhook.
- Внешние инструменты зависят от доступности сторонних API.
- `get_public_holidays` работает только для стран, доступных в Nager.Date, хотя принимает русские и английские названия стран и ISO-коды.
