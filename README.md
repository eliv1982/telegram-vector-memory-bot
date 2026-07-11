# Telegram Vector Memory Bot

Репозиторий содержит две связанные реализации:

- принятый v1-бот в `src/telegram_vector_memory_bot` с долгосрочной памятью и Haystack Agent;
- финальный v2-бот в `hay_v2_bot`, который добавляет загрузку PDF/DOCX, Docling-конвертацию, документный RAG и переход к принятому v1 Agent.

## Stage 6: `hay_v2_bot`

### Цель новой версии

Собрать рабочую вторую версию Telegram-бота на `aiogram 3`, которая:

- принимает PDF и DOCX;
- прогоняет файл через Docling;
- сохраняет чанки документа в персональном Pinecone namespace;
- сразу возвращает короткое русское резюме одним предложением;
- отвечает на вопросы по загруженным документам и показывает краткий блок источников;
- при нехватке документного контекста делает переход к принятому v1 Agent с памятью и инструментами.

### Архитектура v2

```text
Telegram
-> обработчик документов
-> адаптер Docling
-> пайплайн индексации
-> Pinecone

вопрос
-> RAG-пайплайн
-> ответ по контексту документа + источники

если документного контекста недостаточно
-> переход к принятому v1 Haystack Agent
```

В `hay_v2_bot` используются три реальные Haystack Pipelines:

- пайплайн индексации: эмбеддинг и запись нормализованных Docling-чанков в Pinecone;
- пайплайн резюме: одно безопасное предложение после успешной загрузки;
- RAG-пайплайн: поиск и генерация только по документному namespace пользователя.

### Поддерживаемые форматы

- PDF
- DOCX

### Разделение namespace

- v1 memory namespace: `telegram-user-*`
- v2 document namespace: `telegram-documents-user-*`

Загрузка документов не дублирует их содержимое в v1 memory namespace.

## Возможности

- Telegram-бот на `aiogram 3` с long polling.
- Текущий live-путь генерации ответа через `HaystackAgentService` и `haystack.components.agents.Agent`.
- OpenAI-compatible API для chat completion и embeddings с поддержкой `OPENAI_BASE_URL`.
- Pinecone как долговременная память с отдельным namespace для каждого пользователя.
- `recall` перед генерацией ответа и `remember` только после успешной отправки всех частей ответа.
- Exact и semantic deduplication для пользовательских текстовых сообщений.
- Русский системный промпт и русские descriptions инструментов; агент сам выбирает, нужен ли tool call.
- Ровно пять активных инструментов: `get_current_weather`, `convert_currency`, `get_current_time`, `get_public_holidays`, `get_pypi_package_info`.
- Безопасная обработка неизвестных команд, некорректного slash-текста и нетекстовых сообщений.
- Автоматическое разбиение длинных ответов под лимит Telegram по UTF-16 code units.
- Локальные unit-тесты и отдельные smoke-скрипты для инструментов и памяти.

### Сценарий работы

1. Пользователь отправляет PDF или DOCX.
2. Бот подтверждает приём файла.
3. Docling нормализует документ, а пайплайн индексации пишет чанки в Pinecone.
4. Бот отправляет одно предложение-резюме.
5. Пользователь задаёт вопрос обычным текстом.
6. Бот сначала пытается ответить по документам и, если получил ответ по контексту документа, добавляет блок `Источники`.
7. Если документов недостаточно, бот один раз переходит к принятому v1 Agent с памятью и инструментами.

### Ограничения по безопасности

- Первый живой запуск Docling может инициализировать или скачать model artifacts.
- Для PDF в источниках показываются страницы, когда page metadata доступна.
- Для DOCX источники откатываются к человекочитаемым номерам фрагментов.
- Поддерживаются только PDF и DOCX.

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
- `get_pypi_package_info` возвращает название пакета, последнюю версию, краткое описание (`summary`), требуемую версию Python, лицензию и ссылку. Для нескольких частых вариантов запроса есть нормализация, например `Haystack AI` -> `haystack-ai`, `Pinecone Haystack` -> `pinecone-haystack`, `айограм` -> `aiogram`.

## Архитектура

- `PineconeManager` создает embeddings через OpenAI-compatible API, валидирует Pinecone index и выполняет namespace-scoped операции `query`, `fetch`, `upsert`, `delete` и чтение статистики.
- `MemoryPolicy` содержит чистую детерминированную логику нормализации текста, хеширования и правил deduplication.
- `MemoryService` объединяет `MemoryPolicy` и `PineconeManager`, строит namespace пользователя и реализует `remember`, `recall`, `get_memory_count` и `forget_user`.
- `HaystackAgentService` строит Haystack Agent поверх `OpenAIChatGenerator`, передает ему недоверенный контекст памяти и набор из пяти инструментов.
- `tools.py` содержит реализации и `Tool`-обертки всех пяти активных инструментов, включая HTTP timeout, резервные ответы и валидацию payload.
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

`pyproject.toml` является основным источником зависимостей и dev-зависимостей проекта. На Windows для Stage 6 удобно использовать короткое локальное или внешнее окружение, чтобы не упираться в длинные пути к пакетам Docling/Haystack.

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
| `DOCUSCOPE_MAX_FILE_BYTES` | нет | `20971520` | максимальный размер входного PDF/DOCX для v2 |
| `DOCUSCOPE_MAX_CHUNKS_PER_DOCUMENT` | нет | `2000` | верхняя граница числа Docling-чанков |
| `DOCUSCOPE_EMBEDDING_DIMENSIONS` | нет | `1536` | ожидаемая размерность Pinecone index для document RAG |
| `DOCUSCOPE_RETRIEVAL_TOP_K` | нет | `4` | сколько document chunks поднимать в RAG |
| `DOCUSCOPE_MAX_SUMMARY_CHARS` | нет | `12000` | предел контекста для пайплайна резюме |
| `DOCUSCOPE_MAX_QUESTION_CHARS` | нет | `4000` | предел длины вопроса для document RAG |

Реальные секреты в README не приводятся и не должны попадать в Git.

## Настройка Pinecone

- Pinecone index должен быть создан заранее и находиться в состоянии `ready`.
- Метрика index должна быть `cosine`.
- Размерность index должна совпадать с embedding model. Для `text-embedding-3-small` по умолчанию это `1536`.
- Каждый пользователь хранится в отдельном memory namespace с префиксом `telegram-user-`, а документы v2 — в отдельном namespace с префиксом `telegram-documents-user-`.
- `PineconeManager` валидирует конфигурацию index при запуске и не продолжит работу, если index не готов или настроен неверно.

## Запуск

Из корня репозитория:

```powershell
.\.venv\Scripts\python.exe -m hay_v2_bot
```

Что важно при запуске:

- Команду нужно выполнять из корня репозитория, чтобы `.env` был найден корректно.
- Бот работает через long polling и не завершится сам по себе, пока не будет остановлен.
- Остановка выполняется через `Ctrl+C`.
- Нельзя держать два polling-процесса с одним и тем же `TELEGRAM_BOT_TOKEN`.
- `python -m hay_v2_bot.main` использует тот же runtime, что и `python -m hay_v2_bot`.

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
- PDF/DOCX сообщения в v2 обрабатываются отдельно: файл сохраняется, получает резюме и потом участвует в ответах по контексту документа.

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

- Данные пользователей изолированы по отдельным Pinecone namespace и не смешиваются между собой.
- В prompt попадает только текст извлеченных воспоминаний, без служебных идентификаторов, content hash, score, timestamp и без секретов.
- Извлеченная память явно маркируется как недоверенный контекст, а не инструкции.
- Агенту и legacy `ChatService` запрещено исполнять или трактовать текст из памяти как команду.
- Реальные ключи и токены не сохраняются в Pinecone и не передаются модели.
- Полные пользовательские сообщения не пишутся в обычные логи бота; для инструментов логируются только безопасные аргументы вызова и типы ошибок.
- Все HTTP-инструменты используют `httpx` с timeout `10.0`, безопасными резервными ответами и проверкой структуры JSON на верхнем и вложенном уровне.
- Если внешний сервис недоступен или прислал malformed payload, инструмент возвращает понятное русское сообщение вместо необработанного исключения.

## Тестирование

Команды:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/hay_v2_bot/test_bot_messages.py -q -p no:cacheprovider
.\.venv\Scripts\python.exe -m pytest tests/hay_v2_bot/test_bot_handlers.py -q -p no:cacheprovider
.\.venv\Scripts\python.exe -m pytest tests/hay_v2_bot/test_bot_runtime.py -q -p no:cacheprovider
.\.venv\Scripts\python.exe -m pytest tests/hay_v2_bot -q -p no:cacheprovider
.\.venv\Scripts\python.exe -m pytest -p no:cacheprovider
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m pip check
```

Назначение smoke-скриптов:

- `scripts\smoke_test_tools.py` вызывает пять инструментов напрямую, без Telegram и без model routing.
- `scripts\smoke_test_memory.py --require-semantic-skip` прогоняет live-сценарий памяти через отдельный синтетический namespace, с очисткой до и после теста.
- `scripts\smoke_test_docling.py` прогоняет один live Docling conversion path без Telegram.
- `scripts\smoke_test_document_rag.py` прогоняет live ingestion + резюме + RAG + cleanup без Telegram.

### Доступные live smoke-сценарии

- `scripts\smoke_test_docling.py` принимает путь к образцу PDF/DOCX, content type, изолированный тестовый namespace и путь к JSON-отчету.
- `scripts\smoke_test_document_rag.py` принимает те же параметры и дополнительно вопрос для проверки ответа по контексту документа.

### Итоги финальной локальной проверки

| Команда | Результат |
|---|---|
| `.\.venv\Scripts\python.exe -m pytest -p no:cacheprovider` | PASS: `876 passed`, итоговое покрытие `97%` |
| `.\.venv\Scripts\python.exe -m ruff check .` | PASS: `All checks passed!` |
| `.\.venv\Scripts\python.exe -m pip check` | PASS: `No broken requirements found.` |

## Финальная приемка v2

| Сценарий | Подтверждение | Статус |
|---|---|---|
| `/help` | справка явно упоминает поддержку PDF и DOCX | PASS |
| Загрузка PDF и русское резюме одним предложением | документ принимается, бот возвращает краткое русское резюме | PASS |
| Ответ по бюджету из PDF | ответ приходит с единственным источником: страница 1 PDF | PASS |
| Ответ по согласованию версии из PDF | ответ приходит с единственным источником: страница 2 PDF | PASS |
| Загрузка отдельного DOCX и ответ по нему | ответ приходит с источником из DOCX-фрагмента | PASS |
| Переход к legacy Agent через инструмент текущего времени | при отсутствии документного ответа срабатывает переход к v1 Agent | PASS |

Проверка live weather сейчас не отмечается как PASS: во время финальной приемки внешний погодный сервис вернул внешнюю ошибку.

## Ограничения MVP

- Сохраняются все обычные текстовые сообщения пользователя, включая вопросы, если они не отфильтрованы deduplication.
- Нет отдельного classifier, который решает, что именно стоит запоминать.
- Нет автоматического разрешения противоречий между разными фактами пользователя.
- Нельзя удалить или отредактировать одну запись памяти; доступно только удаление всего namespace через `/forget_me`.
- Бот работает через long polling, а не через webhook.
- Внешние инструменты зависят от доступности сторонних API и могут временно возвращать внешние ошибки.
- `get_public_holidays` работает только для стран, доступных в Nager.Date, хотя принимает русские и английские названия стран и ISO-коды.
