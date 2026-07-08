"""Инструменты Haystack-агента: погода, валюта, время, праздники и данные PyPI.

Каждый инструмент поверх HTTP оборачивает один публичный API, не требующий
ключа, ограниченным таймаутом и универсальным путём обработки ошибок --
сетевой сбой, таймаут, не-2xx ответ или нечитаемый upstream-payload никогда
не прокидываются наружу как исключение из вызова инструмента.
``get_current_time`` -- единственное исключение: он использует только
стандартный модуль ``zoneinfo``, не выполняет I/O и поэтому не нуждается в
таймауте -- но следует тем же соглашениям по fallback и логированию, что и
остальные.

Каждый инструмент при неудаче возвращает короткую, понятную человеку
fallback-строку, которая становится обычным содержимым tool-call, на которое
агент может отреагировать без падения (например, извиниться и продолжить
разговор), а не аварийно завершиться.

Каждый вызов инструмента логируется только с его (уже недоверенными, но не
секретными) входными аргументами и результатом -- никогда с полными телами
ответов -- в соответствии с остальной практикой безопасного логирования в
этом проекте.

``get_country_info`` (restcountries.com), ``get_wikipedia_summary``
(Wikipedia REST summary API), ``get_instant_answer`` (DuckDuckGo Instant
Answer API) и прежний инструмент с фактами о числах были исключены из
активного набора после нестабильного поведения в живом Telegram QA или
низкой практической ценности. Их заменили ``get_public_holidays``
(Nager.Date) и ``get_pypi_package_info`` (PyPI JSON API) как более
стабильные и полезные альтернативы.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from functools import lru_cache
from typing import Any, Final
from urllib.parse import quote
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import httpx
from babel import Locale
from haystack.tools import Tool

logger = logging.getLogger(__name__)

_HTTP_TIMEOUT: Final = 10.0

_GEOCODING_URL: Final = "https://geocoding-api.open-meteo.com/v1/search"
_FORECAST_URL: Final = "https://api.open-meteo.com/v1/forecast"
_EXCHANGE_RATE_URL: Final = "https://open.er-api.com/v6/latest/{base}"
_NAGER_HOLIDAYS_URL: Final = "https://date.nager.at/api/v4/Holidays/{country_code}/{year}"
_PYPI_PACKAGE_INFO_URL: Final = "https://pypi.org/pypi/{package_name}/json"
_PYPI_PROJECT_PAGE_URL: Final = "https://pypi.org/project/{package_name}/"
_PYPI_HEADERS: Final[dict[str, str]] = {
    "Accept": "application/json",
    "User-Agent": "telegram-vector-memory-bot/0.1 educational project",
}

# Фиксированный набор распространённых алиасов городов -> ключи IANA timezone.
# Намеренно небольшой и явный, а не полноценная база городов: get_current_time
# также принимает любое корректное имя IANA timezone напрямую (например,
# 'Europe/Helsinki', 'UTC'), так что эта таблица лишь сглаживает горстку
# названий городов, которые пользователь может ввести вместо самого ключа зоны.
_CITY_TIMEZONE_ALIASES: Final[dict[str, str]] = {
    "helsinki": "Europe/Helsinki",
    "moscow": "Europe/Moscow",
    "london": "Europe/London",
    "new york": "America/New_York",
    "tokyo": "Asia/Tokyo",
    "berlin": "Europe/Berlin",
    "paris": "Europe/Paris",
}

# Open-Meteo geocoding не всегда надёжно находит город по русскому названию
# напрямую, поэтому небольшой фиксированный список распространённых русских
# названий городов сопоставляется с английским перед geocoding-запросом. По
# аналогии с _CITY_TIMEZONE_ALIASES выше.
_WEATHER_CITY_ALIASES: Final[dict[str, str]] = {
    "хельсинки": "Helsinki",
    "москва": "Moscow",
    "лондон": "London",
    "токио": "Tokyo",
    "берлин": "Berlin",
    "париж": "Paris",
    "нью-йорк": "New York",
}

# Nager.Date принимает ISO alpha-2 коды. Канонические названия стран
# резолвятся из Babel territory data для ru/en, а здесь оставлены только
# разговорные или неканонические формы, которые хотелось бы поддержать явно.
_COUNTRY_CODE_SPECIAL_ALIASES: Final[dict[str, str]] = {
    "сша": "US",
    "америка": "US",
    "англия": "GB",
    "великобритания": "GB",
    "южная корея": "KR",
    "северная корея": "KP",
    "чехия": "CZ",
}

_PYPI_PACKAGE_ALIASES: Final[dict[str, str]] = {
    "haystack": "haystack-ai",
    "haystack ai": "haystack-ai",
    "pinecone haystack": "pinecone-haystack",
    "пакет pinecone для haystack": "pinecone-haystack",
    "айограм": "aiogram",
}


def _log_tool_call(tool_name: str, **fields: Any) -> None:
    rendered = " ".join(f"{key}={value!r}" for key, value in fields.items())
    logger.info("event=tool_call tool=%s %s", tool_name, rendered)


def _log_tool_failure(tool_name: str, exc: Exception) -> None:
    logger.warning("event=tool_call_failed tool=%s error_type=%s", tool_name, type(exc).__name__)


def _log_malformed_payload(tool_name: str, detail: str) -> None:
    """Залогировать JSON-валидный, но неверной формы upstream-payload как сбой инструмента.

    Использует синтетический ``ValueError``, чтобы залогированный
    ``error_type`` совпадал с ветками ошибок декодирования JSON ниже -- оба
    случая означают "ответ upstream не удалось превратить в данные, которые
    ожидает этот инструмент", просто пойманные на разных этапах (декодирование
    против проверки формы).
    """
    _log_tool_failure(tool_name, ValueError(detail))


# Кортеж типов исключений, которые может вызвать JSON-валидный, но неверной
# формы payload, когда его форма не совпадает с ожиданиями инструмента
# (например, поле, которое должно быть объектом, оказалось строкой, или
# список пуст там, где предполагается индекс). Каждый HTTP-инструмент ниже
# перехватывает их наряду с ошибками httpx/декодирования JSON, так что
# неожиданная форма upstream-ответа никогда не может вырваться наружу как
# необработанное исключение.
_SHAPE_ERRORS: Final = (AttributeError, TypeError, KeyError, IndexError)


def _normalize_alias_lookup_key(value: str) -> str:
    """Нормализовать текст для lookup по небольшим таблицам алиасов."""
    return " ".join(value.replace("Ё", "Е").replace("ё", "е").split()).casefold()


def _is_alpha2_country_code(value: str) -> bool:
    stripped = value.strip()
    return len(stripped) == 2 and stripped.isascii() and stripped.isalpha()


@lru_cache(maxsize=1)
def _build_country_name_lookup() -> dict[str, str]:
    """Построить lookup локализованных названий стран -> ISO alpha-2 кода из Babel."""
    lookup: dict[str, str] = {}
    ambiguous_keys: set[str] = set()

    for locale_name in ("ru", "en"):
        territories = Locale.parse(locale_name).territories
        for territory_code, localized_name in territories.items():
            if not isinstance(territory_code, str) or not _is_alpha2_country_code(territory_code):
                continue
            if not isinstance(localized_name, str):
                continue

            normalized_name = _normalize_alias_lookup_key(localized_name)
            if not normalized_name:
                continue

            alpha2_code = territory_code.upper()
            existing = lookup.get(normalized_name)
            if existing is None:
                lookup[normalized_name] = alpha2_code
            elif existing != alpha2_code:
                ambiguous_keys.add(normalized_name)

    for ambiguous_key in ambiguous_keys:
        lookup.pop(ambiguous_key, None)

    lookup.update(_COUNTRY_CODE_SPECIAL_ALIASES)
    return lookup


def _normalize_weather_city(city: str) -> str:
    """Сопоставить распространённое русское название города с английским для Open-Meteo.

    Для всего, чего нет в ``_WEATHER_CITY_ALIASES`` (включая любое английское
    название), возвращает *city* без изменений.
    """
    return _WEATHER_CITY_ALIASES.get(_normalize_alias_lookup_key(city), city)


def get_current_weather(city: str) -> str:
    """Вернуть текущую температуру и скорость ветра для *city* через Open-Meteo."""
    _log_tool_call("get_current_weather", city=city)
    if not isinstance(city, str) or not city.strip():
        return "Не удалось определить погоду: название города не может быть пустым."

    query = city.strip()
    lookup_city = _normalize_weather_city(query)
    unreadable = f"Не удалось определить погоду для {query!r}: получен нечитаемый ответ."
    try:
        with httpx.Client(timeout=_HTTP_TIMEOUT, follow_redirects=True) as client:
            geo_response = client.get(
                _GEOCODING_URL, params={"name": lookup_city, "count": 1, "format": "json"}
            )
            geo_response.raise_for_status()
            geo_data = geo_response.json()

            if not isinstance(geo_data, dict):
                _log_malformed_payload(
                    "get_current_weather", "geocoding response was not an object"
                )
                return unreadable

            results = geo_data.get("results")
            if not isinstance(results, list) or not results:
                return f"Не удалось определить погоду: город {query!r} не найден."

            location = results[0]
            if not isinstance(location, dict):
                _log_malformed_payload(
                    "get_current_weather", "geocoding result entry was not an object"
                )
                return unreadable

            latitude = location.get("latitude")
            longitude = location.get("longitude")
            if not isinstance(latitude, int | float) or not isinstance(longitude, int | float):
                _log_malformed_payload(
                    "get_current_weather", "geocoding result was missing numeric coordinates"
                )
                return f"Не удалось определить погоду: не найдены координаты для {query!r}."

            forecast_response = client.get(
                _FORECAST_URL,
                params={"latitude": latitude, "longitude": longitude, "current_weather": "true"},
            )
            forecast_response.raise_for_status()
            forecast_data = forecast_response.json()

            if not isinstance(forecast_data, dict):
                _log_malformed_payload("get_current_weather", "forecast response was not an object")
                return unreadable

            current = forecast_data.get("current_weather")
            if not isinstance(current, dict):
                _log_malformed_payload(
                    "get_current_weather", "forecast response was missing current_weather"
                )
                return f"Не удалось определить погоду для {query!r}: нет данных о текущей погоде."

            temperature = current.get("temperature")
            wind_speed = current.get("windspeed")
            resolved_name = location.get("name", query)
            country = location.get("country")
    except httpx.HTTPError as exc:
        _log_tool_failure("get_current_weather", exc)
        return f"Не удалось определить погоду для {query!r}: сервис погоды сейчас недоступен."
    except (ValueError, *_SHAPE_ERRORS) as exc:
        _log_tool_failure("get_current_weather", exc)
        return unreadable

    location_label = f"{resolved_name}, {country}" if country else resolved_name

    return (
        f"Текущая погода в {location_label}: температура {temperature}°C, "
        f"скорость ветра {wind_speed} км/ч."
    )


def convert_currency(amount: float, from_currency: str, to_currency: str) -> str:
    """Сконвертировать *amount* из *from_currency* в *to_currency* по актуальному курсу."""
    _log_tool_call(
        "convert_currency", amount=amount, from_currency=from_currency, to_currency=to_currency
    )
    if not isinstance(from_currency, str) or not from_currency.strip():
        return "Не удалось конвертировать валюту: код исходной валюты не может быть пустым."
    if not isinstance(to_currency, str) or not to_currency.strip():
        return "Не удалось конвертировать валюту: код целевой валюты не может быть пустым."
    if isinstance(amount, bool) or not isinstance(amount, int | float):
        return "Не удалось конвертировать валюту: сумма должна быть числом."

    base = from_currency.strip().upper()
    target = to_currency.strip().upper()
    unreadable = "Не удалось конвертировать валюту: получен нечитаемый ответ."

    try:
        with httpx.Client(timeout=_HTTP_TIMEOUT, follow_redirects=True) as client:
            response = client.get(_EXCHANGE_RATE_URL.format(base=quote(base)))
            response.raise_for_status()
            data = response.json()

        if not isinstance(data, dict):
            _log_malformed_payload("convert_currency", "exchange rate response was not an object")
            return unreadable

        if data.get("result") != "success":
            return f"Не удалось конвертировать валюту: нет курсов для базовой валюты {base!r}."

        rates = data.get("rates")
        if not isinstance(rates, dict) or target not in rates:
            return f"Не удалось конвертировать валюту: нет курса для {base!r} -> {target!r}."

        rate = rates[target]
        if isinstance(rate, bool) or not isinstance(rate, int | float):
            _log_malformed_payload("convert_currency", "exchange rate value was not numeric")
            return f"Не удалось конвертировать валюту: некорректный курс для {target!r}."

        converted = amount * rate
    except httpx.HTTPError as exc:
        _log_tool_failure("convert_currency", exc)
        return f"Не удалось конвертировать валюту: сервис курсов валют недоступен для {base!r}."
    except (ValueError, *_SHAPE_ERRORS) as exc:
        _log_tool_failure("convert_currency", exc)
        return unreadable

    return f"{amount} {base} = {converted:.2f} {target} (курс: 1 {base} = {rate} {target})."


def _normalize_country_code(country: str) -> str:
    """Сопоставить имя страны на русском/английском или ISO-код с alpha-2 кодом."""
    normalized_query = _normalize_alias_lookup_key(country)
    if not normalized_query:
        return ""

    if _is_alpha2_country_code(normalized_query):
        return normalized_query.upper()

    return _build_country_name_lookup().get(normalized_query, "")


def get_public_holidays(country: str, year: int | None = None) -> str:
    """Вернуть список праздников страны из Nager.Date за *year*.

    *country* должен быть названием страны в именительном падеже на русском
    или английском либо ISO 3166-1 alpha-2 кодом. Если *year* не передан,
    используется текущий год (UTC).
    """
    _log_tool_call("get_public_holidays", country=country, year=year)
    if not isinstance(country, str) or not country.strip():
        return "Не удалось найти праздники: название страны не может быть пустым."
    if year is not None and (isinstance(year, bool) or not isinstance(year, int)):
        return "Не удалось найти праздники: год должен быть целым числом."

    query = country.strip()
    country_code = _normalize_country_code(query)
    if not country_code:
        _log_tool_failure("get_public_holidays", LookupError(query))
        return (
            f"Не удалось найти праздники: не удалось распознать страну {query!r}. "
            "Передайте название страны на русском или английском в именительном падеже "
            "или ISO-код alpha-2."
        )
    resolved_year = year if year is not None else datetime.now(UTC).year
    unreadable = f"Не удалось найти праздники для {query!r}: получен нечитаемый ответ."

    try:
        with httpx.Client(timeout=_HTTP_TIMEOUT, follow_redirects=True) as client:
            response = client.get(
                _NAGER_HOLIDAYS_URL.format(country_code=country_code, year=resolved_year)
            )
            if response.status_code == 404:
                _log_tool_failure("get_public_holidays", LookupError(query))
                return (
                    "Не удалось найти праздники: сервис Nager.Date не предоставляет данные "
                    f"для страны {query!r}."
                )
            response.raise_for_status()
            data = response.json()

        if not isinstance(data, list):
            _log_malformed_payload("get_public_holidays", "holidays response was not a list")
            return unreadable

        if not data:
            return (
                "Не удалось найти праздники: сервис Nager.Date не предоставляет данные "
                f"для страны {query!r} в {resolved_year} году."
            )

        lines: list[str] = []
        for entry in data:
            if not isinstance(entry, dict):
                _log_malformed_payload("get_public_holidays", "holiday entry was not an object")
                return unreadable

            date = entry.get("date")
            if not isinstance(date, str) or not date:
                _log_malformed_payload("get_public_holidays", "holiday entry missing date")
                return unreadable

            local_name = entry.get("localName")
            name = entry.get("name")
            label = local_name if isinstance(local_name, str) and local_name else name
            if not isinstance(label, str) or not label:
                label = "праздник"

            if isinstance(name, str) and name and name != label:
                lines.append(f"{date} — {label} ({name})")
            else:
                lines.append(f"{date} — {label}")
    except httpx.HTTPError as exc:
        _log_tool_failure("get_public_holidays", exc)
        return (
            f"Не удалось найти праздники для {query!r}: сервис данных о праздниках "
            "сейчас недоступен."
        )
    except (ValueError, *_SHAPE_ERRORS) as exc:
        _log_tool_failure("get_public_holidays", exc)
        return unreadable

    holidays_text = "; ".join(lines)
    return f"Праздники в {country_code} ({resolved_year}): {holidays_text}."


def _optional_string_field(container: dict[str, Any], key: str) -> str | None:
    value = container.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{key} was not a string")
    stripped = value.strip()
    return stripped or None


def _extract_project_url(info: dict[str, Any], package_name: str) -> str:
    project_urls = info.get("project_urls")
    if project_urls is not None and not isinstance(project_urls, dict):
        raise ValueError("project_urls was not an object")

    if isinstance(project_urls, dict):
        for preferred_key in ("Homepage", "Source", "Repository", "Documentation", "Project"):
            preferred_value = project_urls.get(preferred_key)
            if preferred_value is None:
                continue
            if not isinstance(preferred_value, str):
                raise ValueError(f"project_urls[{preferred_key!r}] was not a string")
            stripped_value = preferred_value.strip()
            if stripped_value:
                return stripped_value

        for key, value in project_urls.items():
            if value is None:
                continue
            if not isinstance(value, str):
                raise ValueError(f"project_urls[{key!r}] was not a string")
            stripped_value = value.strip()
            if stripped_value:
                return stripped_value

    for field_name in ("home_page", "project_url", "package_url"):
        field_value = _optional_string_field(info, field_name)
        if field_value:
            return field_value

    return _PYPI_PROJECT_PAGE_URL.format(package_name=quote(package_name))


def _normalize_pypi_package_name(package_name: str) -> str:
    alias = _PYPI_PACKAGE_ALIASES.get(_normalize_alias_lookup_key(package_name))
    if alias is not None:
        return alias
    return package_name.strip().lower()


def get_pypi_package_info(package_name: str) -> str:
    """Вернуть текущую информацию о публичном Python-пакете из PyPI JSON API."""
    _log_tool_call("get_pypi_package_info", package_name=package_name)
    if not isinstance(package_name, str) or not package_name.strip():
        return "Не удалось получить данные PyPI: название пакета не может быть пустым."

    query = package_name.strip()
    normalized_package_name = _normalize_pypi_package_name(query)
    unreadable = (
        f"Не удалось получить данные о пакете {query!r} из PyPI: получен нечитаемый ответ."
    )

    try:
        with httpx.Client(timeout=_HTTP_TIMEOUT, follow_redirects=True) as client:
            response = client.get(
                _PYPI_PACKAGE_INFO_URL.format(package_name=quote(normalized_package_name)),
                headers=_PYPI_HEADERS,
            )
            if response.status_code == 404:
                _log_tool_failure("get_pypi_package_info", LookupError(normalized_package_name))
                return f"Пакет {query!r} не найден на PyPI."
            response.raise_for_status()
            data = response.json()

        if not isinstance(data, dict):
            _log_malformed_payload("get_pypi_package_info", "pypi response was not an object")
            return unreadable

        info = data.get("info")
        if not isinstance(info, dict):
            _log_malformed_payload("get_pypi_package_info", "pypi info was not an object")
            return unreadable

        resolved_name = _optional_string_field(info, "name")
        latest_version = _optional_string_field(info, "version")
        summary = _optional_string_field(info, "summary")
        requires_python = _optional_string_field(info, "requires_python")
        license_name = _optional_string_field(info, "license_expression") or _optional_string_field(
            info, "license"
        )
        if resolved_name is None or latest_version is None:
            _log_malformed_payload(
                "get_pypi_package_info", "pypi info was missing package name or version"
            )
            return unreadable

        project_url = _extract_project_url(info, normalized_package_name)
    except httpx.HTTPError as exc:
        _log_tool_failure("get_pypi_package_info", exc)
        return f"Не удалось получить данные о пакете {query!r} из PyPI: сервис сейчас недоступен."
    except (ValueError, *_SHAPE_ERRORS) as exc:
        _log_tool_failure("get_pypi_package_info", exc)
        return unreadable

    summary_text = summary if summary is not None else "не указано"
    requires_python_text = requires_python if requires_python is not None else "не указана"
    license_text = license_name if license_name is not None else "не указана"
    return (
        f"Пакет {resolved_name}: последняя версия {latest_version}. "
        f"Описание: {summary_text}. "
        f"Требуемая версия Python: {requires_python_text}. "
        f"Лицензия: {license_text}. "
        f"Ссылка: {project_url}."
    )


# datetime.strftime('%A') всегда выводит английское название дня недели
# независимо от локали ОС (locale.setlocale нигде в проекте не вызывается),
# поэтому фиксированная таблица с английскими ключами -- надёжный способ
# вывести его на русском, не полагаясь на установленные локали.
_WEEKDAY_NAMES_RU: Final[dict[str, str]] = {
    "Monday": "понедельник",
    "Tuesday": "вторник",
    "Wednesday": "среда",
    "Thursday": "четверг",
    "Friday": "пятница",
    "Saturday": "суббота",
    "Sunday": "воскресенье",
}


def _resolve_timezone(location: str) -> ZoneInfo | None:
    alias_key = location.casefold()
    iana_name = _CITY_TIMEZONE_ALIASES.get(alias_key, location)
    try:
        return ZoneInfo(iana_name)
    except (ZoneInfoNotFoundError, ValueError):
        return None


def get_current_time(location: str) -> str:
    """Вернуть текущие дату, время, день недели и часовой пояс для *location*.

    *location* может быть известным алиасом города (см.
    ``_CITY_TIMEZONE_ALIASES``) или именем IANA timezone напрямую (например,
    ``'Europe/Helsinki'``, ``'UTC'``). Использует только стандартный модуль
    ``zoneinfo`` -- нет ни внешнего API, ни необходимости в сетевом таймауте.
    """
    _log_tool_call("get_current_time", location=location)
    if not isinstance(location, str) or not location.strip():
        return (
            "Не удалось определить время: название города или часового пояса "
            "не может быть пустым."
        )

    query = location.strip()
    zone = _resolve_timezone(query)

    if zone is None:
        _log_tool_failure("get_current_time", ZoneInfoNotFoundError(query))
        examples = ", ".join(sorted(_CITY_TIMEZONE_ALIASES))
        return (
            f"Не удалось определить время: неизвестный город или часовой пояс {query!r}. "
            f"Поддерживаемые примеры городов: {examples}. Также можно передать имя IANA "
            "timezone напрямую, например 'Europe/Helsinki' или 'UTC'."
        )

    now = datetime.now(zone)
    weekday_ru = _WEEKDAY_NAMES_RU[now.strftime("%A")]
    return (
        f"Текущее время для {query!r} ({zone.key}): "
        f"{now.strftime('%Y-%m-%d %H:%M:%S %Z')}, {weekday_ru}."
    )


weather_tool = Tool(
    name="get_current_weather",
    description=(
        "Получить текущую погоду (температуру в градусах Цельсия, скорость ветра в км/ч) "
        "для указанного города. Используйте этот инструмент, когда пользователь спрашивает "
        "про погоду где-либо."
    ),
    parameters={
        "type": "object",
        "properties": {
            "city": {
                "type": "string",
                "description": (
                    "Название города на любом языке, например 'Helsinki' или 'Хельсинки'."
                ),
            },
        },
        "required": ["city"],
    },
    function=get_current_weather,
)

currency_tool = Tool(
    name="convert_currency",
    description=(
        "Конвертировать сумму денег из одной валюты в другую по текущему курсу обмена. "
        "Используйте этот инструмент, когда пользователь спрашивает, сколько стоит сумма "
        "в другой валюте."
    ),
    parameters={
        "type": "object",
        "properties": {
            "amount": {"type": "number", "description": "Сумма для конвертации."},
            "from_currency": {
                "type": "string",
                "description": "Код исходной валюты по ISO 4217, например 'EUR'.",
            },
            "to_currency": {
                "type": "string",
                "description": "Код целевой валюты по ISO 4217, например 'USD'.",
            },
        },
        "required": ["amount", "from_currency", "to_currency"],
    },
    function=convert_currency,
)

time_tool = Tool(
    name="get_current_time",
    description=(
        "Получить текущую дату, время, день недели и часовой пояс для города или имени "
        "IANA timezone. Используйте этот инструмент, когда пользователь спрашивает, "
        "который час или какой сегодня день где-либо."
    ),
    parameters={
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": (
                    "Название города (например, 'Helsinki', 'Moscow', 'London', 'New York', "
                    "'Tokyo', 'Berlin', 'Paris') или имя IANA timezone напрямую (например, "
                    "'Europe/Helsinki', 'UTC')."
                ),
            },
        },
        "required": ["location"],
    },
    function=get_current_time,
)

public_holidays_tool = Tool(
    name="get_public_holidays",
    description=(
        "Получить список государственных праздников страны за год (дата и название) "
        "для стран, поддерживаемых Nager.Date. Передавайте название страны на русском "
        "или английском в именительном падеже либо ISO 3166-1 alpha-2 код. Используйте "
        "этот инструмент, когда пользователь спрашивает про праздники или выходные дни "
        "в какой-либо стране."
    ),
    parameters={
        "type": "object",
        "properties": {
            "country": {
                "type": "string",
                "description": (
                    "Название страны на русском или английском в именительном падеже, "
                    "например 'Sweden' или 'Швеция', либо готовый код ISO 3166-1 alpha-2, "
                    "например 'SE'. Поддерживаются страны, доступные в Nager.Date."
                ),
            },
            "year": {
                "type": "integer",
                "description": (
                    "Год, за который нужны праздники, например 2026. Если не передан, "
                    "используется текущий год."
                ),
            },
        },
        "required": ["country"],
    },
    function=get_public_holidays,
)

pypi_package_info_tool = Tool(
    name="get_pypi_package_info",
    description=(
        "Получить актуальную информацию о публичном Python-пакете из PyPI: последнюю "
        "версию, краткое описание, требуемую версию Python, лицензию и ссылку на проект. "
        "Используйте этот инструмент, когда пользователь спрашивает про пакет, версию "
        "библиотеки или совместимость Python."
    ),
    parameters={
        "type": "object",
        "properties": {
            "package_name": {
                "type": "string",
                "description": (
                    "Название пакета на PyPI, например 'haystack-ai', 'aiogram', "
                    "'Haystack AI' или 'Pinecone Haystack'."
                ),
            },
        },
        "required": ["package_name"],
    },
    function=get_pypi_package_info,
)


def build_default_tools() -> list[Tool]:
    """Вернуть фиксированный набор инструментов, доступных Haystack-агенту."""
    return [
        weather_tool,
        currency_tool,
        time_tool,
        public_holidays_tool,
        pypi_package_info_tool,
    ]
