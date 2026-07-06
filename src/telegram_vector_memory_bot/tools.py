"""Tools for the Haystack agent: weather, currency, country facts, Wikipedia, time.

Every HTTP-backed tool wraps a single public, keyless API behind a bounded
timeout and a catch-all error path -- a network failure, timeout, non-2xx
response, or malformed upstream payload never raises out of a tool call.
``get_current_time`` is the one exception: it uses only the standard-library
``zoneinfo`` module, performs no I/O, and so needs no timeout -- but it still
follows the same fallback and logging conventions as the others.

Every tool returns a short, human-readable fallback string on failure, which
becomes ordinary tool-call content the agent can react to gracefully (e.g.
apologize and continue the conversation) rather than a crash.

Each tool call is logged with only its (already-untrusted, but non-secret)
input arguments and outcome -- never full response bodies -- consistent with
the rest of this project's safe-logging discipline.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Final
from urllib.parse import quote
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import httpx
from haystack.tools import Tool

logger = logging.getLogger(__name__)

_HTTP_TIMEOUT: Final = 10.0

_GEOCODING_URL: Final = "https://geocoding-api.open-meteo.com/v1/search"
_FORECAST_URL: Final = "https://api.open-meteo.com/v1/forecast"
_EXCHANGE_RATE_URL: Final = "https://open.er-api.com/v6/latest/{base}"
_RESTCOUNTRIES_URL: Final = "https://restcountries.com/v3.1/name/{name}"
_WIKIPEDIA_SUMMARY_URL: Final = "https://{language}.wikipedia.org/api/rest_v1/page/summary/{title}"

# Fixed set of common city aliases -> IANA timezone keys. Deliberately small
# and explicit rather than a full city database: get_current_time also
# accepts any IANA timezone name directly (e.g. 'Europe/Helsinki', 'UTC'),
# so this table only smooths over the handful of city names a user is
# likely to type instead of the underlying zone key.
_CITY_TIMEZONE_ALIASES: Final[dict[str, str]] = {
    "helsinki": "Europe/Helsinki",
    "moscow": "Europe/Moscow",
    "london": "Europe/London",
    "new york": "America/New_York",
    "tokyo": "Asia/Tokyo",
    "berlin": "Europe/Berlin",
    "paris": "Europe/Paris",
}


def _log_tool_call(tool_name: str, **fields: Any) -> None:
    rendered = " ".join(f"{key}={value!r}" for key, value in fields.items())
    logger.info("event=tool_call tool=%s %s", tool_name, rendered)


def _log_tool_failure(tool_name: str, exc: Exception) -> None:
    logger.warning("event=tool_call_failed tool=%s error_type=%s", tool_name, type(exc).__name__)


def _log_malformed_payload(tool_name: str, detail: str) -> None:
    """Log a JSON-valid-but-wrong-shape upstream payload as a tool failure.

    Uses a synthetic ``ValueError`` so the logged ``error_type`` matches the
    JSON-decode-failure branches below -- both represent "the upstream
    response could not be turned into the data this tool expects", just
    caught at a different stage (decoding vs. shape validation).
    """
    _log_tool_failure(tool_name, ValueError(detail))


# Tuple of exception types that a malformed-but-JSON-valid payload can trigger
# when its shape doesn't match what a tool expects (e.g. a field that should
# be an object is a string, or a list is empty where an index is assumed).
# Every HTTP tool below catches these alongside httpx/JSON-decode errors so
# an unexpected upstream shape can never escape as an unhandled exception.
_SHAPE_ERRORS: Final = (AttributeError, TypeError, KeyError, IndexError)


def get_current_weather(city: str) -> str:
    """Return current temperature and wind speed for *city* via Open-Meteo."""
    _log_tool_call("get_current_weather", city=city)
    if not isinstance(city, str) or not city.strip():
        return "Weather lookup failed: city name must not be empty."

    query = city.strip()
    unreadable = f"Weather lookup failed for {query!r}: received an unreadable response."
    try:
        with httpx.Client(timeout=_HTTP_TIMEOUT) as client:
            geo_response = client.get(
                _GEOCODING_URL, params={"name": query, "count": 1, "format": "json"}
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
                return f"Weather lookup failed: city {query!r} was not found."

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
                return f"Weather lookup failed: no coordinates found for {query!r}."

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
                return f"Weather lookup failed for {query!r}: no current weather data available."

            temperature = current.get("temperature")
            wind_speed = current.get("windspeed")
            resolved_name = location.get("name", query)
            country = location.get("country")
    except httpx.HTTPError as exc:
        _log_tool_failure("get_current_weather", exc)
        return f"Weather lookup failed for {query!r}: the weather service is unavailable right now."
    except (ValueError, *_SHAPE_ERRORS) as exc:
        _log_tool_failure("get_current_weather", exc)
        return unreadable

    location_label = f"{resolved_name}, {country}" if country else resolved_name

    return (
        f"Current weather in {location_label}: temperature {temperature}°C, "
        f"wind speed {wind_speed} km/h."
    )


def convert_currency(amount: float, from_currency: str, to_currency: str) -> str:
    """Convert *amount* from *from_currency* to *to_currency* using live exchange rates."""
    _log_tool_call(
        "convert_currency", amount=amount, from_currency=from_currency, to_currency=to_currency
    )
    if not isinstance(from_currency, str) or not from_currency.strip():
        return "Currency conversion failed: source currency code must not be empty."
    if not isinstance(to_currency, str) or not to_currency.strip():
        return "Currency conversion failed: target currency code must not be empty."
    if isinstance(amount, bool) or not isinstance(amount, int | float):
        return "Currency conversion failed: amount must be numeric."

    base = from_currency.strip().upper()
    target = to_currency.strip().upper()
    unreadable = "Currency conversion failed: received an unreadable response."

    try:
        with httpx.Client(timeout=_HTTP_TIMEOUT) as client:
            response = client.get(_EXCHANGE_RATE_URL.format(base=quote(base)))
            response.raise_for_status()
            data = response.json()

        if not isinstance(data, dict):
            _log_malformed_payload("convert_currency", "exchange rate response was not an object")
            return unreadable

        if data.get("result") != "success":
            return f"Currency conversion failed: no rates available for base currency {base!r}."

        rates = data.get("rates")
        if not isinstance(rates, dict) or target not in rates:
            return f"Currency conversion failed: no rate available for {base!r} -> {target!r}."

        rate = rates[target]
        if isinstance(rate, bool) or not isinstance(rate, int | float):
            _log_malformed_payload("convert_currency", "exchange rate value was not numeric")
            return f"Currency conversion failed: malformed rate for {target!r}."

        converted = amount * rate
    except httpx.HTTPError as exc:
        _log_tool_failure("convert_currency", exc)
        return f"Currency conversion failed: exchange rate service unavailable for {base!r}."
    except (ValueError, *_SHAPE_ERRORS) as exc:
        _log_tool_failure("convert_currency", exc)
        return unreadable

    return f"{amount} {base} = {converted:.2f} {target} (rate: 1 {base} = {rate} {target})."


def get_country_info(country: str) -> str:
    """Return a short factual summary of *country* via REST Countries."""
    _log_tool_call("get_country_info", country=country)
    if not isinstance(country, str) or not country.strip():
        return "Country lookup failed: country name must not be empty."

    query = country.strip()
    unreadable = f"Country lookup failed for {query!r}: received an unreadable response."
    try:
        with httpx.Client(timeout=_HTTP_TIMEOUT) as client:
            response = client.get(
                _RESTCOUNTRIES_URL.format(name=quote(query)),
                params={"fields": "name,capital,currencies,languages,population,region"},
            )
            if response.status_code == 404:
                return f"Country lookup failed: no country found matching {query!r}."
            response.raise_for_status()
            data = response.json()

        if not isinstance(data, list) or not data:
            return f"Country lookup failed: no country found matching {query!r}."

        entry = data[0]
        if not isinstance(entry, dict):
            _log_malformed_payload("get_country_info", "country entry was not an object")
            return unreadable

        entry_name = entry.get("name")
        name = entry_name.get("common", query) if isinstance(entry_name, dict) else query
        capital_list = entry.get("capital")
        capital = (
            capital_list[0] if isinstance(capital_list, list) and capital_list else "unknown"
        )

        currencies = entry.get("currencies")
        currency_names = (
            ", ".join(
                f"{info.get('name', code)} ({code})" if isinstance(info, dict) else str(code)
                for code, info in currencies.items()
            )
            if isinstance(currencies, dict) and currencies
            else "unknown"
        )

        languages = entry.get("languages")
        language_names = (
            ", ".join(str(value) for value in languages.values())
            if isinstance(languages, dict) and languages
            else "unknown"
        )

        population = entry.get("population", "unknown")
        region = entry.get("region", "unknown")
    except httpx.HTTPError as exc:
        _log_tool_failure("get_country_info", exc)
        return (
            f"Country lookup failed for {query!r}: the country info service is "
            "unavailable right now."
        )
    except (ValueError, *_SHAPE_ERRORS) as exc:
        _log_tool_failure("get_country_info", exc)
        return unreadable

    return (
        f"{name}: capital {capital}, region {region}, population {population}, "
        f"currencies: {currency_names}, languages: {language_names}."
    )


def get_wikipedia_summary(topic: str, language: str = "en") -> str:
    """Return a short Wikipedia summary for *topic* in *language* via the REST summary endpoint."""
    _log_tool_call("get_wikipedia_summary", topic=topic, language=language)
    if not isinstance(topic, str) or not topic.strip():
        return "Wikipedia lookup failed: topic must not be empty."

    query = topic.strip()
    lang = language.strip().lower() if isinstance(language, str) and language.strip() else "en"
    unreadable = f"Wikipedia lookup failed for {query!r}: received an unreadable response."

    try:
        with httpx.Client(timeout=_HTTP_TIMEOUT, follow_redirects=True) as client:
            response = client.get(
                _WIKIPEDIA_SUMMARY_URL.format(language=quote(lang), title=quote(query))
            )
            if response.status_code == 404:
                return f"Wikipedia lookup failed: no article found for {query!r}."
            response.raise_for_status()
            data = response.json()

        if not isinstance(data, dict):
            _log_malformed_payload("get_wikipedia_summary", "summary response was not an object")
            return unreadable

        title = data.get("title", query)
        summary = data.get("extract")
        if not isinstance(summary, str) or not summary.strip():
            return f"Wikipedia lookup failed: no summary available for {query!r}."

        content_urls = data.get("content_urls")
        source_url = None
        if isinstance(content_urls, dict):
            desktop = content_urls.get("desktop")
            if isinstance(desktop, dict):
                source_url = desktop.get("page")
    except httpx.HTTPError as exc:
        _log_tool_failure("get_wikipedia_summary", exc)
        return (
            f"Wikipedia lookup failed for {query!r}: the Wikipedia service is "
            "unavailable right now."
        )
    except (ValueError, *_SHAPE_ERRORS) as exc:
        _log_tool_failure("get_wikipedia_summary", exc)
        return unreadable

    result = f"{title}: {summary}"
    if isinstance(source_url, str) and source_url:
        result += f" (source: {source_url})"
    return result


def _resolve_timezone(location: str) -> ZoneInfo | None:
    alias_key = location.casefold()
    iana_name = _CITY_TIMEZONE_ALIASES.get(alias_key, location)
    try:
        return ZoneInfo(iana_name)
    except (ZoneInfoNotFoundError, ValueError):
        return None


def get_current_time(location: str) -> str:
    """Return the current date, time, weekday, and timezone for *location*.

    *location* may be a known city alias (see ``_CITY_TIMEZONE_ALIASES``) or
    an IANA timezone name directly (e.g. ``'Europe/Helsinki'``, ``'UTC'``).
    Uses only the standard-library ``zoneinfo`` module -- there is no
    external API and no network timeout is needed.
    """
    _log_tool_call("get_current_time", location=location)
    if not isinstance(location, str) or not location.strip():
        return "Time lookup failed: city or timezone name must not be empty."

    query = location.strip()
    zone = _resolve_timezone(query)

    if zone is None:
        _log_tool_failure("get_current_time", ZoneInfoNotFoundError(query))
        examples = ", ".join(sorted(_CITY_TIMEZONE_ALIASES))
        return (
            f"Time lookup failed: unknown city or timezone {query!r}. "
            f"Supported city examples: {examples}. You can also pass an IANA "
            "timezone name directly, e.g. 'Europe/Helsinki' or 'UTC'."
        )

    now = datetime.now(zone)
    return (
        f"Current time for {query!r} ({zone.key}): "
        f"{now.strftime('%Y-%m-%d %H:%M:%S %Z')}, {now.strftime('%A')}."
    )


weather_tool = Tool(
    name="get_current_weather",
    description=(
        "Get the current weather (temperature in Celsius, wind speed in km/h) for a named "
        "city. Use this whenever the user asks about the weather somewhere."
    ),
    parameters={
        "type": "object",
        "properties": {
            "city": {
                "type": "string",
                "description": "City name, in any language, e.g. 'Helsinki' or 'Хельсинки'.",
            },
        },
        "required": ["city"],
    },
    function=get_current_weather,
)

currency_tool = Tool(
    name="convert_currency",
    description=(
        "Convert an amount of money from one currency to another using current exchange "
        "rates. Use this whenever the user asks how much money is worth in another currency."
    ),
    parameters={
        "type": "object",
        "properties": {
            "amount": {"type": "number", "description": "The amount to convert."},
            "from_currency": {
                "type": "string",
                "description": "Source currency ISO 4217 code, e.g. 'EUR'.",
            },
            "to_currency": {
                "type": "string",
                "description": "Target currency ISO 4217 code, e.g. 'USD'.",
            },
        },
        "required": ["amount", "from_currency", "to_currency"],
    },
    function=convert_currency,
)

country_info_tool = Tool(
    name="get_country_info",
    description=(
        "Get a short factual summary about a country: capital, region, population, "
        "official currencies, and languages. Use this when the user asks about a country."
    ),
    parameters={
        "type": "object",
        "properties": {
            "country": {
                "type": "string",
                "description": "Country name, in any language, e.g. 'Finland' or 'Финляндия'.",
            },
        },
        "required": ["country"],
    },
    function=get_country_info,
)

wikipedia_summary_tool = Tool(
    name="get_wikipedia_summary",
    description=(
        "Get a short Wikipedia summary about a person, place, or topic, plus its source URL. "
        "Use this when the user asks who or what something/someone is."
    ),
    parameters={
        "type": "object",
        "properties": {
            "topic": {
                "type": "string",
                "description": "The person, place, or topic to look up on Wikipedia.",
            },
            "language": {
                "type": "string",
                "description": (
                    "Wikipedia language code matching the user's message language, "
                    "e.g. 'en' or 'ru'. Defaults to 'en' if omitted."
                ),
            },
        },
        "required": ["topic"],
    },
    function=get_wikipedia_summary,
)


time_tool = Tool(
    name="get_current_time",
    description=(
        "Get the current date, time, weekday, and timezone for a city or an IANA timezone "
        "name. Use this whenever the user asks what time or day it is somewhere."
    ),
    parameters={
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": (
                    "City name (e.g. 'Helsinki', 'Moscow', 'London', 'New York', 'Tokyo', "
                    "'Berlin', 'Paris') or an IANA timezone name (e.g. 'Europe/Helsinki', "
                    "'UTC')."
                ),
            },
        },
        "required": ["location"],
    },
    function=get_current_time,
)


def build_default_tools() -> list[Tool]:
    """Return the fixed set of tools available to the Haystack agent."""
    return [
        weather_tool,
        currency_tool,
        country_info_tool,
        wikipedia_summary_tool,
        time_tool,
    ]
