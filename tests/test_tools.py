"""Модульные тесты для telegram_vector_memory_bot.tools.

Все тесты работают против фейкового httpx.Client -- ни один реальный
сетевой вызов никогда не выполняется. Каждая функция-инструмент проверяется
напрямую (успех, не найдено, сбой транспорта), а каждая обёртка Tool
проверяется на наличие ожидаемого name/function/required-parameters, от
которых зависит агент.
"""

from __future__ import annotations

import logging
import re
from typing import Any

import httpx
import pytest
from haystack.tools import Tool

from telegram_vector_memory_bot import tools

_UNSET: Any = object()


class FakeResponse:
    def __init__(self, *, status_code: int = 200, json_data: Any = _UNSET) -> None:
        self.status_code = status_code
        # Настоящий sentinel (а не просто `None`) отличает "json_data не
        # передан -> по умолчанию {}" от "тест хочет, чтобы .json() буквально
        # вернул None/список/строку", что нужно тестам malformed-payload ниже.
        self._json_data = {} if json_data is _UNSET else json_data

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            request = httpx.Request("GET", "https://example.invalid")
            response = httpx.Response(self.status_code, request=request)
            raise httpx.HTTPStatusError("error", request=request, response=response)

    def json(self) -> Any:
        return self._json_data


class FakeHttpxClient:
    """Фейковая замена httpx.Client -- возвращает заскриптованные ответы, без сетевого I/O."""

    def __init__(
        self,
        responses: list[FakeResponse] | None = None,
        *,
        exception: Exception | None = None,
    ):
        self._responses = iter(responses or [])
        self._exception = exception
        self.calls: list[dict[str, Any]] = []
        # Заполняется _install_fake_client теми же kwargs, что тестируемый
        # инструмент передал в httpx.Client(...), так что тесты могут
        # проверять, например, timeout/follow_redirects без обращения к
        # реальному сетевому коду.
        self.client_kwargs: dict[str, Any] = {}

    def get(self, url: str, params: dict[str, Any] | None = None, **kwargs: Any) -> FakeResponse:
        self.calls.append({"url": url, "params": params, "kwargs": kwargs})
        if self._exception is not None:
            raise self._exception
        return next(self._responses)

    def __enter__(self) -> FakeHttpxClient:
        return self

    def __exit__(self, *args: Any) -> bool:
        return False


def _install_fake_client(monkeypatch: pytest.MonkeyPatch, client: FakeHttpxClient) -> None:
    def _factory(*args: Any, **kwargs: Any) -> FakeHttpxClient:
        client.client_kwargs = kwargs
        return client

    monkeypatch.setattr(tools.httpx, "Client", _factory)


# ---------------------------------------------------------------------------
# get_current_weather
# ---------------------------------------------------------------------------


def test_get_current_weather_success(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_client = FakeHttpxClient(
        [
            FakeResponse(
                json_data={
                    "results": [
                        {
                            "name": "Helsinki",
                            "country": "Finland",
                            "latitude": 60.17,
                            "longitude": 24.94,
                        }
                    ]
                }
            ),
            FakeResponse(json_data={"current_weather": {"temperature": 5.0, "windspeed": 10.0}}),
        ]
    )
    _install_fake_client(monkeypatch, fake_client)

    result = tools.get_current_weather("Helsinki")

    assert "Helsinki" in result
    assert "Finland" in result
    assert "5.0" in result
    assert "10.0" in result
    assert fake_client.calls[0]["url"] == tools._GEOCODING_URL
    assert fake_client.calls[1]["url"] == tools._FORECAST_URL


def test_get_current_weather_normalizes_russian_helsinki_alias(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_client = FakeHttpxClient(
        [
            FakeResponse(
                json_data={
                    "results": [
                        {
                            "name": "Helsinki",
                            "country": "Finland",
                            "latitude": 60.17,
                            "longitude": 24.94,
                        }
                    ]
                }
            ),
            FakeResponse(json_data={"current_weather": {"temperature": 5.0, "windspeed": 10.0}}),
        ]
    )
    _install_fake_client(monkeypatch, fake_client)

    result = tools.get_current_weather("Хельсинки")

    assert "Helsinki" in result
    assert "Finland" in result
    assert fake_client.calls[0]["params"]["name"] == "Helsinki"


@pytest.mark.parametrize(
    ("russian_form", "expected_english"),
    [
        ("Хельсинки", "Helsinki"),
        ("Москва", "Moscow"),
        ("Лондон", "London"),
        ("Токио", "Tokyo"),
        ("Берлин", "Berlin"),
        ("Париж", "Paris"),
        ("Нью-Йорк", "New York"),
    ],
)
def test_normalize_weather_city_maps_all_documented_russian_aliases(
    russian_form: str, expected_english: str
) -> None:
    assert tools._normalize_weather_city(russian_form) == expected_english


def test_normalize_weather_city_leaves_english_names_unchanged() -> None:
    assert tools._normalize_weather_city("Helsinki") == "Helsinki"


def test_get_current_weather_uses_expected_client_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_client = FakeHttpxClient(
        [
            FakeResponse(
                json_data={
                    "results": [{"name": "Helsinki", "latitude": 60.17, "longitude": 24.94}]
                }
            ),
            FakeResponse(json_data={"current_weather": {"temperature": 5.0, "windspeed": 10.0}}),
        ]
    )
    _install_fake_client(monkeypatch, fake_client)

    tools.get_current_weather("Helsinki")

    assert fake_client.client_kwargs["timeout"] == 10.0
    assert fake_client.client_kwargs["follow_redirects"] is True


def test_get_current_weather_city_not_found(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_client = FakeHttpxClient([FakeResponse(json_data={"results": []})])
    _install_fake_client(monkeypatch, fake_client)

    result = tools.get_current_weather("Nonexistentville")

    assert "не найден" in result.lower()


def test_get_current_weather_blank_city_rejected_without_http_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _explode(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("must not perform HTTP calls for blank input")

    monkeypatch.setattr(tools.httpx, "Client", _explode)

    result = tools.get_current_weather("   ")

    assert "не может быть пустым" in result.lower()


def test_get_current_weather_transport_failure_returns_safe_fallback(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    fake_client = FakeHttpxClient(exception=httpx.ConnectTimeout("timed out"))
    _install_fake_client(monkeypatch, fake_client)

    with caplog.at_level(logging.WARNING):
        result = tools.get_current_weather("Helsinki")

    assert "недоступен" in result.lower()
    assert "event=tool_call_failed" in caplog.text
    assert "tool=get_current_weather" in caplog.text


def test_get_current_weather_missing_coordinates_returns_safe_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_client = FakeHttpxClient(
        [FakeResponse(json_data={"results": [{"name": "Helsinki"}]})]
    )
    _install_fake_client(monkeypatch, fake_client)

    result = tools.get_current_weather("Helsinki")

    assert "координаты" in result.lower()


def test_get_current_weather_missing_current_weather_returns_safe_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_client = FakeHttpxClient(
        [
            FakeResponse(
                json_data={"results": [{"name": "Helsinki", "latitude": 60.17, "longitude": 24.94}]}
            ),
            FakeResponse(json_data={}),
        ]
    )
    _install_fake_client(monkeypatch, fake_client)

    result = tools.get_current_weather("Helsinki")

    assert "нет данных о текущей погоде" in result.lower()


def test_get_current_weather_malformed_json_returns_safe_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class BrokenResponse(FakeResponse):
        def json(self) -> Any:
            raise ValueError("not json")

    fake_client = FakeHttpxClient([BrokenResponse()])
    _install_fake_client(monkeypatch, fake_client)

    result = tools.get_current_weather("Helsinki")

    assert "нечитаемый" in result.lower()


@pytest.mark.parametrize("malformed_geo_data", [[], "not-an-object", None, 42])
def test_get_current_weather_non_dict_geocoding_response_returns_safe_fallback(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, malformed_geo_data: Any
) -> None:
    fake_client = FakeHttpxClient([FakeResponse(json_data=malformed_geo_data)])
    _install_fake_client(monkeypatch, fake_client)

    with caplog.at_level(logging.WARNING):
        result = tools.get_current_weather("Helsinki")

    assert "нечитаемый" in result.lower()
    assert "event=tool_call_failed" in caplog.text
    assert "tool=get_current_weather" in caplog.text


@pytest.mark.parametrize("malformed_entry", ["not-an-object", 42, ["nested-list"], None])
def test_get_current_weather_non_dict_result_entry_returns_safe_fallback(
    monkeypatch: pytest.MonkeyPatch, malformed_entry: Any
) -> None:
    fake_client = FakeHttpxClient([FakeResponse(json_data={"results": [malformed_entry]})])
    _install_fake_client(monkeypatch, fake_client)

    result = tools.get_current_weather("Helsinki")

    assert "нечитаемый" in result.lower()


@pytest.mark.parametrize("malformed_forecast_data", [[], "not-an-object", None])
def test_get_current_weather_non_dict_forecast_response_returns_safe_fallback(
    monkeypatch: pytest.MonkeyPatch, malformed_forecast_data: Any
) -> None:
    fake_client = FakeHttpxClient(
        [
            FakeResponse(
                json_data={
                    "results": [{"name": "Helsinki", "latitude": 60.17, "longitude": 24.94}]
                }
            ),
            FakeResponse(json_data=malformed_forecast_data),
        ]
    )
    _install_fake_client(monkeypatch, fake_client)

    result = tools.get_current_weather("Helsinki")

    assert "нечитаемый" in result.lower()


# ---------------------------------------------------------------------------
# convert_currency
# ---------------------------------------------------------------------------


def test_convert_currency_success(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_client = FakeHttpxClient(
        [FakeResponse(json_data={"result": "success", "rates": {"USD": 1.1}})]
    )
    _install_fake_client(monkeypatch, fake_client)

    result = tools.convert_currency(100, "eur", "usd")

    assert "100" in result
    assert "EUR" in result
    assert "110.00" in result
    assert "USD" in result


def test_convert_currency_uses_expected_client_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_client = FakeHttpxClient(
        [FakeResponse(json_data={"result": "success", "rates": {"USD": 1.1}})]
    )
    _install_fake_client(monkeypatch, fake_client)

    tools.convert_currency(100, "EUR", "USD")

    assert fake_client.client_kwargs["timeout"] == 10.0
    assert fake_client.client_kwargs["follow_redirects"] is True


def test_convert_currency_blank_target_currency_rejected_without_http_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _explode(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("must not perform HTTP calls for blank input")

    monkeypatch.setattr(tools.httpx, "Client", _explode)

    result = tools.convert_currency(100, "EUR", "")

    assert "не может быть пустым" in result.lower()


def test_convert_currency_non_numeric_amount_rejected_without_http_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _explode(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("must not perform HTTP calls for non-numeric amount")

    monkeypatch.setattr(tools.httpx, "Client", _explode)

    result = tools.convert_currency(True, "EUR", "USD")  # bool must be rejected, not treated as 1

    assert "должна быть числом" in result.lower()


def test_convert_currency_result_not_success(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_client = FakeHttpxClient([FakeResponse(json_data={"result": "error"})])
    _install_fake_client(monkeypatch, fake_client)

    result = tools.convert_currency(100, "EUR", "USD")

    assert "нет курсов" in result.lower()


def test_convert_currency_malformed_rate_type(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_client = FakeHttpxClient(
        [FakeResponse(json_data={"result": "success", "rates": {"USD": "not-a-number"}})]
    )
    _install_fake_client(monkeypatch, fake_client)

    result = tools.convert_currency(100, "EUR", "USD")

    assert "некорректный курс" in result.lower()


def test_convert_currency_malformed_json_returns_safe_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class BrokenResponse(FakeResponse):
        def json(self) -> Any:
            raise ValueError("not json")

    fake_client = FakeHttpxClient([BrokenResponse()])
    _install_fake_client(monkeypatch, fake_client)

    result = tools.convert_currency(100, "EUR", "USD")

    assert "нечитаемый" in result.lower()


@pytest.mark.parametrize("malformed_data", [[], "not-an-object", None, 42])
def test_convert_currency_non_dict_response_returns_safe_fallback(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, malformed_data: Any
) -> None:
    fake_client = FakeHttpxClient([FakeResponse(json_data=malformed_data)])
    _install_fake_client(monkeypatch, fake_client)

    with caplog.at_level(logging.WARNING):
        result = tools.convert_currency(100, "EUR", "USD")

    assert "нечитаемый" in result.lower()
    assert "event=tool_call_failed" in caplog.text
    assert "tool=convert_currency" in caplog.text


def test_convert_currency_unknown_target_rate(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_client = FakeHttpxClient([FakeResponse(json_data={"result": "success", "rates": {}})])
    _install_fake_client(monkeypatch, fake_client)

    result = tools.convert_currency(100, "EUR", "XYZ")

    assert "нет курса" in result.lower()


def test_convert_currency_blank_currency_rejected_without_http_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _explode(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("must not perform HTTP calls for blank input")

    monkeypatch.setattr(tools.httpx, "Client", _explode)

    result = tools.convert_currency(100, "", "USD")

    assert "не может быть пустым" in result.lower()


def test_convert_currency_transport_failure_returns_safe_fallback(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    fake_client = FakeHttpxClient(exception=httpx.ConnectError("connection failed"))
    _install_fake_client(monkeypatch, fake_client)

    with caplog.at_level(logging.WARNING):
        result = tools.convert_currency(100, "EUR", "USD")

    assert "недоступен" in result.lower()
    assert "event=tool_call_failed" in caplog.text
    assert "tool=convert_currency" in caplog.text


# ---------------------------------------------------------------------------
# get_public_holidays
# ---------------------------------------------------------------------------


def test_get_public_holidays_success(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_client = FakeHttpxClient(
        [
            FakeResponse(
                json_data=[
                    {
                        "date": "2026-01-01",
                        "localName": "Uudenvuodenpäivä",
                        "name": "New Year's Day",
                    },
                    {
                        "date": "2026-12-06",
                        "localName": "Itsenäisyyspäivä",
                        "name": "Independence Day",
                    },
                ]
            )
        ]
    )
    _install_fake_client(monkeypatch, fake_client)

    result = tools.get_public_holidays("FI", year=2026)

    assert "2026-01-01" in result
    assert "Uudenvuodenpäivä" in result
    assert "New Year's Day" in result
    assert "2026-12-06" in result
    assert "Independence Day" in result
    assert fake_client.calls[0]["url"] == tools._NAGER_HOLIDAYS_URL.format(
        year=2026, country_code="FI"
    )


def test_get_public_holidays_uses_expected_client_options(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_client = FakeHttpxClient(
        [FakeResponse(json_data=[{"date": "2026-01-01", "localName": "X", "name": "X"}])]
    )
    _install_fake_client(monkeypatch, fake_client)

    tools.get_public_holidays("FI", year=2026)

    assert fake_client.client_kwargs["timeout"] == 10.0
    assert fake_client.client_kwargs["follow_redirects"] is True


def test_get_public_holidays_uses_v4_denmark_url(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_client = FakeHttpxClient(
        [
            FakeResponse(
                json_data=[
                    {
                        "date": "2026-01-01",
                        "localName": "Nytårsdag",
                        "name": "New Year's Day",
                    }
                ]
            )
        ]
    )
    _install_fake_client(monkeypatch, fake_client)

    tools.get_public_holidays("Дания", year=2026)

    assert (
        fake_client.calls[0]["url"] == "https://date.nager.at/api/v4/Holidays/DK/2026"
    )


def test_get_public_holidays_uses_v4_sweden_url(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_client = FakeHttpxClient(
        [
            FakeResponse(
                json_data=[
                    {
                        "date": "2026-01-01",
                        "localName": "Nyårsdagen",
                        "name": "New Year's Day",
                    }
                ]
            )
        ]
    )
    _install_fake_client(monkeypatch, fake_client)

    tools.get_public_holidays("Швеция", year=2026)

    assert (
        fake_client.calls[0]["url"] == "https://date.nager.at/api/v4/Holidays/SE/2026"
    )


def test_get_public_holidays_omits_duplicate_name_when_same_as_local_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_client = FakeHttpxClient(
        [FakeResponse(json_data=[{"date": "2026-01-01", "localName": "X", "name": "X"}])]
    )
    _install_fake_client(monkeypatch, fake_client)

    result = tools.get_public_holidays("FI", year=2026)

    assert result.count("X") == 1


def test_get_public_holidays_uses_current_year_when_year_omitted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from datetime import UTC, datetime

    fake_client = FakeHttpxClient(
        [FakeResponse(json_data=[{"date": "2026-01-01", "localName": "X", "name": "X"}])]
    )
    _install_fake_client(monkeypatch, fake_client)

    current_year = datetime.now(UTC).year
    result = tools.get_public_holidays("FI")

    assert str(current_year) in result
    assert fake_client.calls[0]["url"] == tools._NAGER_HOLIDAYS_URL.format(
        year=current_year, country_code="FI"
    )


@pytest.mark.parametrize(
    ("country_name", "expected_code"),
    [
        ("Швеция", "SE"),
        ("Sweden", "SE"),
        ("se", "SE"),
        ("Дания", "DK"),
        ("Denmark", "DK"),
        ("Финляндия", "FI"),
        ("Finland", "FI"),
        ("Германия", "DE"),
        ("Germany", "DE"),
        ("Франция", "FR"),
        ("France", "FR"),
        ("Япония", "JP"),
        ("Japan", "JP"),
        ("США", "US"),
        ("Америка", "US"),
        ("United States", "US"),
        ("Великобритания", "GB"),
        ("Англия", "GB"),
        ("United Kingdom", "GB"),
        ("Южная Корея", "KR"),
        ("Северная Корея", "KP"),
        ("Чехия", "CZ"),
    ],
)
def test_normalize_country_code_resolves_supported_names_and_codes(
    country_name: str, expected_code: str
) -> None:
    assert tools._normalize_country_code(country_name) == expected_code


def test_normalize_country_code_leaves_iso_code_unchanged() -> None:
    assert tools._normalize_country_code("FI") == "FI"


def test_get_public_holidays_normalizes_russian_country_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_client = FakeHttpxClient(
        [FakeResponse(json_data=[{"date": "2026-01-01", "localName": "X", "name": "X"}])]
    )
    _install_fake_client(monkeypatch, fake_client)

    result = tools.get_public_holidays("Финляндия", year=2026)

    assert "FI" in result
    assert fake_client.calls[0]["url"] == tools._NAGER_HOLIDAYS_URL.format(
        year=2026, country_code="FI"
    )


def test_get_public_holidays_normalizes_russian_sweden_country_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_client = FakeHttpxClient(
        [
            FakeResponse(
                json_data=[
                    {
                        "date": "2026-01-01",
                        "localName": "Nyårsdagen",
                        "name": "New Year's Day",
                    }
                ]
            )
        ]
    )
    _install_fake_client(monkeypatch, fake_client)

    result = tools.get_public_holidays("Швеция", year=2026)

    assert "SE" in result
    assert fake_client.calls[0]["url"] == tools._NAGER_HOLIDAYS_URL.format(
        year=2026, country_code="SE"
    )


def test_get_public_holidays_blank_country_rejected_without_http_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _explode(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("must not perform HTTP calls for blank input")

    monkeypatch.setattr(tools.httpx, "Client", _explode)

    result = tools.get_public_holidays("   ")

    assert "не может быть пустым" in result.lower()


@pytest.mark.parametrize("bad_year", ["2026", True, 2026.5])
def test_get_public_holidays_non_int_year_rejected_without_http_call(
    monkeypatch: pytest.MonkeyPatch, bad_year: Any
) -> None:
    def _explode(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("must not perform HTTP calls for non-integer year")

    monkeypatch.setattr(tools.httpx, "Client", _explode)

    result = tools.get_public_holidays("FI", year=bad_year)

    assert "год должен быть целым числом" in result.lower()


def test_get_public_holidays_unknown_country_returns_safe_error_without_http_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _explode(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("must not perform HTTP calls for unknown country name")

    monkeypatch.setattr(tools.httpx, "Client", _explode)

    result = tools.get_public_holidays("Nowhereland", year=2026)

    assert "не удалось распознать страну" in result.lower()
    assert "именительном падеже" in result.lower()


def test_get_public_holidays_invalid_iso_code_returns_not_found(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_client = FakeHttpxClient([FakeResponse(status_code=404)])
    _install_fake_client(monkeypatch, fake_client)

    result = tools.get_public_holidays("ZZ", year=2026)

    assert "не предоставляет данные" in result.lower()


def test_get_public_holidays_empty_list(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_client = FakeHttpxClient([FakeResponse(json_data=[])])
    _install_fake_client(monkeypatch, fake_client)

    result = tools.get_public_holidays("FI", year=2026)

    assert "не предоставляет данные" in result.lower()


@pytest.mark.parametrize("malformed_top_level", [{"not": "a list"}, "not-a-list", None, 42])
def test_get_public_holidays_non_list_top_level_returns_safe_fallback(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, malformed_top_level: Any
) -> None:
    fake_client = FakeHttpxClient([FakeResponse(json_data=malformed_top_level)])
    _install_fake_client(monkeypatch, fake_client)

    with caplog.at_level(logging.WARNING):
        result = tools.get_public_holidays("FI", year=2026)

    assert "нечитаемый" in result.lower()
    assert "event=tool_call_failed" in caplog.text
    assert "tool=get_public_holidays" in caplog.text


@pytest.mark.parametrize("malformed_entry", ["not-an-object", 42, ["nested-list"], None])
def test_get_public_holidays_non_dict_entry_returns_safe_fallback(
    monkeypatch: pytest.MonkeyPatch, malformed_entry: Any
) -> None:
    fake_client = FakeHttpxClient([FakeResponse(json_data=[malformed_entry])])
    _install_fake_client(monkeypatch, fake_client)

    result = tools.get_public_holidays("FI", year=2026)

    assert "нечитаемый" in result.lower()


@pytest.mark.parametrize("malformed_date", [None, 42, ""])
def test_get_public_holidays_missing_date_returns_safe_fallback(
    monkeypatch: pytest.MonkeyPatch, malformed_date: Any
) -> None:
    fake_client = FakeHttpxClient(
        [FakeResponse(json_data=[{"date": malformed_date, "localName": "X", "name": "X"}])]
    )
    _install_fake_client(monkeypatch, fake_client)

    result = tools.get_public_holidays("FI", year=2026)

    assert "нечитаемый" in result.lower()


def test_get_public_holidays_missing_local_name_and_name_uses_generic_label(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_client = FakeHttpxClient([FakeResponse(json_data=[{"date": "2026-01-01"}])])
    _install_fake_client(monkeypatch, fake_client)

    result = tools.get_public_holidays("FI", year=2026)

    assert "праздник" in result.lower()


def test_get_public_holidays_missing_local_name_falls_back_to_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_client = FakeHttpxClient(
        [FakeResponse(json_data=[{"date": "2026-01-01", "name": "New Year's Day"}])]
    )
    _install_fake_client(monkeypatch, fake_client)

    result = tools.get_public_holidays("FI", year=2026)

    assert "New Year's Day" in result


def test_get_public_holidays_malformed_json_returns_safe_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class BrokenResponse(FakeResponse):
        def json(self) -> Any:
            raise ValueError("not json")

    fake_client = FakeHttpxClient([BrokenResponse()])
    _install_fake_client(monkeypatch, fake_client)

    result = tools.get_public_holidays("FI", year=2026)

    assert "нечитаемый" in result.lower()


def test_get_public_holidays_transport_failure_returns_safe_fallback(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    fake_client = FakeHttpxClient(exception=httpx.ConnectTimeout("timed out"))
    _install_fake_client(monkeypatch, fake_client)

    with caplog.at_level(logging.WARNING):
        result = tools.get_public_holidays("FI", year=2026)

    assert "недоступен" in result.lower()
    assert "event=tool_call_failed" in caplog.text
    assert "tool=get_public_holidays" in caplog.text

# ---------------------------------------------------------------------------
# get_pypi_package_info
# ---------------------------------------------------------------------------


def test_get_pypi_package_info_haystack_ai_success(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_client = FakeHttpxClient(
        [
            FakeResponse(
                json_data={
                    "info": {
                        "name": "haystack-ai",
                        "version": "2.4.0",
                        "summary": "LLM orchestration framework",
                        "requires_python": ">=3.9",
                        "license_expression": "Apache-2.0",
                        "project_urls": {"Homepage": "https://haystack.deepset.ai/"},
                    }
                }
            )
        ]
    )
    _install_fake_client(monkeypatch, fake_client)

    result = tools.get_pypi_package_info("haystack-ai")

    assert "Пакет haystack-ai" in result
    assert "2.4.0" in result
    assert "LLM orchestration framework" in result
    assert ">=3.9" in result
    assert "Apache-2.0" in result
    assert "https://haystack.deepset.ai/" in result
    assert fake_client.calls[0]["url"] == tools._PYPI_PACKAGE_INFO_URL.format(
        package_name="haystack-ai"
    )


def test_get_pypi_package_info_aiogram_success(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_client = FakeHttpxClient(
        [
            FakeResponse(
                json_data={
                    "info": {
                        "name": "aiogram",
                        "version": "3.22.0",
                        "summary": "Modern and fully asynchronous framework for Telegram Bot API",
                        "requires_python": ">=3.9",
                        "license": "MIT",
                        "project_urls": {"Documentation": "https://docs.aiogram.dev/"},
                    }
                }
            )
        ]
    )
    _install_fake_client(monkeypatch, fake_client)

    result = tools.get_pypi_package_info("aiogram")

    assert "Пакет aiogram" in result
    assert "3.22.0" in result
    assert "Telegram Bot API" in result
    assert "MIT" in result
    assert "https://docs.aiogram.dev/" in result


def test_get_pypi_package_info_normalizes_russian_alias(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_client = FakeHttpxClient(
        [
            FakeResponse(
                json_data={
                    "info": {
                        "name": "aiogram",
                        "version": "3.22.0",
                        "summary": "Async Telegram framework",
                        "requires_python": ">=3.9",
                        "license": "MIT",
                        "package_url": "https://pypi.org/project/aiogram/",
                    }
                }
            )
        ]
    )
    _install_fake_client(monkeypatch, fake_client)

    result = tools.get_pypi_package_info("айограм")

    assert "Пакет aiogram" in result
    assert fake_client.calls[0]["url"] == tools._PYPI_PACKAGE_INFO_URL.format(
        package_name="aiogram"
    )


@pytest.mark.parametrize(
    ("alias", "expected_package_name"),
    [
        ("Haystack", "haystack-ai"),
        ("Haystack AI", "haystack-ai"),
        ("Pinecone Haystack", "pinecone-haystack"),
        ("пакет Pinecone для Haystack", "pinecone-haystack"),
        ("айограм", "aiogram"),
    ],
)
def test_normalize_pypi_package_name_maps_documented_aliases(
    alias: str, expected_package_name: str
) -> None:
    assert tools._normalize_pypi_package_name(alias) == expected_package_name


def test_get_pypi_package_info_404_package_not_found(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    fake_client = FakeHttpxClient([FakeResponse(status_code=404)])
    _install_fake_client(monkeypatch, fake_client)

    with caplog.at_level(logging.WARNING):
        result = tools.get_pypi_package_info("does-not-exist")

    assert "не найден" in result.lower()
    assert "event=tool_call_failed" in caplog.text
    assert "tool=get_pypi_package_info" in caplog.text


@pytest.mark.parametrize("malformed_payload", [[], {"info": []}])
def test_get_pypi_package_info_malformed_payload_returns_safe_fallback(
    monkeypatch: pytest.MonkeyPatch, malformed_payload: Any
) -> None:
    fake_client = FakeHttpxClient([FakeResponse(json_data=malformed_payload)])
    _install_fake_client(monkeypatch, fake_client)

    result = tools.get_pypi_package_info("haystack-ai")

    assert "нечитаемый" in result.lower()


def test_get_pypi_package_info_network_failure_returns_safe_fallback(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    fake_client = FakeHttpxClient(exception=httpx.ConnectTimeout("timed out"))
    _install_fake_client(monkeypatch, fake_client)

    with caplog.at_level(logging.WARNING):
        result = tools.get_pypi_package_info("haystack-ai")

    assert "сервис сейчас недоступен" in result.lower()
    assert "event=tool_call_failed" in caplog.text
    assert "tool=get_pypi_package_info" in caplog.text


def test_get_pypi_package_info_uses_expected_client_options_and_headers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_client = FakeHttpxClient(
        [
            FakeResponse(
                json_data={
                    "info": {
                        "name": "haystack-ai",
                        "version": "2.4.0",
                        "summary": "LLM orchestration framework",
                        "requires_python": ">=3.9",
                        "license": "Apache-2.0",
                        "package_url": "https://pypi.org/project/haystack-ai/",
                    }
                }
            )
        ]
    )
    _install_fake_client(monkeypatch, fake_client)

    tools.get_pypi_package_info("Haystack AI")

    assert fake_client.client_kwargs["timeout"] == 10.0
    assert fake_client.client_kwargs["follow_redirects"] is True
    assert fake_client.calls[0]["url"] == tools._PYPI_PACKAGE_INFO_URL.format(
        package_name="haystack-ai"
    )
    assert fake_client.calls[0]["kwargs"]["headers"] == {
        "Accept": "application/json",
        "User-Agent": "telegram-vector-memory-bot/0.1 educational project",
    }


def test_get_pypi_package_info_blank_package_name_rejected_without_http_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _explode(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("must not perform HTTP calls for blank package name")

    monkeypatch.setattr(tools.httpx, "Client", _explode)

    result = tools.get_pypi_package_info("   ")

    assert "не может быть пустым" in result.lower()


# ---------------------------------------------------------------------------
# get_current_time
# ---------------------------------------------------------------------------


def test_get_current_time_success_by_iana_timezone() -> None:
    result = tools.get_current_time("UTC")

    assert "UTC" in result
    assert "Текущее время для 'UTC'" in result


def test_get_current_time_success_by_city_alias() -> None:
    result = tools.get_current_time("Helsinki")

    assert "Europe/Helsinki" in result
    assert "Текущее время для 'Helsinki'" in result


@pytest.mark.parametrize(
    ("alias", "expected_zone"),
    [
        ("Helsinki", "Europe/Helsinki"),
        ("Moscow", "Europe/Moscow"),
        ("London", "Europe/London"),
        ("New York", "America/New_York"),
        ("Tokyo", "Asia/Tokyo"),
        ("Berlin", "Europe/Berlin"),
        ("Paris", "Europe/Paris"),
    ],
)
def test_get_current_time_all_documented_city_aliases_resolve(
    alias: str, expected_zone: str
) -> None:
    result = tools.get_current_time(alias)

    assert expected_zone in result


def test_get_current_time_city_alias_is_case_insensitive() -> None:
    result = tools.get_current_time("hELSINKI")

    assert "Europe/Helsinki" in result


def test_get_current_time_unknown_location_returns_fallback_with_examples(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING):
        result = tools.get_current_time("Nowhereland")

    assert "неизвестный город или часовой пояс" in result.lower()
    assert "helsinki" in result.lower()
    assert "moscow" in result.lower()
    assert "event=tool_call_failed" in caplog.text
    assert "tool=get_current_time" in caplog.text


def test_get_current_time_blank_location_rejected() -> None:
    result = tools.get_current_time("   ")

    assert "не может быть пустым" in result.lower()


def test_get_current_time_includes_weekday() -> None:
    import datetime as datetime_module

    result = tools.get_current_time("UTC")

    weekday_names_ru = {
        "понедельник",
        "вторник",
        "среда",
        "четверг",
        "пятница",
        "суббота",
        "воскресенье",
    }
    assert any(day in result for day in weekday_names_ru)
    # sanity: сегодняшний реальный день недели должен быть среди присутствующих названий
    today_english = datetime_module.datetime.now(datetime_module.UTC).strftime("%A")
    assert tools._WEEKDAY_NAMES_RU[today_english] in result


# ---------------------------------------------------------------------------
# Регистрация инструментов
# ---------------------------------------------------------------------------


def test_build_default_tools_returns_all_five_tools_with_expected_names() -> None:
    tool_list = tools.build_default_tools()

    assert [tool.name for tool in tool_list] == [
        "get_current_weather",
        "convert_currency",
        "get_current_time",
        "get_public_holidays",
        "get_pypi_package_info",
    ]


def test_weather_tool_function_and_required_parameters() -> None:
    assert tools.weather_tool.function is tools.get_current_weather
    assert tools.weather_tool.parameters["required"] == ["city"]


def test_currency_tool_function_and_required_parameters() -> None:
    assert tools.currency_tool.function is tools.convert_currency
    assert tools.currency_tool.parameters["required"] == ["amount", "from_currency", "to_currency"]


def test_public_holidays_tool_function_and_required_parameters() -> None:
    assert tools.public_holidays_tool.function is tools.get_public_holidays
    assert tools.public_holidays_tool.parameters["required"] == ["country"]


def test_pypi_package_info_tool_function_and_required_parameters() -> None:
    assert tools.pypi_package_info_tool.function is tools.get_pypi_package_info
    assert tools.pypi_package_info_tool.parameters["required"] == ["package_name"]


def test_time_tool_function_and_required_parameters() -> None:
    assert tools.time_tool.function is tools.get_current_time
    assert tools.time_tool.parameters["required"] == ["location"]


def test_build_default_tools_returns_a_fresh_list_each_call() -> None:
    first = tools.build_default_tools()
    second = tools.build_default_tools()

    assert first == second
    assert first is not second


_CYRILLIC_PATTERN = re.compile(r"[Ѐ-ӿ]")


@pytest.mark.parametrize(
    "tool",
    [
        tools.weather_tool,
        tools.currency_tool,
        tools.time_tool,
        tools.public_holidays_tool,
        tools.pypi_package_info_tool,
    ],
    ids=lambda tool: tool.name,
)
def test_tool_description_is_in_russian(tool: Tool) -> None:
    assert _CYRILLIC_PATTERN.search(tool.description), (
        f"{tool.name} description must be in Russian: {tool.description!r}"
    )


@pytest.mark.parametrize(
    "tool",
    [
        tools.weather_tool,
        tools.currency_tool,
        tools.time_tool,
        tools.public_holidays_tool,
        tools.pypi_package_info_tool,
    ],
    ids=lambda tool: tool.name,
)
def test_tool_parameter_descriptions_are_in_russian(tool: Tool) -> None:
    for param_name, schema in tool.parameters["properties"].items():
        assert _CYRILLIC_PATTERN.search(schema["description"]), (
            f"{tool.name}.{param_name} description must be in Russian: {schema['description']!r}"
        )
