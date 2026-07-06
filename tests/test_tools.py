"""Unit tests for telegram_vector_memory_bot.tools.

All tests run against a fake httpx.Client -- no real network call is ever
made. Each tool function is exercised directly (success, not-found, and
transport-failure paths), and each Tool wrapper is checked for having the
expected name/function/required-parameters wiring the agent depends on.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx
import pytest

from telegram_vector_memory_bot import tools

_UNSET: Any = object()


class FakeResponse:
    def __init__(self, *, status_code: int = 200, json_data: Any = _UNSET) -> None:
        self.status_code = status_code
        # A real sentinel (not just `None`) distinguishes "no json_data passed
        # -> default to {}" from "the test wants .json() to literally return
        # None/a list/a string", which malformed-payload tests need below.
        self._json_data = {} if json_data is _UNSET else json_data

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            request = httpx.Request("GET", "https://example.invalid")
            response = httpx.Response(self.status_code, request=request)
            raise httpx.HTTPStatusError("error", request=request, response=response)

    def json(self) -> Any:
        return self._json_data


class FakeHttpxClient:
    """Fake stand-in for httpx.Client -- returns scripted responses, no network I/O."""

    def __init__(
        self,
        responses: list[FakeResponse] | None = None,
        *,
        exception: Exception | None = None,
    ):
        self._responses = iter(responses or [])
        self._exception = exception
        self.calls: list[dict[str, Any]] = []
        # Populated by _install_fake_client with whatever kwargs the tool
        # under test passed to httpx.Client(...), so tests can assert on
        # e.g. timeout/follow_redirects without touching real network code.
        self.client_kwargs: dict[str, Any] = {}

    def get(self, url: str, params: dict[str, Any] | None = None, **kwargs: Any) -> FakeResponse:
        self.calls.append({"url": url, "params": params})
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


def test_get_current_weather_city_not_found(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_client = FakeHttpxClient([FakeResponse(json_data={"results": []})])
    _install_fake_client(monkeypatch, fake_client)

    result = tools.get_current_weather("Nonexistentville")

    assert "not found" in result.lower()


def test_get_current_weather_blank_city_rejected_without_http_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _explode(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("must not perform HTTP calls for blank input")

    monkeypatch.setattr(tools.httpx, "Client", _explode)

    result = tools.get_current_weather("   ")

    assert "must not be empty" in result.lower()


def test_get_current_weather_transport_failure_returns_safe_fallback(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    fake_client = FakeHttpxClient(exception=httpx.ConnectTimeout("timed out"))
    _install_fake_client(monkeypatch, fake_client)

    with caplog.at_level(logging.WARNING):
        result = tools.get_current_weather("Helsinki")

    assert "unavailable" in result.lower()
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

    assert "no coordinates" in result.lower()


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

    assert "no current weather data" in result.lower()


def test_get_current_weather_malformed_json_returns_safe_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class BrokenResponse(FakeResponse):
        def json(self) -> Any:
            raise ValueError("not json")

    fake_client = FakeHttpxClient([BrokenResponse()])
    _install_fake_client(monkeypatch, fake_client)

    result = tools.get_current_weather("Helsinki")

    assert "unreadable" in result.lower()


@pytest.mark.parametrize("malformed_geo_data", [[], "not-an-object", None, 42])
def test_get_current_weather_non_dict_geocoding_response_returns_safe_fallback(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, malformed_geo_data: Any
) -> None:
    fake_client = FakeHttpxClient([FakeResponse(json_data=malformed_geo_data)])
    _install_fake_client(monkeypatch, fake_client)

    with caplog.at_level(logging.WARNING):
        result = tools.get_current_weather("Helsinki")

    assert "unreadable" in result.lower()
    assert "event=tool_call_failed" in caplog.text
    assert "tool=get_current_weather" in caplog.text


@pytest.mark.parametrize("malformed_entry", ["not-an-object", 42, ["nested-list"], None])
def test_get_current_weather_non_dict_result_entry_returns_safe_fallback(
    monkeypatch: pytest.MonkeyPatch, malformed_entry: Any
) -> None:
    fake_client = FakeHttpxClient([FakeResponse(json_data={"results": [malformed_entry]})])
    _install_fake_client(monkeypatch, fake_client)

    result = tools.get_current_weather("Helsinki")

    assert "unreadable" in result.lower()


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

    assert "unreadable" in result.lower()


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


def test_convert_currency_blank_target_currency_rejected_without_http_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _explode(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("must not perform HTTP calls for blank input")

    monkeypatch.setattr(tools.httpx, "Client", _explode)

    result = tools.convert_currency(100, "EUR", "")

    assert "must not be empty" in result.lower()


def test_convert_currency_non_numeric_amount_rejected_without_http_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _explode(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("must not perform HTTP calls for non-numeric amount")

    monkeypatch.setattr(tools.httpx, "Client", _explode)

    result = tools.convert_currency(True, "EUR", "USD")  # bool must be rejected, not treated as 1

    assert "amount must be numeric" in result.lower()


def test_convert_currency_result_not_success(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_client = FakeHttpxClient([FakeResponse(json_data={"result": "error"})])
    _install_fake_client(monkeypatch, fake_client)

    result = tools.convert_currency(100, "EUR", "USD")

    assert "no rates available" in result.lower()


def test_convert_currency_malformed_rate_type(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_client = FakeHttpxClient(
        [FakeResponse(json_data={"result": "success", "rates": {"USD": "not-a-number"}})]
    )
    _install_fake_client(monkeypatch, fake_client)

    result = tools.convert_currency(100, "EUR", "USD")

    assert "malformed rate" in result.lower()


def test_convert_currency_malformed_json_returns_safe_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class BrokenResponse(FakeResponse):
        def json(self) -> Any:
            raise ValueError("not json")

    fake_client = FakeHttpxClient([BrokenResponse()])
    _install_fake_client(monkeypatch, fake_client)

    result = tools.convert_currency(100, "EUR", "USD")

    assert "unreadable" in result.lower()


@pytest.mark.parametrize("malformed_data", [[], "not-an-object", None, 42])
def test_convert_currency_non_dict_response_returns_safe_fallback(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, malformed_data: Any
) -> None:
    fake_client = FakeHttpxClient([FakeResponse(json_data=malformed_data)])
    _install_fake_client(monkeypatch, fake_client)

    with caplog.at_level(logging.WARNING):
        result = tools.convert_currency(100, "EUR", "USD")

    assert "unreadable" in result.lower()
    assert "event=tool_call_failed" in caplog.text
    assert "tool=convert_currency" in caplog.text


def test_convert_currency_unknown_target_rate(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_client = FakeHttpxClient([FakeResponse(json_data={"result": "success", "rates": {}})])
    _install_fake_client(monkeypatch, fake_client)

    result = tools.convert_currency(100, "EUR", "XYZ")

    assert "no rate available" in result.lower()


def test_convert_currency_blank_currency_rejected_without_http_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _explode(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("must not perform HTTP calls for blank input")

    monkeypatch.setattr(tools.httpx, "Client", _explode)

    result = tools.convert_currency(100, "", "USD")

    assert "must not be empty" in result.lower()


def test_convert_currency_transport_failure_returns_safe_fallback(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    fake_client = FakeHttpxClient(exception=httpx.ConnectError("connection failed"))
    _install_fake_client(monkeypatch, fake_client)

    with caplog.at_level(logging.WARNING):
        result = tools.convert_currency(100, "EUR", "USD")

    assert "unavailable" in result.lower()
    assert "event=tool_call_failed" in caplog.text
    assert "tool=convert_currency" in caplog.text


# ---------------------------------------------------------------------------
# get_country_info
# ---------------------------------------------------------------------------


def test_get_country_info_success(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_client = FakeHttpxClient(
        [
            FakeResponse(
                json_data=[
                    {
                        "name": {"common": "Finland"},
                        "capital": ["Helsinki"],
                        "currencies": {"EUR": {"name": "Euro"}},
                        "languages": {"fin": "Finnish", "swe": "Swedish"},
                        "population": 5540720,
                        "region": "Europe",
                    }
                ]
            )
        ]
    )
    _install_fake_client(monkeypatch, fake_client)

    result = tools.get_country_info("Finland")

    assert "Finland" in result
    assert "Helsinki" in result
    assert "Euro" in result
    assert "Finnish" in result
    assert "5540720" in result
    assert "Europe" in result


def test_get_country_info_uses_expected_client_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_client = FakeHttpxClient([FakeResponse(json_data=[{"name": {"common": "Finland"}}])])
    _install_fake_client(monkeypatch, fake_client)

    tools.get_country_info("Finland")

    assert fake_client.client_kwargs["timeout"] == 10.0


def test_get_country_info_not_found(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_client = FakeHttpxClient([FakeResponse(status_code=404)])
    _install_fake_client(monkeypatch, fake_client)

    result = tools.get_country_info("Nowhereland")

    assert "no country found" in result.lower()


def test_get_country_info_empty_result_list(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_client = FakeHttpxClient([FakeResponse(json_data=[])])
    _install_fake_client(monkeypatch, fake_client)

    result = tools.get_country_info("Nowhereland")

    assert "no country found" in result.lower()


@pytest.mark.parametrize("malformed_top_level", [{"not": "a list"}, "not-a-list", None, 42])
def test_get_country_info_non_list_top_level_returns_safe_fallback(
    monkeypatch: pytest.MonkeyPatch, malformed_top_level: Any
) -> None:
    fake_client = FakeHttpxClient([FakeResponse(json_data=malformed_top_level)])
    _install_fake_client(monkeypatch, fake_client)

    result = tools.get_country_info("Nowhereland")

    assert "no country found" in result.lower()


@pytest.mark.parametrize("malformed_entry", ["not-an-object", 42, ["nested-list"], None])
def test_get_country_info_non_dict_entry_returns_safe_fallback(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, malformed_entry: Any
) -> None:
    fake_client = FakeHttpxClient([FakeResponse(json_data=[malformed_entry])])
    _install_fake_client(monkeypatch, fake_client)

    with caplog.at_level(logging.WARNING):
        result = tools.get_country_info("Nowhereland")

    assert "unreadable" in result.lower()
    assert "event=tool_call_failed" in caplog.text
    assert "tool=get_country_info" in caplog.text


def test_get_country_info_currency_entry_with_wrong_shape_does_not_raise(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_client = FakeHttpxClient(
        [
            FakeResponse(
                json_data=[
                    {
                        "name": {"common": "Finland"},
                        "capital": ["Helsinki"],
                        # "currencies" values are expected to be objects like
                        # {"name": "Euro"} -- a bare string is a malformed shape.
                        "currencies": {"EUR": "not-an-object"},
                        "languages": {"fin": "Finnish"},
                        "population": 5540720,
                        "region": "Europe",
                    }
                ]
            )
        ]
    )
    _install_fake_client(monkeypatch, fake_client)

    result = tools.get_country_info("Finland")

    assert "Finland" in result
    assert "EUR" in result


def test_get_country_info_language_entry_with_wrong_shape_does_not_raise(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_client = FakeHttpxClient(
        [
            FakeResponse(
                json_data=[
                    {
                        "name": {"common": "Finland"},
                        "capital": ["Helsinki"],
                        "currencies": {"EUR": {"name": "Euro"}},
                        # language values are expected to be strings -- a
                        # nested object/number here is a malformed shape.
                        "languages": {"fin": 12345},
                        "population": 5540720,
                        "region": "Europe",
                    }
                ]
            )
        ]
    )
    _install_fake_client(monkeypatch, fake_client)

    result = tools.get_country_info("Finland")

    assert "Finland" in result
    assert "12345" in result


def test_get_country_info_malformed_json_returns_safe_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class BrokenResponse(FakeResponse):
        def json(self) -> Any:
            raise ValueError("not json")

    fake_client = FakeHttpxClient([BrokenResponse()])
    _install_fake_client(monkeypatch, fake_client)

    result = tools.get_country_info("Finland")

    assert "unreadable" in result.lower()


def test_get_country_info_blank_rejected_without_http_call(monkeypatch: pytest.MonkeyPatch) -> None:
    def _explode(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("must not perform HTTP calls for blank input")

    monkeypatch.setattr(tools.httpx, "Client", _explode)

    result = tools.get_country_info("   ")

    assert "must not be empty" in result.lower()


def test_get_country_info_transport_failure_returns_safe_fallback(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    fake_client = FakeHttpxClient(exception=httpx.ReadTimeout("timed out"))
    _install_fake_client(monkeypatch, fake_client)

    with caplog.at_level(logging.WARNING):
        result = tools.get_country_info("Finland")

    assert "unavailable" in result.lower()
    assert "event=tool_call_failed" in caplog.text
    assert "tool=get_country_info" in caplog.text


# ---------------------------------------------------------------------------
# get_wikipedia_summary
# ---------------------------------------------------------------------------


def test_get_wikipedia_summary_success(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_client = FakeHttpxClient(
        [
            FakeResponse(
                json_data={
                    "title": "Alan Turing",
                    "extract": "English mathematician and computer scientist.",
                    "content_urls": {"desktop": {"page": "https://en.wikipedia.org/wiki/Alan_Turing"}},
                }
            )
        ]
    )
    _install_fake_client(monkeypatch, fake_client)

    result = tools.get_wikipedia_summary("Alan Turing")

    assert "Alan Turing" in result
    assert "mathematician" in result
    assert "https://en.wikipedia.org/wiki/Alan_Turing" in result


def test_get_wikipedia_summary_uses_expected_client_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_client = FakeHttpxClient(
        [FakeResponse(json_data={"title": "Alan Turing", "extract": "A summary."})]
    )
    _install_fake_client(monkeypatch, fake_client)

    tools.get_wikipedia_summary("Alan Turing")

    assert fake_client.client_kwargs["timeout"] == 10.0
    assert fake_client.client_kwargs["follow_redirects"] is True


def test_get_wikipedia_summary_not_found(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_client = FakeHttpxClient([FakeResponse(status_code=404)])
    _install_fake_client(monkeypatch, fake_client)

    result = tools.get_wikipedia_summary("Definitely Not A Real Topic Xyz")

    assert "no article found" in result.lower()


def test_get_wikipedia_summary_malformed_json_returns_safe_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class BrokenResponse(FakeResponse):
        def json(self) -> Any:
            raise ValueError("not json")

    fake_client = FakeHttpxClient([BrokenResponse()])
    _install_fake_client(monkeypatch, fake_client)

    result = tools.get_wikipedia_summary("Alan Turing")

    assert "unreadable" in result.lower()


def test_get_wikipedia_summary_no_extract_returns_safe_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_client = FakeHttpxClient([FakeResponse(json_data={"title": "Alan Turing"})])
    _install_fake_client(monkeypatch, fake_client)

    result = tools.get_wikipedia_summary("Alan Turing")

    assert "no summary available" in result.lower()


@pytest.mark.parametrize("malformed_data", [[], "not-an-object", None, 42])
def test_get_wikipedia_summary_non_dict_response_returns_safe_fallback(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, malformed_data: Any
) -> None:
    fake_client = FakeHttpxClient([FakeResponse(json_data=malformed_data)])
    _install_fake_client(monkeypatch, fake_client)

    with caplog.at_level(logging.WARNING):
        result = tools.get_wikipedia_summary("Alan Turing")

    assert "unreadable" in result.lower()
    assert "event=tool_call_failed" in caplog.text
    assert "tool=get_wikipedia_summary" in caplog.text


def test_get_wikipedia_summary_non_dict_desktop_url_does_not_raise(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_client = FakeHttpxClient(
        [
            FakeResponse(
                json_data={
                    "title": "Alan Turing",
                    "extract": "A summary.",
                    # "desktop" is expected to be an object with a "page" key --
                    # a bare string here is a malformed shape.
                    "content_urls": {"desktop": "not-an-object"},
                }
            )
        ]
    )
    _install_fake_client(monkeypatch, fake_client)

    result = tools.get_wikipedia_summary("Alan Turing")

    assert "Alan Turing" in result
    assert "A summary." in result
    assert "source:" not in result


def test_get_wikipedia_summary_blank_topic_rejected_without_http_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _explode(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("must not perform HTTP calls for blank input")

    monkeypatch.setattr(tools.httpx, "Client", _explode)

    result = tools.get_wikipedia_summary("   ")

    assert "must not be empty" in result.lower()


def test_get_wikipedia_summary_transport_failure_returns_safe_fallback(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    fake_client = FakeHttpxClient(exception=httpx.ConnectTimeout("timed out"))
    _install_fake_client(monkeypatch, fake_client)

    with caplog.at_level(logging.WARNING):
        result = tools.get_wikipedia_summary("Alan Turing")

    assert "unavailable" in result.lower()
    assert "event=tool_call_failed" in caplog.text
    assert "tool=get_wikipedia_summary" in caplog.text


# ---------------------------------------------------------------------------
# get_current_time
# ---------------------------------------------------------------------------


def test_get_current_time_success_by_iana_timezone() -> None:
    result = tools.get_current_time("UTC")

    assert "UTC" in result
    assert "Current time for 'UTC'" in result


def test_get_current_time_success_by_city_alias() -> None:
    result = tools.get_current_time("Helsinki")

    assert "Europe/Helsinki" in result
    assert "Current time for 'Helsinki'" in result


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

    assert "unknown city or timezone" in result.lower()
    assert "helsinki" in result.lower()
    assert "moscow" in result.lower()
    assert "event=tool_call_failed" in caplog.text
    assert "tool=get_current_time" in caplog.text


def test_get_current_time_blank_location_rejected() -> None:
    result = tools.get_current_time("   ")

    assert "must not be empty" in result.lower()


def test_get_current_time_includes_weekday() -> None:
    import datetime as datetime_module

    result = tools.get_current_time("UTC")

    weekday_names = {
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    }
    assert any(day in result for day in weekday_names)
    # sanity: today's real weekday must be one of the names present
    assert datetime_module.datetime.now(datetime_module.UTC).strftime("%A") in result


# ---------------------------------------------------------------------------
# Tool wiring
# ---------------------------------------------------------------------------


def test_build_default_tools_returns_all_five_tools_with_expected_names() -> None:
    tool_list = tools.build_default_tools()

    names = {tool.name for tool in tool_list}
    assert names == {
        "get_current_weather",
        "convert_currency",
        "get_country_info",
        "get_wikipedia_summary",
        "get_current_time",
    }


def test_weather_tool_function_and_required_parameters() -> None:
    assert tools.weather_tool.function is tools.get_current_weather
    assert tools.weather_tool.parameters["required"] == ["city"]


def test_currency_tool_function_and_required_parameters() -> None:
    assert tools.currency_tool.function is tools.convert_currency
    assert tools.currency_tool.parameters["required"] == ["amount", "from_currency", "to_currency"]


def test_country_info_tool_function_and_required_parameters() -> None:
    assert tools.country_info_tool.function is tools.get_country_info
    assert tools.country_info_tool.parameters["required"] == ["country"]


def test_wikipedia_summary_tool_function_and_required_parameters() -> None:
    assert tools.wikipedia_summary_tool.function is tools.get_wikipedia_summary
    assert tools.wikipedia_summary_tool.parameters["required"] == ["topic"]


def test_time_tool_function_and_required_parameters() -> None:
    assert tools.time_tool.function is tools.get_current_time
    assert tools.time_tool.parameters["required"] == ["location"]


def test_build_default_tools_returns_a_fresh_list_each_call() -> None:
    first = tools.build_default_tools()
    second = tools.build_default_tools()

    assert first == second
    assert first is not second
