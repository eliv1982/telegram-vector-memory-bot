"""Direct live smoke test for the five active Haystack tools.

This script intentionally bypasses Telegram and Haystack model routing: it
calls the tool functions from ``telegram_vector_memory_bot.tools`` directly and
prints PASS/FAIL for each one. Any failing check makes the process exit
non-zero.
"""

from __future__ import annotations

from collections.abc import Callable

from telegram_vector_memory_bot import tools


def _looks_like_failure(result: str) -> bool:
    return result.startswith("Не удалось")


def _run_case(
    name: str,
    call: Callable[[], str],
    validator: Callable[[str], bool],
) -> bool:
    try:
        result = call()
    except Exception as exc:  # pragma: no cover - smoke script guardrail
        print(f"FAIL {name}: unexpected exception {type(exc).__name__}")
        return False

    if _looks_like_failure(result) or not validator(result):
        print(f"FAIL {name}: {result}")
        return False

    print(f"PASS {name}: {result}")
    return True


def main() -> int:
    checks = [
        (
            "weather",
            lambda: tools.get_current_weather("Хельсинки"),
            lambda result: "Текущая погода" in result and "км/ч" in result,
        ),
        (
            "currency",
            lambda: tools.convert_currency(100, "USD", "EUR"),
            lambda result: "100 USD =" in result and "EUR" in result,
        ),
        (
            "time",
            lambda: tools.get_current_time("Tokyo"),
            lambda result: "Asia/Tokyo" in result and "Текущее время" in result,
        ),
        (
            "holidays_sweden",
            lambda: tools.get_public_holidays("Sweden", year=2026),
            lambda result: "Праздники в SE (2026):" in result,
        ),
        (
            "holidays_denmark",
            lambda: tools.get_public_holidays("Дания", year=2026),
            lambda result: "Праздники в DK (2026):" in result,
        ),
        (
            "pypi",
            lambda: tools.get_pypi_package_info("haystack-ai"),
            lambda result: "Пакет haystack-ai:" in result and "Ссылка:" in result,
        ),
    ]

    failures = 0
    for name, call, validator in checks:
        if not _run_case(name, call, validator):
            failures += 1

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
