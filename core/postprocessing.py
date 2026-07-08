"""Post-processing and normalization for extracted fields."""

from __future__ import annotations

import re
from datetime import datetime
from typing import Dict, Tuple


MONTHS = {
    "января": 1,
    "февраля": 2,
    "марта": 3,
    "апреля": 4,
    "мая": 5,
    "июня": 6,
    "июля": 7,
    "августа": 8,
    "сентября": 9,
    "октября": 10,
    "ноября": 11,
    "декабря": 12,
}

def _format_numeric_date(value: datetime) -> str:
    """Format a date as DD.MM.YYYY."""

    return value.strftime("%d.%m.%Y")


def _normalize_date(value: str) -> str | None:
    cleaned_value = re.sub(r"\s*([./-])\s*", r"\1", value.strip())
    long_date = re.search(
        r"(?:«|\"|')?\s*(\d{1,2})\s*(?:»|\"|')?\s+"
        r"(января|февраля|марта|апреля|мая|июня|июля|августа|сентября|октября|ноября|декабря)\s+"
        r"(\d{4})",
        cleaned_value.lower(),
    )
    if long_date:
        day = int(long_date.group(1))
        month = MONTHS[long_date.group(2)]
        year = int(long_date.group(3))
        try:
            return _format_numeric_date(datetime(year, month, day))
        except ValueError:
            return None

    for fmt in ("%d.%m.%Y", "%d/%m/%Y", "%d-%m-%Y", "%Y-%m-%d", "%d.%m.%y", "%d/%m/%y", "%d-%m-%y"):
        try:
            return _format_numeric_date(datetime.strptime(cleaned_value, fmt))
        except ValueError:
            continue
    return None


def normalize_fields(fields: Dict[str, dict]) -> Tuple[Dict[str, dict], list[str]]:
    """Validate and normalize known field values."""

    warnings: list[str] = []
    normalized = {key: dict(value) for key, value in fields.items()}

    inn = normalized.get("inn", {}).get("value")
    if inn and not re.fullmatch(r"\d{10}|\d{12}", inn):
        warnings.append("Extracted INN has an unexpected format")

    kpp = normalized.get("kpp", {}).get("value")
    if kpp and not re.fullmatch(r"\d{9}", kpp):
        warnings.append("Extracted KPP has an unexpected format")

    date = normalized.get("document_date", {}).get("value")
    if date:
        parsed = _normalize_date(date)
        if parsed:
            normalized["document_date"]["value"] = parsed
        else:
            warnings.append("Document date could not be normalized")

    for field in normalized.values():
        if isinstance(field.get("value"), str):
            field["value"] = " ".join(field["value"].split())
    return normalized, warnings
