"""Shared helpers for API endpoint modules."""

import json


def parse_json_list(raw: str | None, field_name: str = "field") -> list[str]:
    """Parse a JSON-encoded list from a form field.

    Accepts either a JSON array (``["a","b"]``) or a comma-separated
    string (``"a,b,c"``) as a fallback.

    Args:
        raw: Raw form value (JSON array or comma-separated string).
        field_name: Name of the field (used in error messages).

    Returns:
        List of non-empty strings.
    """
    if raw is None or not str(raw).strip():
        return []
    text = str(raw).strip()

    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return [str(x).strip() for x in parsed if str(x).strip()]
    except json.JSONDecodeError:
        pass

    return [item.strip() for item in text.split(",") if item.strip()]


def decode_suggestions(raw: str | None) -> list[str]:
    """Decode improvement suggestions from form data.

    Accepts a JSON array of strings or a newline-separated blob.

    Args:
        raw: Raw form value.

    Returns:
        List of non-empty suggestion strings.
    """
    if not raw:
        return []
    raw = raw.strip()
    if not raw:
        return []

    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return [str(item).strip() for item in parsed if str(item).strip()]
    except json.JSONDecodeError:
        pass

    return [line.strip() for line in raw.splitlines() if line.strip()]
