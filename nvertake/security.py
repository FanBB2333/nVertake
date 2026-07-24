"""Helpers for keeping credentials out of previews and diagnostics."""

from __future__ import annotations

import re
from typing import Dict, Mapping


REDACTED_VALUE = "<redacted>"
_SECRET_ENV_NAME = re.compile(
    r"(?:^|_)(?:"
    r"API_KEY|ACCESS_KEY|AUTH|CREDENTIAL|CREDENTIALS|"
    r"PASSWORD|PASSWD|PRIVATE_KEY|SECRET|TOKEN"
    r")(?:_|$)",
    re.IGNORECASE,
)


def is_secret_environment_name(name: str) -> bool:
    """Return whether an environment-variable name is likely confidential."""

    return bool(_SECRET_ENV_NAME.search(str(name)))


def redact_environment(
    environment: Mapping[str, str],
    *,
    reveal_secrets: bool = False,
) -> Dict[str, str]:
    """Copy an environment mapping, replacing likely credential values."""

    if reveal_secrets:
        return {str(name): str(value) for name, value in environment.items()}
    return {
        str(name): (
            REDACTED_VALUE
            if is_secret_environment_name(str(name))
            else str(value)
        )
        for name, value in environment.items()
    }
