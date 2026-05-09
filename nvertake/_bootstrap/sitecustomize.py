"""Auto-loaded by Python when nVertake prepends this directory to PYTHONPATH."""

from __future__ import annotations

import os
import sys


if os.environ.get("NVERTAKE_AUTO_PRIORITY", "") in {"1", "true", "yes", "on"}:
    try:
        from nvertake.auto_priority import install_from_env

        install_from_env()
    except Exception as exc:
        if os.environ.get("NVERTAKE_AUTO_PRIORITY_STRICT", "") in {
            "1",
            "true",
            "yes",
            "on",
        }:
            raise
        if os.environ.get("NVERTAKE_AUTO_PRIORITY_QUIET", "") not in {
            "1",
            "true",
            "yes",
            "on",
        }:
            print(
                f"nVertake: failed to install auto priority hook: {exc}",
                file=sys.stderr,
            )
