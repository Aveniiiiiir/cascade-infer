"""Logging helpers. Deliberately stdlib-only (Windows-safe, no fork assumptions)."""

from __future__ import annotations

import logging

_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
_configured = False


def _ensure_configured() -> None:
    global _configured
    if not _configured:
        logging.basicConfig(level=logging.INFO, format=_FORMAT)
        _configured = True


def get_logger(name: str) -> logging.Logger:
    _ensure_configured()
    return logging.getLogger(f"cascade.{name}")


__all__ = ["get_logger"]
