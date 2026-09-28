"""Centralized logging configuration.

Windows note: Vietnamese diacritics break through a cp1252/cp437 console. We
reconfigure the standard streams to UTF-8 so ``print``/``logging`` never crash
on accented text.
"""

from __future__ import annotations

import logging
import sys

_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"


def configure_logging(level: int = logging.INFO) -> None:
    """Configure root logging with a consistent format and UTF-8 output."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8")

    logging.basicConfig(level=level, format=_FORMAT, stream=sys.stdout, force=True)
