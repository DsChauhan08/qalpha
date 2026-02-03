"""
Logging configuration for Quantum Alpha.
"""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from typing import Optional


def configure_logging(level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
    root = logging.getLogger()

    for handler in list(root.handlers):
        root.removeHandler(handler)

    level_name = str(level).upper()
    root.setLevel(level_name)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    root.addHandler(stream_handler)

    if log_file:
        file_handler = RotatingFileHandler(log_file, maxBytes=5_000_000, backupCount=3)
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

    logging.captureWarnings(True)
    return root
