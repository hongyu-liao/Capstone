"""
Utility helpers for logging configuration.

All code comments are written in English.
"""

import logging


def setup_basic_logging(level: int = logging.INFO) -> None:
    """
    Configure root logger with a reasonable default format.
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )


__all__ = ["setup_basic_logging"]


