from __future__ import annotations

import logging
from typing import Optional


_DEF_FORMAT = "%(asctime)s | %(levelname)s | %(message)s"


def get_logger(name: str = "sr", level: int = logging.INFO, fmt: str = _DEF_FORMAT) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(level)
    handler = logging.StreamHandler()
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)
    logger.propagate = False
    return logger
