from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

from ..config import APP_NAME, LOG_DIR, LOG_FILE, LOG_LEVEL


def get_logger(name: str | None = None) -> logging.Logger:
    """
    Idempotent logger with console + rotating file handlers.
    Usage: log = get_logger(__name__); log.info("msg")
    """
    logger_name = name or APP_NAME
    logger = logging.getLogger(logger_name)

    if logger.handlers:
        return logger  # already configured

    LOG_DIR.mkdir(parents=True, exist_ok=True)

    logger.setLevel(LOG_LEVEL)

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)

    fh = RotatingFileHandler(Path(LOG_FILE), maxBytes=1_000_000, backupCount=3)
    fh.setFormatter(fmt)

    logger.addHandler(sh)
    logger.addHandler(fh)
    logger.propagate = False
    return logger
