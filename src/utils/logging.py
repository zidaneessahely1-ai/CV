"""Logging helpers for YOLO Vision Agent."""

from __future__ import annotations

import logging
import sys
from pathlib import Path


def setup_logger(
    name: str = "yolo_agent",
    level: int = logging.INFO,
    log_dir: str | Path | None = None,
) -> logging.Logger:
    """Return a configured logger that writes to *stderr* and optionally to a file."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(level)
    fmt = logging.Formatter(
        "%(asctime)s | %(name)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console = logging.StreamHandler(sys.stderr)
    console.setFormatter(fmt)
    logger.addHandler(console)

    # Optional file handler
    if log_dir is not None:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_path / f"{name}.log")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger
