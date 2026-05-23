from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logger(log_path: str, level: int = logging.INFO, name: Optional[str] = None) -> logging.Logger:
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    logger_name = name or str(path.resolve())
    logger = logging.getLogger(logger_name)
    logger.handlers.clear()
    logger.setLevel(level)

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    file_handler = logging.FileHandler(path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(level)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(level)
    logger.addHandler(stream_handler)

    logger.propagate = False
    return logger
