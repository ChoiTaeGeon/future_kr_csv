"""
System Logging Module
- Centralized, thread-safe file logging with rotation (10MB max, backupCount 5).
- Generates:
  - output/logs/error.log: All warnings, errors, and unhandled exceptions with full tracebacks.
  - output/logs/sync.log: Detailed logs for CSV ingestion and DuckDB synchronization.
  - output/logs/app.log: General server events and API access logs.
"""
import os
import sys
import logging
from logging.handlers import RotatingFileHandler
import traceback
from pathlib import Path
from config import LOGS_DIR

LOGS_DIR.mkdir(parents=True, exist_ok=True)

DETAILED_FORMATTER = logging.Formatter(
    fmt="[%(asctime)s] [%(levelname)s] [%(name)s:%(lineno)d] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
SYNC_FORMATTER = logging.Formatter(
    fmt="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

def _create_rotating_handler(filename: str, formatter: logging.Formatter, level=logging.INFO) -> RotatingFileHandler:
    file_path = LOGS_DIR / filename
    handler = RotatingFileHandler(
        file_path,
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8"
    )
    handler.setFormatter(formatter)
    handler.setLevel(level)
    return handler

error_logger = logging.getLogger("QuantError")
error_logger.setLevel(logging.WARNING)
if not error_logger.handlers:
    error_logger.addHandler(_create_rotating_handler("error.log", DETAILED_FORMATTER, level=logging.WARNING))

sync_logger = logging.getLogger("QuantSync")
sync_logger.setLevel(logging.INFO)
if not sync_logger.handlers:
    sync_logger.addHandler(_create_rotating_handler("sync.log", SYNC_FORMATTER, level=logging.INFO))
    sync_logger.addHandler(_create_rotating_handler("error.log", DETAILED_FORMATTER, level=logging.ERROR))

app_logger = logging.getLogger("QuantApp")
app_logger.setLevel(logging.INFO)
if not app_logger.handlers:
    app_logger.addHandler(_create_rotating_handler("app.log", DETAILED_FORMATTER, level=logging.INFO))

def log_error(msg: str, exc: Exception = None):
    if exc is not None:
        tb = traceback.format_exc()
        if tb.strip() == "NoneType: None":
            tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        error_logger.error(f"{msg}\nException: {exc}\nTraceback:\n{tb}")
    else:
        error_logger.error(msg)

def log_sync(msg: str, level: str = "INFO", exc: Exception = None):
    lvl = level.upper()
    if lvl == "ERROR":
        if exc is not None:
            tb = traceback.format_exc()
            sync_logger.error(f"{msg} | Exception: {exc}\n{tb}")
        else:
            sync_logger.error(msg)
    elif lvl == "WARNING":
        sync_logger.warning(msg)
    else:
        sync_logger.info(msg)

def log_info(msg: str):
    app_logger.info(msg)
