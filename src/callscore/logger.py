"""Structured logging configuration for callscore."""

import os

import structlog
from structlog.types import Processor


def _add_call_id(
    _logger: structlog.types.WrappedLogger, _method_name: str, event_dict: dict
) -> dict:
    """Add call_id to log if present in context."""
    return event_dict


def configure_logging() -> None:
    """Configure structlog for JSON output."""
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()

    processors: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        _add_call_id,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.JSONRenderer(),
    ]

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Also configure standard library logging
    import logging

    logging.basicConfig(
        format="%(message)s",
        level=getattr(logging, log_level, logging.INFO),
    )


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Get a structured logger instance."""
    return structlog.get_logger(name)
