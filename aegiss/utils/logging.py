"""Logging configuration for AEGISS using structlog.

``import logging`` below resolves to the stdlib via Python 3 absolute imports,
not to this module.
"""

from __future__ import annotations

import logging  # stdlib — not this module
import sys
from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from pathlib import Path

    from structlog.types import Processor

# ── Format constants ─────────────────────────────────────────────────────────

TIMESTAMP_FMT: str = "%Y-%m-%d %H:%M:%S"

# Noisy third-party loggers clamped to WARNING regardless of root level.
_SILENCED: tuple[str, ...] = ("matplotlib", "h5py")


# ── Public API ────────────────────────────────────────────────────────────────


def configure_logging(
    level: int = logging.INFO,
    log_file: str | Path | None = "log.out",
    timestamp_fmt: str = TIMESTAMP_FMT,
) -> None:
    """Configure structlog over stdlib logging.

    Args:
        level: Root log level (e.g. ``logging.DEBUG``, ``logging.INFO``).
        log_file: Destination file, or ``None`` to write to stdout.
        timestamp_fmt: ``strftime`` format for the timestamp column.
    """
    shared_processors: list[Processor] = [
        structlog.stdlib.add_log_level,
        # structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt=timestamp_fmt),
        structlog.stdlib.PositionalArgumentsFormatter(),
    ]

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        processor=structlog.dev.ConsoleRenderer(colors=False),
        foreign_pre_chain=shared_processors,
    )

    handler: logging.Handler = (
        logging.FileHandler(log_file)
        if log_file is not None
        else logging.StreamHandler(sys.stdout)
    )
    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(level)
    root.addHandler(handler)

    for name in _SILENCED:
        logging.getLogger(name).setLevel(logging.WARNING)
