# Copyright 2026 Algorithmiq
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
    from structlog.types import Processor

# ── Format constants ─────────────────────────────────────────────────────────

TIMESTAMP_FMT: str = "%Y-%m-%d %H:%M:%S"

# Noisy third-party loggers clamped to WARNING regardless of root level.
_SILENCED: tuple[str, ...] = ("matplotlib", "h5py")


# ── Public API ────────────────────────────────────────────────────────────────


def configure_logging(
    level: int = logging.INFO,
    timestamp_fmt: str = TIMESTAMP_FMT,
) -> None:
    """Configure structlog over stdlib logging, writing to stdout.

    Replaces any handlers on the root logger and clamps noisy third-party
    loggers (matplotlib, h5py) to ``WARNING``. Redirect stdout from the shell to
    control the destination:

        python main.py > log.out        # local
        #SBATCH -o slurm-%j.out         # SLURM

    Args:
        level (int): Root logger level (default ``logging.INFO``).
        timestamp_fmt (str): ``strftime`` format for the timestamp of each log
            line (default ``"%Y-%m-%d %H:%M:%S"``).
    """
    shared_processors: list[Processor] = [
        structlog.stdlib.add_log_level,
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

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(level)
    root.addHandler(handler)

    for name in _SILENCED:
        logging.getLogger(name).setLevel(logging.WARNING)
