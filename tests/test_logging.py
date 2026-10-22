"""Tests for aegiss.utils.logging.configure_logging."""

from __future__ import annotations

import logging
import re
import sys
from typing import TYPE_CHECKING

import pytest
import structlog

from aegiss.utils.logging import _SILENCED, configure_logging

if TYPE_CHECKING:
    from collections.abc import Iterator

_LOGGER = "aegiss.test_logging"


@pytest.fixture(autouse=True)
def _restore_logging() -> Iterator[None]:
    """configure_logging rewires the root logger and structlog globally; undo it."""
    root = logging.getLogger()
    handlers, level = root.handlers[:], root.level
    silenced = {name: logging.getLogger(name).level for name in _SILENCED}
    yield
    root.handlers[:] = handlers
    root.setLevel(level)
    for name, lvl in silenced.items():
        logging.getLogger(name).setLevel(lvl)
    structlog.reset_defaults()


class TestConfigureLogging:
    def test_installs_single_stdout_handler(self) -> None:
        configure_logging()
        configure_logging()  # calling twice must not stack handlers
        root = logging.getLogger()
        assert len(root.handlers) == 1
        handler = root.handlers[0]
        assert isinstance(handler, logging.StreamHandler)
        assert handler.stream is sys.stdout

    @pytest.mark.parametrize("level", [logging.DEBUG, logging.INFO, logging.WARNING])
    def test_sets_root_level(self, level: int) -> None:
        configure_logging(level=level)
        assert logging.getLogger().level == level

    def test_silences_noisy_third_party_loggers(self) -> None:
        configure_logging(level=logging.DEBUG)
        for name in ("matplotlib", "h5py"):
            assert logging.getLogger(name).level == logging.WARNING

    def test_stdlib_record_is_rendered_to_stdout(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        configure_logging()
        logging.getLogger(_LOGGER).info("selected %d orbitals", 6)
        out = capsys.readouterr().out
        assert "selected 6 orbitals" in out
        assert "info" in out
        assert re.search(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", out)

    def test_records_below_level_are_dropped(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        configure_logging(level=logging.INFO)
        logging.getLogger(_LOGGER).debug("per-orbital detail")
        assert "per-orbital detail" not in capsys.readouterr().out

    def test_debug_level_shows_debug_records(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        configure_logging(level=logging.DEBUG)
        logging.getLogger(_LOGGER).debug("per-orbital detail")
        assert "per-orbital detail" in capsys.readouterr().out

    def test_custom_timestamp_format(self, capsys: pytest.CaptureFixture[str]) -> None:
        configure_logging(timestamp_fmt="[%H:%M]")
        logging.getLogger(_LOGGER).warning("hello")
        assert re.search(r"\[\d{2}:\d{2}\]", capsys.readouterr().out)

    def test_structlog_logger_goes_through_same_handler(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        configure_logging()
        structlog.get_logger(_LOGGER).info("from structlog")
        assert "from structlog" in capsys.readouterr().out
