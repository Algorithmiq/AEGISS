"""Display base for orbital indices.

AEGISS stores and computes every orbital index 0-based (PySCF convention).
Some users, however, prefer to *read* and *type* indices 1-based (the Fortran /
quantum-chemistry convention used by ORCA, Molpro, Molcas, ...).

This module owns that single presentation concern. The chosen base affects only:

* how indices are **printed** (logs, plot labels, cube-file names), and
* how indices the user **passes in** are interpreted before being converted
  back to the internal 0-based representation.

It never affects stored values, array slicing, or the indices returned by the
public API -- those are always 0-based so results round-trip safely.

Resolution precedence (highest first):

1. the ``AEGISS_ORB_BASE`` environment variable (runtime override), then
2. ``orbital_base`` in the persisted config file (see :func:`config_path`,
   written once at install time via ``aegiss config set-base``), then
3. the default of ``0``.
"""

from __future__ import annotations

import logging
import os
import tomllib
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable

logger = logging.getLogger(__name__)

# Public: shared with the CLI (aegiss.cli) so the accepted bases and the
# override variable name have a single source of truth.
ENV_VAR = "AEGISS_ORB_BASE"
VALID_BASES = (0, 1)

_CONFIG_KEY = "orbital_base"
_DEFAULT_BASE = 0


def config_path() -> Path:
    """Return the path to the persisted AEGISS config file.

    Honours ``XDG_CONFIG_HOME`` and falls back to ``~/.config``. The file is
    not required to exist; callers must treat a missing file as "unset".

    Returns:
        Path: ``$XDG_CONFIG_HOME/aegiss/config.toml`` (or the ``~/.config``
        equivalent).
    """
    root = os.environ.get("XDG_CONFIG_HOME", "").strip()
    base_dir = Path(root) if root else Path.home() / ".config"
    return base_dir / "aegiss" / "config.toml"


def _validate(value: int, *, source: str) -> int | None:
    """Return *value* if it is a valid base, else warn and return ``None``."""
    if value in VALID_BASES:
        return value
    logger.warning(
        "%s=%r is not a valid orbital base (expected one of %s); ignoring.",
        source,
        value,
        VALID_BASES,
    )
    return None


def _base_from_env() -> int | None:
    """Resolve the base from the environment variable, or ``None`` if unset."""
    raw = os.environ.get(ENV_VAR)
    if raw is None:
        return None
    stripped = raw.strip()
    try:
        return _validate(int(stripped), source=ENV_VAR)
    except ValueError:
        logger.warning("%s=%r is not an integer; ignoring.", ENV_VAR, stripped)
        return None


def _base_from_config() -> int | None:
    """Resolve the base from the persisted config file, or ``None`` if absent."""
    path = config_path()
    if not path.is_file():
        return None
    try:
        with path.open("rb") as handle:
            data = tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        logger.warning("Could not read %s: %s; ignoring.", path, exc)
        return None
    if _CONFIG_KEY not in data:
        return None
    value = data[_CONFIG_KEY]
    if not isinstance(value, int) or isinstance(value, bool):
        logger.warning("%s in %s is not an integer; ignoring.", _CONFIG_KEY, path)
        return None
    return _validate(value, source=f"{path}:{_CONFIG_KEY}")


@lru_cache(maxsize=1)
def orbital_base() -> int:
    """Return the configured display base for orbital indices (``0`` or ``1``).

    Applies the precedence documented in the module docstring: environment
    variable, then config file, then the default of ``0``. The result is cached
    for the lifetime of the process; call :func:`reset_cache` after mutating the
    environment or config file (e.g. in tests).

    Returns:
        int: ``0`` (default) or ``1``.
    """
    for resolver in (_base_from_env, _base_from_config):
        base = resolver()
        if base is not None:
            return base
    return _DEFAULT_BASE


def reset_cache() -> None:
    """Clear the cached :func:`orbital_base` result.

    Call this after changing ``AEGISS_ORB_BASE`` or rewriting the config file
    within a running process (primarily in tests).
    """
    orbital_base.cache_clear()


def to_display(index: int) -> int:
    """Convert a stored 0-based *index* to the configured display base.

    Args:
        index (int): A 0-based orbital index (as stored internally).

    Returns:
        int: The index shifted into the display base for printing.
    """
    return int(index) + orbital_base()


def to_display_all(indices: Iterable[int]) -> list[int]:
    """Vectorised :func:`to_display` for an iterable of 0-based indices.

    Args:
        indices (Iterable[int]): 0-based orbital indices.

    Returns:
        list[int]: Indices shifted into the display base.
    """
    base = orbital_base()
    return [int(i) + base for i in indices]


def from_display(index: int) -> int:
    """Convert a user-supplied *index* (in the display base) to 0-based storage.

    Args:
        index (int): An orbital index as typed by the user, in the display base.

    Returns:
        int: The equivalent 0-based index for internal use.
    """
    base = orbital_base()
    idx = int(index)
    if idx < base:
        raise ValueError(
            f"Orbital index {idx} is invalid for base {base} (expected >= {base})."
        )
    return idx - base


def from_display_all(indices: Iterable[int]) -> list[int]:
    """Vectorised :func:`from_display` for user-supplied indices.

    Args:
        indices (Iterable[int]): Orbital indices as typed by the user, in the
            display base.

    Returns:
        list[int]: The equivalent 0-based indices for internal use.
    """
    base = orbital_base()
    raw = [int(i) for i in indices]
    bad = [i for i in raw if i < base]
    if bad:
        raise ValueError(
            f"Orbital indices {bad} are invalid for base {base} (expected >= {base})."
        )
    return [i - base for i in raw]
