"""Centralized environment variable configuration for AEGISS resource management."""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

_DEFAULT_MEM_GIB = 4


def mem() -> int:
    """Return ``stack_mem`` in bytes from the ``MEM`` environment variable.

    ``MEM`` must be a positive integer number of GiB.  If it is unset, not a
    valid integer, zero, or negative, the default of ``_DEFAULT_MEM_GIB`` GiB
    is used and a warning is logged.
    """
    raw = os.environ.get("MEM")
    if raw is None:
        return _DEFAULT_MEM_GIB * (1 << 30)

    stripped = raw.strip()
    try:
        mem_gib = int(stripped)
    except ValueError:
        logger.warning(
            "MEM=%r is not a valid integer; using default %d GiB.",
            stripped,
            _DEFAULT_MEM_GIB,
        )
        return _DEFAULT_MEM_GIB * (1 << 30)

    if mem_gib <= 0:
        logger.warning(
            "MEM=%d is not a positive integer; using default %d GiB.",
            mem_gib,
            _DEFAULT_MEM_GIB,
        )
        return _DEFAULT_MEM_GIB * (1 << 30)

    return mem_gib * (1 << 30)


def omp_num_threads() -> int:
    """Return the number of OpenMP threads (``OMP_NUM_THREADS``, default: 4)."""
    value = os.environ.get("OMP_NUM_THREADS", "4").strip() or "4"
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(
            f"Invalid value for OMP_NUM_THREADS: {value!r}. Expected an integer."
        ) from exc


def scratch_dir() -> str:
    """Return the DMRG scratch directory (``SCRATCH_DIR``, default: ``'./tmp'``)."""
    value = os.environ.get("SCRATCH_DIR", "").strip()
    return value or "./tmp"


def pyscf_max_memory() -> int:
    """Return the PySCF max memory in MB (``PYSCF_MAX_MEMORY``, default: 4000)."""
    value = os.environ.get("PYSCF_MAX_MEMORY", "4000").strip() or "4000"
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(
            f"Invalid value for PYSCF_MAX_MEMORY: {value!r}. Expected an integer number of MB."
        ) from exc


# PySCF scratch directory: set the PYSCF_TMPDIR environment variable before
# importing PySCF. PySCF reads it at import time and uses it as its temp dir.
