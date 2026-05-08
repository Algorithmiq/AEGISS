"""Centralized environment variable configuration for AEGISS resource management."""

from __future__ import annotations

import os


def omp_num_threads() -> int:
    """Return the number of OpenMP threads (``OMP_NUM_THREADS``, default: 4)."""
    return int(os.environ.get("OMP_NUM_THREADS", "4"))


def scratch_dir() -> str:
    """Return the DMRG scratch directory (``SCRATCH_DIR``, default: ``'./tmp'``)."""
    return os.environ.get("SCRATCH_DIR", "./tmp")


def pyscf_max_memory() -> int:
    """Return the PySCF max memory in MB (``PYSCF_MAX_MEMORY``, default: 4000)."""
    value = os.environ.get("PYSCF_MAX_MEMORY", "4000")
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(
            f"Invalid value for PYSCF_MAX_MEMORY: {value!r}. Expected an integer number of MB."
        ) from exc


def pyscf_scratch_dir() -> str | None:
    """Return the PySCF scratch directory (``PYSCF_SCRATCH_DIR``), or ``None`` if unset."""
    value = os.environ.get("PYSCF_SCRATCH_DIR")
    if value is None:
        return None
    value = value.strip()
    return value or None
