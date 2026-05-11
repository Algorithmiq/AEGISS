"""DMRG utilities: default schedule and quantum mutual information computation."""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from pyblock2.driver.core import DMRGDriver

logger = logging.getLogger(__name__)

_DEFAULT_MEM_GIB = 4


def _validate_mem_env() -> int:
    """Return ``stack_mem`` in bytes from the ``MEM`` environment variable.

    ``MEM`` must be a positive integer number of GiB.  If it is unset, not a
    valid integer, or negative, the default of ``_DEFAULT_MEM_GIB`` GiB is
    used and a warning is logged.  Explicitly setting ``MEM=0`` is rejected
    with ``ValueError`` because zero bytes is never a valid stack size.

    Raises:
    ------
    ValueError
        If ``MEM`` is ``"0"``.
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

    if mem_gib == 0:
        raise ValueError(
            f"MEM environment variable must be a positive integer, got {mem_gib}."
        )

    if mem_gib < 0:
        logger.warning(
            "MEM=%d is negative; using default %d GiB.", mem_gib, _DEFAULT_MEM_GIB
        )
        return _DEFAULT_MEM_GIB * (1 << 30)

    return mem_gib * (1 << 30)


def _default_dmrg_schedule(
    init_bond_dim: int = 20,
) -> tuple[list[int], list[float], list[float]]:
    """Define a default DMRG sweep schedule for the Block2 ``DMRGDriver.solve`` method.

    Builds a 16-sweep schedule that starts at *init_bond_dim* and grows through
    multipliers ``[1x, 2x, 4x, 6x]``, with noise and convergence thresholds
    tightened in lock-step.

    Parameters
    ----------
    init_bond_dim : int
        Initial bond dimension for the first sweeps. Subsequent bond dimensions
        are integer multiples of this value.

    Returns:
    -------
    bond_dims : list of int
        Bond dimension for each sweep.
    noises : list of float
        Perturbative noise injected at each sweep.
    thrds : list of float
        Davidson convergence threshold for each sweep.
    """
    if init_bond_dim <= 0:
        raise ValueError("init_bond_dim must be a positive integer")

    multipliers = [1, 1, 1, 1, 2, 2, 2, 2, 4, 4, 4, 4, 6, 6, 6, 6]
    bond_dims = [init_bond_dim * m for m in multipliers]
    noises = [
        1e-4,
        1e-4,
        1e-4,
        1e-4,
        1e-5,
        1e-5,
        1e-5,
        1e-5,
        1e-6,
        1e-6,
        1e-6,
        1e-6,
        1e-7,
        1e-7,
        0.0,
        0.0,
    ]
    thrds = [
        1e-6,
        1e-6,
        1e-7,
        1e-7,
        1e-8,
        1e-8,
        1e-9,
        1e-9,
        1e-10,
        1e-10,
        1e-10,
        1e-10,
        1e-10,
        1e-10,
        1e-10,
        1e-10,
    ]
    return bond_dims, noises, thrds


def compute_qmi(
    driver: DMRGDriver,
    ket: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute quantum mutual information (QMI) from a DMRG state.

    Parameters
    ----------
    driver : DMRGDriver
        Block2 DMRG driver after a completed calculation.
    ket : MPS
        Many-particle state (MPS) from which entropies are extracted.

    Returns:
    -------
    ordm1 : np.ndarray
        Single-orbital entropies.
    ordm2 : np.ndarray
        Two-orbital entropies.
    minfo : np.ndarray
        Quantum mutual information matrix.
    """
    ordm1 = driver.get_orbital_entropies(ket, orb_type=1)
    ordm2 = driver.get_orbital_entropies(ket, orb_type=2)

    minfo = (
        0.5 * (ordm1[:, None] + ordm1[None, :] - ordm2) * (1 - np.identity(len(ordm1)))
    )

    return ordm1, ordm2, minfo
