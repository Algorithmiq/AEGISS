"""DMRG utilities: default sweep schedule."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


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
