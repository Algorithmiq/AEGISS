"""DMRG utilities: default schedule and quantum mutual information computation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from pyblock2.driver.core import DMRGDriver


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


def _refinement_schedule(
    n_sweeps: int,
    bond_dim: int,
) -> tuple[list[int], list[float], list[float]]:
    """Build a fixed-bond-dim, zero-noise schedule for state-specific refinement.

    Parameters
    ----------
    n_sweeps : int
        Number of refinement sweeps (must be positive).
    bond_dim : int
        Bond dimension to hold constant throughout (typically the final SA bond dim).

    Returns:
    -------
    bond_dims : list of int
    noises : list of float
    thrds : list of float
    """
    if n_sweeps <= 0:
        raise ValueError("n_sweeps must be a positive integer")
    if bond_dim <= 0:
        raise ValueError("bond_dim must be a positive integer")
    noises = [1e-5] * (n_sweeps - 4) + [0.0]
    return [bond_dim] * n_sweeps, noises, [1e-10] * n_sweeps


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
        Single-root MPS from which entropies are extracted.  For multi-root
        MPS, split individual roots first via ``driver.split_mps``.

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
