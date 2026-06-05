"""DMRG utilities: default sweep schedule."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_SCHEDULE_WAYPOINTS = [0, 4, 8, 12, 16]
_SCHEDULE_MAX_SWEEPS = 30
_SCHEDULE_NOISES = [1e-4, 1e-4, 5e-5, 5e-5, 0.0]
_SCHEDULE_THRDS = [1e-8, 1e-10, 1e-12, 1e-12, 1e-12]


def _default_dmrg_schedule(
    max_bond_dim: int = 500,
) -> tuple[list[int], list[float], list[float]]:
    """Define a default DMRG sweep schedule for the Block2 ``DMRGDriver.dmrg`` method.

    Builds a 30-sweep schedule from fixed waypoints at sweeps ``[0, 4, 8, 12, 16]``.
    The first segment uses ``max_bond_dim // 2``; all subsequent segments use
    ``max_bond_dim``.  Noise and thresholds are tightened at each waypoint.

    Parameters
    ----------
    max_bond_dim : int
        Target bond dimension.  Must be a positive multiple of 4
        so that both ``max_bond_dim`` and ``max_bond_dim // 2`` are even.

    Returns:
    -------
    bond_dims : list of int
        Bond dimension for each sweep.
    noises : list of float
        Perturbative noise injected at each sweep.
    thrds : list of float
        Davidson convergence threshold for each sweep.

    Note:
        Suggested in: https://github.com/block-hczhai/block2-preview/issues/63
    """
    if max_bond_dim <= 0:
        raise ValueError("max_bond_dim must be a positive multiple of 4")
    if max_bond_dim % 4 != 0:
        raise ValueError(
            "max_bond_dim must be a multiple of 4 to keep all bond dimensions even"
        )

    bd_at_waypoint = [max_bond_dim // 2] + [max_bond_dim] * (
        len(_SCHEDULE_WAYPOINTS) - 1
    )

    bond_dims: list[int] = []
    noises: list[float] = []
    thrds: list[float] = []
    for i, start in enumerate(_SCHEDULE_WAYPOINTS):
        end = (
            _SCHEDULE_WAYPOINTS[i + 1]
            if i + 1 < len(_SCHEDULE_WAYPOINTS)
            else _SCHEDULE_MAX_SWEEPS
        )
        n = end - start
        bond_dims.extend([bd_at_waypoint[i]] * n)
        noises.extend([_SCHEDULE_NOISES[i]] * n)
        thrds.extend([_SCHEDULE_THRDS[i]] * n)

    return bond_dims, noises, thrds
