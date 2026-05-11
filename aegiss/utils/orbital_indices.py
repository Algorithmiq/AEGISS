"""Utilities for partitioning MO indices into orbital subspaces."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def resolve_nelecas(nelecas: int | tuple[int, int]) -> tuple[int, int]:
    """Normalise *nelecas* to an (nalpha, nbeta) pair.

    Args:
        nelecas (int or tuple[int, int]): Either a total electron count (PySCF closed-shell convention) or an
            explicit ``(nalpha, nbeta)`` pair.  Odd integers are treated as
            high-spin doublets (nalpha = ceil, nbeta = floor).

    Returns:
        nalpha, nbeta (tuple[int, int]):"""
    if isinstance(nelecas, tuple):
        return nelecas
    return (nelecas + 1) // 2, nelecas // 2


def combine_candidate_entropies(
    occ_orbs: list[tuple[int, float]], virt_orbs: list[tuple[int, float]]
) -> list[tuple[int, float]]:
    """Merge and sort occupied and virtual orbital candidate lists.

    Args:
        occ_orbs (list of tuple[int, float]): Occupied orbital index/entropy pairs.
        virt_orbs (list of tuple[int, float]): Virtual orbital index/entropy pairs.

    Returns:
        combined_space (list of tuple[int, float]): Merged list sorted by orbital index."""
    combined_space = occ_orbs + virt_orbs
    combined_space.sort(key=lambda x: x[0])
    return combined_space


def get_orbital_spaces(
    mf: Any,
    orbital_list: list[int],
    *,
    unique_virt_space: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Partition MO indices into four subspaces.

    Args:
        mf (PySCF SCF object): Converged mean-field object exposing ``mo_occ``.
        orbital_list (list of int): Orbital indices (0-based) that define the active / selected space.
        unique_virt_space (bool): If ``True``, the virtual space is partitioned into active and frozen
            virtuals based on *orbital_list*. If ``False``, all virtual orbitals
            are considered active and *orbital_list* only affects the occupied space.

    Returns:
        frozen_occ (np.ndarray of int): Occupied orbitals **not** in *orbital_list* (core / frozen occupied).
        active_occ (np.ndarray of int): Occupied orbitals that **are** in *orbital_list*.
        active_virt (np.ndarray of int): Virtual orbitals that **are** in *orbital_list* (empty when
            *unique_virt_space* is ``True``).
        frozen_virt (np.ndarray of int): Virtual orbitals **not** in *orbital_list* (external / frozen virtual).

    Example:
    >>> frozen_occ, active_occ, active_virt, frozen_virt = get_orbital_spaces(mf, [3, 4, 5, 6])
"""
    selected = set(orbital_list)
    if not selected:
        logger.info(
            "Using all the occupied orbitals and all virtual orbitals as"
            " frozen and active spaces, respectively."
        )
        selected = set(range(len(mf.mo_occ)))
    occ_mask = mf.mo_occ > 0

    all_occ = np.where(occ_mask)[0]
    all_virt = np.where(~occ_mask)[0]

    frozen_occ = np.array([i for i in all_occ if i not in selected])
    active_occ = np.array([i for i in all_occ if i in selected])

    if unique_virt_space:
        return frozen_occ, active_occ, np.array([], dtype=int), np.array(list(all_virt))

    active_virt = np.array([i for i in all_virt if i in selected])
    frozen_virt = np.array([i for i in all_virt if i not in selected])

    return frozen_occ, active_occ, active_virt, frozen_virt


def get_subspaces_from_cas(
    mf: Any,
    ncas: int,
    nelecas: int | tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Partition MO indices into active and virtual spaces from a CAS definition.

    Args:
        mf (PySCF SCF object): Converged mean-field object exposing ``mo_occ``.
        ncas (int): Number of active orbitals in the CAS.
        nelecas (int or tuple[int, int]): Number of active electrons, either as a total count or an explicit
            ``(nalpha, nbeta)`` pair (same convention as PySCF's CASCI).

    Returns:
        active_orbs (np.ndarray of int): Orbital indices (0-based) of the active space.
        virt_orbs (np.ndarray of int): Orbital indices (0-based) of the virtual space (not in the active space)."""
    logger.info(
        "----------DMRG CAS Subspace Selection (ncas=%d, nelecas=%s)----------",
        ncas,
        nelecas,
    )
    nalpha, nbeta = resolve_nelecas(nelecas)
    neletot = int(np.sum(mf.mo_occ))
    nmos = mf.mo_coeff.shape[1]

    if ncas > nmos:
        msg = "Number of active orbitals cannot exceed total number of MOs."
        raise ValueError(msg)
    if nbeta < 0 or nalpha < nbeta:
        msg = "nelecas must satisfy nalpha >= nbeta >= 0."
        raise ValueError(msg)
    if nalpha + nbeta > neletot:
        msg = "Number of active electrons cannot exceed total number of electrons."
        raise ValueError(msg)

    nocc = np.count_nonzero(mf.mo_occ > 0)
    logger.info(
        "Total MOs: %d, Occupied MOs: %d, Virtual MOs: %d", nmos, nocc, nmos - nocc
    )

    n_active_occ = nalpha  # highest-energy nalpha occupied orbitals enter the CAS
    n_active_virt = ncas - n_active_occ
    if nmos - nocc - n_active_virt < 0:
        msg = "Not enough virtual orbitals to accommodate the active space."
        raise ValueError(msg)
    logger.info("Active occupied orbitals: %d", n_active_occ)
    logger.info("Active virtual orbitals: %d", n_active_virt)

    active_occ = np.arange(nocc - n_active_occ, nocc)
    active_virt = np.arange(nocc, nocc + n_active_virt)
    logger.info("Active occupied orbital indices: %s", active_occ)
    logger.info("Active virtual orbital indices: %s\n\n", active_virt)

    return active_occ, active_virt
