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

"""Utilities for partitioning MO indices into orbital subspaces."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from pyscf.data.elements import chemcore

from aegiss.utils.orb_display import to_display_all

logger = logging.getLogger(__name__)


def resolve_nelecas(nelecas: int | tuple[int, int]) -> tuple[int, int]:
    """Normalise *nelecas* to an (nalpha, nbeta) pair.

    Args:
        nelecas (int or tuple[int, int]): Either a total electron count (PySCF closed-shell convention) or an
            explicit ``(nalpha, nbeta)`` pair.  Odd integers are treated as
            high-spin doublets (nalpha = ceil, nbeta = floor).

    Returns:
        nalpha, nbeta (tuple[int, int]):
    """
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
        combined_space (list of tuple[int, float]): Merged list sorted by orbital index.
    """
    combined_space = occ_orbs + virt_orbs
    combined_space.sort(key=lambda x: x[0])
    return combined_space


def valence_occupied_orbitals(mf: Any, *, exclude_core: bool = True) -> list[int]:
    """Return the occupied MO indices, optionally excluding the chemical core.

    Convenience helper for building a ``restricted_space`` of plain orbital
    indices (see :func:`aegiss.active_space.ao_projection.select_by_ao_projection`)
    without needing entropy scores.

    Args:
        mf (PySCF SCF object): Converged mean-field object exposing ``mo_occ`` and ``mol``.
        exclude_core (bool, keyword-only): If ``True`` (default), drop the frozen chemical-core orbitals as
            defined by :func:`pyscf.data.elements.chemcore` (1s for first-row
            atoms, 1s2s2p for second-row, and so on). The core is assumed to
            occupy the lowest ``ncore`` orbital indices, following PySCF's
            frozen-core convention; this holds for canonical SCF orbitals but
            not necessarily for localized orbitals.

    Returns:
        orbitals (list of int): Occupied orbital indices (0-based), sorted ascending.
    """
    occ = np.where(mf.mo_occ > 0)[0]
    if not exclude_core:
        return [int(i) for i in occ]
    ncore = chemcore(mf.mol)
    valence = occ[occ >= ncore]
    logger.info(
        "Excluding %d chemical-core orbital(s); %d valence occupied orbital(s) remain.",
        ncore,
        len(valence),
    )
    return [int(i) for i in valence]


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
            An empty list selects every orbital: no occupied orbital is frozen
            and, when *unique_virt_space* is ``False``, every virtual orbital is
            active.
        unique_virt_space (bool): If ``True``, *orbital_list* only affects the occupied space:
            every virtual orbital is returned in *frozen_virt* and *active_virt*
            is empty. If ``False``, the virtual space is partitioned into active
            and frozen virtuals based on *orbital_list*.

    Returns:
        frozen_occ (np.ndarray of int): Occupied orbitals **not** in *orbital_list* (core / frozen occupied).
        active_occ (np.ndarray of int): Occupied orbitals that **are** in *orbital_list*.
        active_virt (np.ndarray of int): Virtual orbitals that **are** in *orbital_list* (empty when
            *unique_virt_space* is ``True``).
        frozen_virt (np.ndarray of int): Virtual orbitals **not** in *orbital_list* (external / frozen virtual);
            all virtual orbitals when *unique_virt_space* is ``True``.
    """
    selected = set(orbital_list)
    if not selected:
        logger.info(
            "Empty orbital list: treating all orbitals as selected"
            " (no frozen occupied orbitals)."
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
    """Split the CAS window into its occupied and virtual parts.

    The window holds the ``ncas`` orbitals directly above the closed frozen
    core. Its orbitals are split at the reference HOMO/LUMO boundary.

    Args:
        mf (PySCF SCF object): Converged mean-field object exposing ``mo_occ``.
        ncas (int): Number of active orbitals in the CAS.
        nelecas (int or tuple[int, int]): Number of active electrons, either as a total count or an explicit
            ``(nalpha, nbeta)`` pair (same convention as PySCF's CASCI).

    Returns:
        active_occ (np.ndarray of int): Orbital indices (0-based) of the CAS window that are occupied
            in the reference (``mo_occ > 0``, SOMOs included).
        active_virt (np.ndarray of int): Orbital indices (0-based) of the CAS window that are virtual
            in the reference.

    Raises:
        ValueError: If *ncas* exceeds the number of MOs, if *nelecas* does not
            satisfy ``nalpha >= nbeta >= 0``, if it exceeds the total electron
            count, if the window does not fit above the frozen core, or, for an
            unrestricted reference, if the active spin ``nalpha - nbeta``
            differs from the reference spin.
    """
    logger.info(
        "----------DMRG CAS Subspace Selection (ncas=%d, nelecas=%s)----------",
        ncas,
        nelecas,
    )
    nalpha, nbeta = resolve_nelecas(nelecas)
    mo_occ = np.asarray(mf.mo_occ)
    # Collapse spin so the logic is identical for restricted (1-D ``mo_occ``) and
    # unrestricted (2-D, shape ``(2, nmo)``) references: a per-spatial-orbital
    # occupation with values 0 / 1 / 2.
    occ_per_orb = mo_occ.sum(axis=0) if mo_occ.ndim == 2 else mo_occ
    neletot = round(float(occ_per_orb.sum()))
    # UHF ``mo_coeff`` is a tuple/array of two (nao, nmo) blocks, restricted is a
    # single (nao, nmo) array; ``shape[-1]`` is nmo in both cases.
    nmos = np.asarray(mf.mo_coeff).shape[-1]
    nact = nalpha + nbeta

    if ncas > nmos:
        msg = "Number of active orbitals cannot exceed total number of MOs."
        raise ValueError(msg)
    if nbeta < 0 or nalpha < nbeta:
        msg = "nelecas must satisfy nalpha >= nbeta >= 0."
        raise ValueError(msg)
    if nact > neletot:
        msg = "Number of active electrons cannot exceed total number of electrons."
        raise ValueError(msg)

    # For an unrestricted reference, alpha and beta are distinct orbital sets, so a
    # single active-orbital window only exists when their frozen cores coincide:
    # ncore_a = Na - nalpha == Nb - nbeta = ncore_b, i.e. the reference spin
    # polarisation must equal the requested active spin. (A *restricted* reference
    # has one orbital set, so an open-shell active target on it is always fine.)
    if mo_occ.ndim == 2:
        na_ref = round(float(mo_occ[0].sum()))
        nb_ref = round(float(mo_occ[1].sum()))
        if (na_ref - nb_ref) != (nalpha - nbeta):
            msg = (
                f"Unrestricted reference: active spin (nalpha - nbeta = "
                f"{nalpha - nbeta}) must equal the reference spin "
                f"(Na - Nb = {na_ref - nb_ref}); otherwise the alpha and beta "
                "active windows differ and no single CAS partition exists."
            )
            raise ValueError(msg)

    # The frozen core is closed: ``ncore`` doubly-occupied spatial orbitals sit
    # below the active window, placed by the *total* active electron count (this
    # matches pyscf CASCI/UCASCI). The (nalpha, nbeta) split sets the DMRG target
    # occupation, not the window position.
    ncore = (neletot - nact) // 2
    if ncore + ncas > nmos:
        msg = "Not enough virtual orbitals to accommodate the active space."
        raise ValueError(msg)

    # The occupied/virtual label reflects the *reference* occupation: any orbital
    # with occ > 0 is "occupied" (including the singly-occupied SOMOs of an
    # open-shell reference), so the split sits at the reference HOMO boundary.
    nocc = int(np.count_nonzero(occ_per_orb > 0))
    active = np.arange(ncore, ncore + ncas)
    active_occ = active[active < nocc]
    active_virt = active[active >= nocc]

    logger.info(
        "Total MOs: %d, Occupied MOs: %d, Virtual MOs: %d", nmos, nocc, nmos - nocc
    )
    logger.info("Frozen (doubly-occupied) core orbitals: %d", ncore)
    logger.info("Active occupied orbital indices: %s", to_display_all(active_occ))
    logger.info("Active virtual orbital indices: %s\n\n", to_display_all(active_virt))

    # Return values stay 0-based so downstream slicing/logic is unaffected.
    return active_occ, active_virt
