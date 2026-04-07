"""Utilities for partitioning MO indices into orbital subspaces."""
from __future__ import annotations

import numpy as np


def get_orbital_spaces(mf, orbital_list: list[int],
                       *,
                       unique_virt_space: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Partition MO indices into four subspaces.

    Parameters
    ----------
    mf : PySCF SCF object
        Converged mean-field object exposing ``mo_occ``.
    orbital_list : list of int
        Orbital indices (0-based) that define the active / selected space.
    unique_virt_space : bool
        If ``True``, the virtual space will be partitioned into active and frozen virtuals based on *orbital_list*.
        If ``False``, all virtual orbitals will be considered active, and *orbital_list* will only affect the occupied space.

    Returns
    -------
    frozen_occ : np.ndarray of int
        Occupied orbitals **not** in *orbital_list* (core / frozen occupied).
    active_occ : np.ndarray of int
        Occupied orbitals that **are** in *orbital_list*.
    active_virt : np.ndarray of int
        Virtual orbitals that **are** in *orbital_list*.
    frozen_virt : np.ndarray of int
        Virtual orbitals **not** in *orbital_list* (external / frozen virtual).

    Examples
    --------
    >>> frozen_occ, active_occ, active_virt, frozen_virt = get_orbital_spaces(mf, [3, 4, 5, 6])
    """
    selected = set(orbital_list)
    if not selected:
        print("Using all the occupied orbitals and all virtual orbitals as frozen and active spaces, respectively.")
        selected = set(range(len(mf.mo_occ)))
    occ_mask = mf.mo_occ > 0

    all_occ = np.where(occ_mask)[0]
    all_virt = np.where(~occ_mask)[0]

    frozen_occ = np.array([i for i in all_occ if i not in selected])
    active_occ = np.array([i for i in all_occ if i in selected])
    
    if unique_virt_space:
        return frozen_occ, active_occ, None, np.array([i for i in all_virt])
    
    active_virt = np.array([i for i in all_virt if i in selected])
    frozen_virt = np.array([i for i in all_virt if i not in selected])

    return frozen_occ, active_occ, active_virt, frozen_virt

def get_subspaces_from_cas(mf, ncas: int, nelecas: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Partition MO indices into active and virtual spaces based on a CAS definition.
    
    Parameters
    ----------
    mf : PySCF SCF object
        Converged mean-field object exposing ``mo_occ``.
    ncas : int
        Number of active orbitals in the CAS.
    nelecas : int
        Number of active electrons in the CAS.
    
    Returns
    -------
    active_orbs : np.ndarray of int
        Orbital indices (0-based) of the active space.
    virt_orbs : np.ndarray of int
        Orbital indices (0-based) of the virtual space (not in the active space).
    """
    print(f"\n----------DMRG CAS Subspace Selection (ncas={ncas}, nelecas={nelecas})----------")
    nmos = mf.mol.nao
    neletot = int(np.sum(mf.mo_occ))
    assert ncas <= nmos, "Number of active orbitals cannot exceed total number of MOs."
    assert nelecas >= 0, "Number of active electrons cannot be negative."
    assert nelecas <= neletot, "Number of active electrons cannot exceed total number of electrons."
    
    nocc = np.count_nonzero(mf.mo_occ > 0)
    print(f"\n\nTotal MOs: {nmos}, Occupied MOs: {nocc}, Virtual MOs: {nmos - nocc}")
    
    n_active_occ = nelecas//2  # Number of active occupied orbitals
    n_active_virt = ncas - n_active_occ  # Number of active virtual orbitals
    assert nmos - nocc - n_active_virt >= 0, "Not enough virtual orbitals to accommodate the active space."
    print(f"Active occupied orbitals: {n_active_occ}")
    print(f"Active virtual orbitals: {n_active_virt}")
    
    active_occ = np.arange(nocc-n_active_occ, nocc)
    active_virt = np.arange(nocc, nocc + n_active_virt)
    print(f"Active occupied orbital indices: {active_occ}")
    print(f"Active virtual orbital indices: {active_virt}\n\n")
    
    return active_occ, active_virt