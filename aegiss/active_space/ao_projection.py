"""Active space selection via AO projection onto a target set of atoms/AOs."""
from __future__ import annotations

import numpy as np


def select_by_ao_projection(
    mol,
    mo_coeff: np.ndarray,
    target_atoms: list[int],
    threshold: float = 0.2,
) -> np.ndarray:
    """Select active orbitals by their projection onto a set of target AOs.

    An MO is included in the active space when its Mulliken-style projection
    onto the AOs centred on *target_atoms* exceeds *threshold*.

    Parameters
    ----------
    mol : pyscf.gto.Mole
    mo_coeff : np.ndarray, shape (nao, nmo)
        MO coefficient matrix.
    target_atoms : list of int
        0-based atom indices defining the projection space.
    threshold : float
        Inclusion threshold for the fractional projection weight.

    Returns
    -------
    active_indices : np.ndarray of int
        Indices of MOs selected for the active space.
    """
    from pyscf import gto

    ovlp = mol.intor("int1e_ovlp")
    # collect AO indices belonging to target atoms
    ao_slices = mol.aoslice_by_atom()
    target_ao_idx = []
    for atom in target_atoms:
        p0, p1 = ao_slices[atom, 2], ao_slices[atom, 3]
        target_ao_idx.extend(range(p0, p1))
    target_ao_idx = np.array(target_ao_idx)

    # fractional weight of each MO on target AOs: PS matrix diagonal slice
    PS = ovlp @ mo_coeff  # (nao, nmo)
    weights = np.einsum("pi,pi->i", mo_coeff[target_ao_idx], PS[target_ao_idx])
    active_indices = np.where(weights >= threshold)[0]
    return active_indices
