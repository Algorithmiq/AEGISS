"""Full AEGISS active space selection procedure."""
from __future__ import annotations

import numpy as np
from aegiss.active_space.entropy_screening import select_by_entropy
from aegiss.active_space.ao_projection import select_by_ao_projection


def select_aegiss(
    mol,
    mo_coeff: np.ndarray,
    occ: np.ndarray,
    target_atoms: list[int],
    entropy_threshold: float = 0.1,
    projection_threshold: float = 0.2,
) -> np.ndarray:
    """AEGISS active space selection: entropy screening + AO projection.

    Combines single-orbital entropy screening and AO projection to obtain
    a physically motivated active space.  An orbital is included when it
    satisfies *either* criterion.

    Parameters
    ----------
    mol : pyscf.gto.Mole
    mo_coeff : np.ndarray, shape (nao, nmo)
    occ : np.ndarray, shape (nmo,)
        Natural occupation numbers.
    target_atoms : list of int
        Atoms defining the AO projection space.
    entropy_threshold : float
    projection_threshold : float

    Returns
    -------
    active_indices : np.ndarray of int
        Sorted union of indices selected by both criteria.
    """
    idx_entropy = set(select_by_entropy(occ, threshold=entropy_threshold).tolist())
    idx_proj = set(
        select_by_ao_projection(
            mol, mo_coeff, target_atoms, threshold=projection_threshold
        ).tolist()
    )
    active_indices = np.array(sorted(idx_entropy | idx_proj), dtype=int)
    return active_indices
