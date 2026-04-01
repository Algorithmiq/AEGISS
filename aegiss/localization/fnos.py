"""Frozen natural orbital (FNO) construction."""
from __future__ import annotations

import numpy as np
from aegiss.localization.nos import compute_nos


def compute_fnos(wfn_obj, occ_threshold: float = 1e-4):
    """Construct frozen natural orbitals by truncating weakly occupied NOs.

    Parameters
    ----------
    wfn_obj : correlated PySCF solver object
    occ_threshold : float
        Natural orbitals with occupation below this value are frozen.

    Returns
    -------
    occ_active : np.ndarray
        Occupation numbers of retained NOs.
    fnos : np.ndarray, shape (nao, n_active)
        FNO coefficients in the AO basis.
    frozen_mask : np.ndarray of bool
        True for NOs that were frozen (occupation < threshold).
    """
    occ, nos = compute_nos(wfn_obj)
    frozen_mask = occ < occ_threshold
    active_mask = ~frozen_mask
    return occ[active_mask], nos[:, active_mask], frozen_mask
