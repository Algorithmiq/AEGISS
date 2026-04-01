"""Active space selection via single-orbital entropy screening."""
from __future__ import annotations

import numpy as np


def select_by_entropy(occ: np.ndarray, threshold: float = 0.1) -> np.ndarray:
    """Select active orbitals by single-orbital entanglement entropy.

    The single-orbital entropy is defined as:
        s_i = -n_i * ln(n_i) - (1 - n_i) * ln(1 - n_i)
    where n_i is the natural occupation number (scaled to [0, 1]).

    Parameters
    ----------
    occ : np.ndarray, shape (nmo,)
        Natural occupation numbers (expected range [0, 2]).
    threshold : float
        Orbitals with s_i >= threshold are included in the active space.

    Returns
    -------
    active_indices : np.ndarray of int
        Indices of orbitals selected for the active space.
    """
    n = occ / 2.0  # scale to [0, 1]
    n = np.clip(n, 1e-15, 1 - 1e-15)
    entropy = -n * np.log(n) - (1 - n) * np.log(1 - n)
    active_indices = np.where(entropy >= threshold)[0]
    return active_indices
