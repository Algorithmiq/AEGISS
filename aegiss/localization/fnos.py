"""Frozen natural orbital (FNO) construction."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def compute_fnos(
    wfn_obj: Any,
    frozen: list[int],
    active: list[int],
) -> np.ndarray:
    """Construct frozen natural orbitals by truncating weakly occupied NOs.

    Args:
        wfn_obj (correlated PySCF solver object):
            Must expose a ``make_rdm1`` method returning the 1-RDM in the MO basis.
        frozen (list of int):
            Indices of orbitals to be frozen (0-based indexing).
        active (list of int):
            Indices of orbitals to be retained in the active space (0-based indexing).

    Returns:
        mo_coeff (np.ndarray, shape (nao, nmo)):
            FNO coefficients in the AO basis, including both frozen and active orbitals.
    """
    rdm1 = wfn_obj.make_rdm1()

    nocc = len(frozen) + len(active)
    logger.debug("1-RDM in MO basis:\n%s", rdm1.shape)
    logger.info(
        "Diagonalizing virtual block of the 1-RDM to construct FNOs for virtual orbitals."
    )
    dm_vv = rdm1[nocc:, nocc:]
    occ_vv, u_vv = np.linalg.eigh(dm_vv)
    idx_vv = np.argsort(occ_vv)[::-1]
    occ_vv = occ_vv[idx_vv]
    u_vv = u_vv[:, idx_vv]

    mo_coeff_vv = wfn_obj.mo_coeff[:, nocc:] @ u_vv
    mo_coeff_oo = wfn_obj.mo_coeff[:, :nocc]
    return np.hstack((mo_coeff_oo, mo_coeff_vv))
