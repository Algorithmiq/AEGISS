"""Natural orbital (NO) localization from a correlated wavefunction."""
from __future__ import annotations

import numpy as np


def compute_nos(wfn_obj):
    """Compute natural orbitals from a correlated wavefunction object.

    Extracts the 1-RDM from *wfn_obj* (CISD, CCSD, CASCI/CASSCF, …),
    diagonalises it, and returns the NOs sorted by descending occupation.

    Parameters
    ----------
    wfn_obj : correlated PySCF solver object
        Must expose a ``make_rdm1`` method returning the 1-RDM in the MO basis.

    Returns
    -------
    occ : np.ndarray, shape (nmo,)
        Natural occupation numbers.
    nos : np.ndarray, shape (nao, nmo)
        Natural orbital coefficients in the AO basis.
    """
    rdm1_mo = wfn_obj.make_rdm1()
    occ_mo, u = np.linalg.eigh(rdm1_mo)
    # sort descending
    idx = np.argsort(occ_mo)[::-1]
    occ_mo = occ_mo[idx]
    u = u[:, idx]
    mo_coeff = wfn_obj._scf.mo_coeff if hasattr(wfn_obj, "_scf") else wfn_obj.mo_coeff
    nos = mo_coeff @ u
    return occ_mo, nos
