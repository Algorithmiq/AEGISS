"""Natural orbital (NO) localization from a correlated wavefunction."""
from __future__ import annotations

import numpy as np


def compute_nos(wfn_obj, 
                *, 
                split_nos: bool = False):
    """Compute natural orbitals from a correlated wavefunction object.

    Extracts the 1-RDM from *wfn_obj* (CISD, CCSD, CASCI/CASSCF, …),
    diagonalises it, and returns the NOs sorted by descending occupation.

    Parameters
    ----------
    wfn_obj : correlated PySCF solver object
        Must expose a ``make_rdm1`` method returning the 1-RDM in the MO basis.
    split_nos : bool, optional
        Obtain the Natural Orbital basis by diagonalizing the 1-RDM in the occupied and virtual subspaces separately. 
        This can be useful for certain post-HF methods where the occupied and virtual spaces are treated differently.

    Returns
    -------
    occ : np.ndarray, shape (nmo,)
        Natural occupation numbers.
    nos : np.ndarray, shape (nao, nmo)
        Natural orbital coefficients in the AO basis.
    """
    rdm1_mo = wfn_obj.make_rdm1()
    print("1-RDM in MO basis:\n", rdm1_mo.shape)
    if split_nos:
        print("Building Split NOs by diagonalizing occupied and virtual blocks separately.")
        nocc = wfn_obj.nocc
        rdm1_occ = rdm1_mo[:nocc, :nocc]
        rdm1_virt = rdm1_mo[nocc:, nocc:]

        occ_occ, u_occ = np.linalg.eigh(rdm1_occ)
        occ_virt, u_virt = np.linalg.eigh(rdm1_virt)

        # Sort occupied and virtual NOs separately
        idx_occ = np.argsort(occ_occ)[::-1]
        idx_virt = np.argsort(occ_virt)[::-1]

        occ_occ = occ_occ[idx_occ]
        occ_virt = occ_virt[idx_virt]

        u_occ = u_occ[:, idx_occ]
        u_virt = u_virt[:, idx_virt]

        # Combine occupied and virtual NOs back into a single MO coefficient matrix
        nos_occ = wfn_obj.mo_coeff[:, :nocc] @ u_occ
        nos_virt = wfn_obj.mo_coeff[:, nocc:] @ u_virt
        nos = np.hstack((nos_occ, nos_virt))
        occ_mo = np.concatenate((occ_occ, occ_virt))
    else:
        print("Building standard NOs by diagonalizing the full 1-RDM.")
        occ_mo, u = np.linalg.eigh(rdm1_mo)
        # sort descending
        idx = np.argsort(occ_mo)[::-1]
        occ_mo = occ_mo[idx]
        u = u[:, idx]
        mo_coeff = wfn_obj._scf.mo_coeff if hasattr(wfn_obj, "_scf") else wfn_obj.mo_coeff
        nos = mo_coeff @ u
    return nos
