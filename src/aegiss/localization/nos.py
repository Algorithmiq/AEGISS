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

"""Natural orbital (NO) localization from a correlated wavefunction."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def compute_nos(wfn_obj: Any, *, split_nos: bool = False) -> np.ndarray:
    """Compute natural orbitals from a correlated wavefunction object.

    Extracts the 1-RDM from *wfn_obj* (CISD, CCSD, CASCI/CASSCF, …),
    diagonalises it, and returns the NOs sorted by descending occupation.

    Args:
        wfn_obj (correlated PySCF solver object):
            Must expose a ``make_rdm1`` method returning the 1-RDM in the MO basis.
        split_nos (bool, optional):
            Obtain the Natural Orbital basis by diagonalizing the 1-RDM in the
            occupied and virtual subspaces separately. Useful for certain post-HF
            methods where the occupied and virtual spaces are treated differently.

    Returns:
        nos (np.ndarray, shape (nao, nmo)):
            Natural orbital coefficients in the AO basis.
    """
    rdm1_mo = wfn_obj.make_rdm1()
    logger.debug("1-RDM in MO basis:\n%s", rdm1_mo.shape)
    if split_nos:
        logger.info(
            "Building Split NOs by diagonalizing occupied and virtual blocks separately."
        )
        nocc = wfn_obj.nocc
        rdm1_occ = rdm1_mo[:nocc, :nocc]
        rdm1_virt = rdm1_mo[nocc:, nocc:]

        occ_occ, u_occ = np.linalg.eigh(rdm1_occ)
        occ_virt, u_virt = np.linalg.eigh(rdm1_virt)

        # Sort occupied and virtual NOs separately
        idx_occ = np.argsort(occ_occ)[::-1]
        idx_virt = np.argsort(occ_virt)[::-1]

        u_occ = u_occ[:, idx_occ]
        u_virt = u_virt[:, idx_virt]

        # Combine occupied and virtual NOs back into a single MO coefficient matrix
        nos_occ = wfn_obj.mo_coeff[:, :nocc] @ u_occ
        nos_virt = wfn_obj.mo_coeff[:, nocc:] @ u_virt
        return np.hstack((nos_occ, nos_virt))
    logger.info("Building standard NOs by diagonalizing the full 1-RDM.")
    occ_mo, u = np.linalg.eigh(rdm1_mo)
    # sort descending
    idx = np.argsort(occ_mo)[::-1]
    u = u[:, idx]
    # pyscf post-HF objects expose their SCF only as `_scf`
    scf = getattr(wfn_obj, "_scf", wfn_obj)
    mo_coeff = scf.mo_coeff
    return np.asarray(mo_coeff @ u)
