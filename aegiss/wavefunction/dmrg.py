"""DMRG wavefunction retrieval via block2/PySCF DMRGSCF."""
from __future__ import annotations

from pyscf import mcscf
# from pyscf.dmrgscf import DMRGCI


def run_dmrg(mf, ncas: int, nelecas: int, maxM: int = 500, **kwargs):
    """Run a DMRG-CASSCF calculation.

    Parameters
    ----------
    mf : pyscf mean-field object
        Converged HF or DFT reference.
    ncas : int
        Number of active orbitals.
    nelecas : int
        Number of active electrons.
    maxM : int
        Maximum bond dimension.

    Returns
    -------
    mc : pyscf.mcscf.CASSCF
        Converged CASSCF object with DMRG solver.
    """
    mc = mcscf.CASSCF(mf, ncas, nelecas)
    # mc.fcisolver = DMRGCI(mf.mol, maxM=maxM)
    for key, val in kwargs.items():
        setattr(mc, key, val)
    mc.kernel()
    return mc
