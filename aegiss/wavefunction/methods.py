"""Other wavefunction methods (FCI, CISD, CCSD) via PySCF."""
from __future__ import annotations

from typing import Literal, Any
from pyscf import ci, cc, fci as _fci, mcscf


def run_fci(mf, ncas: int, nelecas: int, **kwargs):
    """Run FCI in an active space.

    Returns
    -------
    mc : pyscf.mcscf.CASCI
    """
    mc = mcscf.CASCI(mf, ncas, nelecas)
    for key, val in kwargs.items():
        setattr(mc, key, val)
    mc.kernel()
    return mc


def run_cisd(mf, **kwargs):
    """Run CISD on top of a mean-field reference.

    Returns
    -------
    myci : pyscf.ci.CISD
    """
    myci = ci.CISD(mf)
    for key, val in kwargs.items():
        setattr(myci, key, val)
    myci.kernel()
    return myci

def run_mp2(mf, **kwargs):
    """Run MP2 on top of a mean-field reference.

    Returns
    -------
    mymp2 : pyscf.mp.MP2
    """
    mymp2 = mf.MP2()
    for key, val in kwargs.items():
        setattr(mymp2, key, val)
    mymp2.kernel()
    return mymp2

def run_ccsd(mf, **kwargs):
    """Run CCSD on top of a mean-field reference.

    Returns
    -------
    mycc : pyscf.cc.CCSD
    """
    mycc = cc.CCSD(mf)
    for key, val in kwargs.items():
        setattr(mycc, key, val)
    mycc.kernel()
    return mycc

def build_correlated_wfn(mf, method: Literal['mp2', 'cisd', 'ccsd', 'casci', 'casscf'], ncas: int = 6, nelecas: int = 6) -> Any:
    """Helper to build a correlated wavefunction object for NO computation."""
    if method == 'mp2':
        return run_mp2(mf)
    elif method == 'cisd':
        return run_cisd(mf)
    elif method == 'ccsd':
        return run_ccsd(mf)
    elif method in ['casci', 'casscf']:
        return run_fci(mf, ncas=6, nelecas=6)  # Example active space
    else:
        raise ValueError(f"Unsupported method for NOs: {method}")