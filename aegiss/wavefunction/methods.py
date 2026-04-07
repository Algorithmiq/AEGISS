"""Other wavefunction methods (FCI, CISD, CCSD) via PySCF."""
from __future__ import annotations

from typing import Literal, Any
from pyscf import ci, cc, mp

def run_cisd(mf, **kwargs):
    """Run CISD on top of a mean-field reference.

    Returns
    -------
    myci : pyscf.ci.CISD
    """
    print("Running CISD with options:", kwargs)
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
    print("Running MP2 with options:", kwargs)
    mymp2 = mp.MP2(mf)
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
    print("Running CCSD with options:", kwargs)
    mycc = cc.CCSD(mf)
    for key, val in kwargs.items():
        setattr(mycc, key, val)
    mycc.kernel()
    return mycc

def build_correlated_wfn(mf, method: Literal['mp2', 'cisd', 'ccsd'], *args, **kwargs) -> Any:
    """Helper to build a correlated wavefunction object for NO computation."""
    if method == 'mp2':
        return run_mp2(mf, **kwargs)
    elif method == 'cisd':
        return run_cisd(mf, **kwargs)
    elif method == 'ccsd':
        return run_ccsd(mf, **kwargs)
    else:
        raise ValueError(f"Unsupported method for NOs: {method}")