"""Other wavefunction methods (FCI, CISD, CCSD) via PySCF."""

from __future__ import annotations

import logging
from typing import Any, Literal

from pyscf import cc, ci, mp

logger = logging.getLogger(__name__)


def run_cisd(mf: Any, **kwargs: Any) -> Any:
    """Run CISD on top of a mean-field reference.

    Args:
        mf (pyscf.scf.hf.SCF): Converged mean-field object.
        **kwargs: Additional attributes forwarded to the PySCF CISD object.

    Returns:
        myci (pyscf.ci.CISD): Converged CISD object."""
    logger.info("Running CISD with options: %s", kwargs)
    myci = ci.CISD(mf)
    for key, val in kwargs.items():
        setattr(myci, key, val)
    myci.kernel()
    return myci


def run_mp2(mf: Any, **kwargs: Any) -> Any:
    """Run MP2 on top of a mean-field reference.

    Args:
        mf (pyscf.scf.hf.SCF): Converged mean-field object.
        **kwargs: Additional attributes forwarded to the PySCF MP2 object.

    Returns:
        mymp2 (pyscf.mp.MP2): Converged MP2 object."""
    logger.info("Running MP2 with options: %s", kwargs)
    mymp2 = mp.MP2(mf)
    for key, val in kwargs.items():
        setattr(mymp2, key, val)
    mymp2.kernel()
    return mymp2


def run_ccsd(mf: Any, **kwargs: Any) -> Any:
    """Run CCSD on top of a mean-field reference.

    Args:
        mf (pyscf.scf.hf.SCF): Converged mean-field object.
        **kwargs: Additional attributes forwarded to the PySCF CCSD object.

    Returns:
        mycc (pyscf.cc.CCSD): Converged CCSD object."""
    logger.info("Running CCSD with options: %s", kwargs)
    mycc = cc.CCSD(mf)
    for key, val in kwargs.items():
        setattr(mycc, key, val)
    mycc.kernel()
    return mycc


def build_correlated_wfn(
    mf: Any,
    method: Literal["mp2", "cisd", "ccsd"],
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Build a correlated wavefunction object for NO computation.

    Args:
        mf (pyscf.scf.hf.SCF): Converged mean-field object.
        method ({'mp2', 'cisd', 'ccsd'}): Correlated method to use.
        *args: Positional arguments forwarded to the method runner.
        **kwargs: Keyword arguments forwarded to the method runner.

    Returns:
        wfn_obj (pyscf correlated solver object): Converged correlated wavefunction object."""
    if method == "mp2":
        return run_mp2(mf, **kwargs)
    if method == "cisd":
        return run_cisd(mf, **kwargs)
    if method == "ccsd":
        return run_ccsd(mf, **kwargs)
    raise ValueError(f"Unsupported method for NOs: {method}")
