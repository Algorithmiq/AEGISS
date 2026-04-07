"""AEGISS pipeline: higher-order function composition."""
from __future__ import annotations

from typing import Callable
import numpy as np
from pyscf import gto, mp, ci


def run_aegiss(
    mol: gto.Mole,
    initial_guess_fn: Callable,
    wavefunction_fn: Callable,
    localization_fn: Callable,
    active_space_fn: Callable,
) -> np.ndarray:
    """Run the full AEGISS pipeline.

    Each stage is supplied as a callable, making it straightforward to swap
    individual steps without changing the orchestration logic.

    Pipeline
    --------
    1. **Initial guess**    mol  → mf
    2. **Wavefunction**     mf   → wfn
    3. **Localization**     wfn  → (occ, loc_coeff)
    4. **Active space**     (mol, occ, loc_coeff) → active_indices

    Parameters
    ----------
    mol : pyscf.gto.Mole
    initial_guess_fn : Callable[[gto.Mole], mf]
        e.g. ``partial(run_hf)`` or ``partial(run_dft, xc="pbe0")``
    wavefunction_fn : Callable[[mf], wfn]
        e.g. ``partial(run_dmrg, ncas=10, nelecas=10, maxM=500)``
    localization_fn : Callable[[wfn], tuple[np.ndarray, np.ndarray]]
        Must return ``(occ, mo_coeff)`` in the AO basis.
        e.g. ``partial(compute_nos)`` wrapped to return (occ, coeff)
    active_space_fn : Callable[[gto.Mole, np.ndarray, np.ndarray], np.ndarray]
        e.g. ``partial(select_aegiss, target_atoms=[0, 1])``

    Returns
    -------
    active_indices : np.ndarray of int
        Orbital indices selected for the active space.

    Example
    -------
    >>> from functools import partial
    >>> from pyscf import gto
    >>> from aegiss.initial_guess import run_hf
    >>> from aegiss.wavefunction import run_dmrg
    >>> from aegiss.localization import compute_nos
    >>> from aegiss.active_space import select_aegiss
    >>>
    >>> def localize(wfn):
    ...     occ, nos = compute_nos(wfn)
    ...     return occ, nos
    >>>
    >>> mol = gto.M(atom="N 0 0 0; N 0 0 1.1", basis="ccpvdz", spin=0)
    >>> active_idx = run_aegiss(
    ...     mol,
    ...     initial_guess_fn=run_hf,
    ...     wavefunction_fn=partial(run_dmrg, ncas=6, nelecas=6, maxM=200),
    ...     localization_fn=localize,
    ...     active_space_fn=partial(select_aegiss, target_atoms=[0, 1]),
    ... )
    """
    mf = initial_guess_fn(mol)
    wfn = wavefunction_fn(mf)
    occ, loc_coeff = localization_fn(wfn)
    active_indices = active_space_fn(mol, loc_coeff, occ)
    return active_indices


if __name__ == "__main__":
    # Example usage
    from aegiss.initial_guess import run_scf, build_mol
    from aegiss.localization.orbital_localization import localize
    from aegiss.wavefunction.dmrg import run_dmrg
    from aegiss.utils.dump_intermediates import  dump_qmi
    #water
    atom =  " O 0 0 0; H 0 0 0.95; H 0.95 0 0"
    mol = build_mol(atom)
    mf = run_scf(mol)
    
    _, soe1, _ , minfo = run_dmrg(mf, 
                                            ncas=7, 
                                            nelecas=mol.tot_electrons(), 
                                            symm_type='SZ',
                                            init_bond_dim=10, 
                                            get_qmi=True)
    
    dump_qmi(minfo=soe1, soe1=minfo, path="./qmi_water.npz")
    # mf.mo_coeff = localize(method='fnos', mf= mf, correlated_method='cisd',active=[2,3,4])
    