"""AEGISS pipeline: higher-order function composition."""
from __future__ import annotations

from typing import Callable
import os
import numpy as np
from pyscf import gto, mp, ci

from aegiss.utils.banner import print_banner
from aegiss.initial_guess import run_scf, build_mol
from aegiss.wavefunction.dmrg import run_dmrg
from aegiss.active_space.entropy_screening import select_by_entropy
from aegiss.active_space.ao_projection import select_by_ao_projection
from aegiss.utils.orbital_indices import get_subspaces_from_cas

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["SCRATCH_DIR"] = "./aegiss/examples/tmp"

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
    
    print_banner()
    atom = './aegiss/examples/data/water.xyz'
    mol = build_mol(atom,basis='sto3g', verbose=4)
    mf = run_scf(mol)
    dmrg_ncas, dmrg_nelecas = 7, 10
    
    active_orbs, virt_orbs = get_subspaces_from_cas(mf, ncas=dmrg_ncas, nelecas=dmrg_nelecas)
    _, soe1, _ , minfo = run_dmrg(  mf, 
                                    ncas=dmrg_ncas, 
                                    nelecas=dmrg_nelecas, 
                                    symm_type='SZ',
                                    init_bond_dim=dmrg_nelecas,
                                    verbosity=0, 
                                    get_qmi=True)
    
    screened_occ, screened_virt = select_by_entropy(occ_orbs=active_orbs, 
                                 virt_orbs=virt_orbs, 
                                 soe1=soe1, 
                                 ranking='relative',
                                 fraction=0,
                                 fraction_type='r')
    
    final_space = select_by_ao_projection(mol, mf,
                            ao_labels=["O.1 2p", "H.0 1s", "H.2 1s"],
                            contribution=0.4,
                            projected_basis='minao',
                            contribution_type='a',
                            restricted_space=(screened_occ,screened_virt))
    
    
    
    