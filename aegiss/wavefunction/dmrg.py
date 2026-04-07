"""DMRG wavefunction retrieval via block2/PySCF DMRGSCF."""
from __future__ import annotations

import os
import time
import numpy as np
from pyscf import mcscf, ao2mo
from pyblock2.driver.core import DMRGDriver, SymmetryTypes
from aegiss.utils.dmrg_utils import _default_dmrg_schedule, compute_qmi


def run_dmrg(mf, 
             ncas: int, 
             nelecas: int, 
             init_bond_dim: int = 10, 
             symm_type: str = "SU2",
             verbosity: int = 0,
             *,
             get_qmi = False) -> tuple[float, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    
    n_threads = int(os.environ.get("OMP_NUM_THREADS", 4))
    scratch_dir = os.environ.get("SCRATCH_DIR", "./tmp")
    if symm_type == "SU2":
        symm = SymmetryTypes.SU2
    elif symm_type == "SZ":
        symm = SymmetryTypes.SZ
    else:
        raise ValueError(f"Unsupported symmetry type: {symm_type}")

    driver = DMRGDriver(scratch=scratch_dir,
                        symm_type=symm, 
                        n_threads=n_threads,
                        n_mkl_threads=1)
    
    mycas = mcscf.CASCI(mf, ncas, nelecas)
    print("Using pyscf.CASCI to obtain integrals for DMRG...")
    print(f"CAS (ncas={ncas}, nelecas={nelecas})")
    h1e, ecore = mycas.get_h1eff()
    g2e = ao2mo.restore(1, mycas.get_h2eff(), ncas)
    
    driver.initialize_system(
        n_sites=ncas,
        n_elec=nelecas,
        spin=0
    )

    print(
        f"System parameters: n_sites={driver.n_sites}, n_elec={driver.n_elec}, spin=0, symm_type={symm_type}"
    )
    print(f"Scratch directory: {scratch_dir}")
    print(f"Number of threads: {n_threads}")

    bond_dims, noises, thrds = _default_dmrg_schedule(init_bond_dim=init_bond_dim)

    n_sweeps = len(bond_dims)
    twosite_to_onesite = max(n_sweeps - 2, 0)
    print(
        f"DMRG settings: n_sweeps = {n_sweeps}, init_bond_dim = {init_bond_dim}, final_bond_dim = {bond_dims[-1]}, twosite_to_onesite = {twosite_to_onesite}"
    )
    print(f"DMRG Noise: init = {noises[0]}, final = {noises[-1]}")
    print(f"DMRG Threshold: init = {thrds[0]}, final = {thrds[-1]}\n")
    mpo = driver.get_qc_mpo(
        h1e=h1e,
        g2e=g2e,
        ecore=ecore,
        iprint=verbosity,
    )

    ket = driver.get_random_mps(tag="GS", bond_dim=init_bond_dim, nroots=1)

    t = time.time()
    energy = driver.dmrg(
        mpo,
        ket,
        n_sweeps=len(bond_dims),
        bond_dims=bond_dims,
        iprint=verbosity,
        noises=noises,
        thrds=thrds,
        twosite_to_onesite=twosite_to_onesite,
    )
    print(f"DMRG completed in {time.time() - t:.3f} [s]")
    print(f"DMRG Energy ({symm_type})\t : {energy:.8f}")

    soe1, soe2, minfo = None, None, None
    if symm_type == "SU2":
        print("Mapping SU(2) MPS and MPO to SZ")
        
        driver.align_mps_center(ket, ref=0)
        zket = driver.mps_change_to_sz(ket, "ZKET")

        driver.symm_type = SymmetryTypes.SZ
        driver.initialize_system(n_sites=ncas, n_elec=nelecas, spin=0)
        mpo = driver.get_qc_mpo(h1e=h1e, g2e=g2e, ecore=ecore, iprint=0)
        impo = driver.get_identity_mpo()
        norm = driver.expectation(zket, impo, zket)
        zket.iscale(1 / norm)
        print(f"SZ Energy: {driver.expectation(zket, mpo, zket):.8f}")
        if get_qmi:
            soe1, soe2, minfo = compute_qmi(driver, zket)
        
    elif get_qmi:
        soe1, soe2, minfo = compute_qmi(driver, ket)
    
    return energy, soe1, soe2, minfo

