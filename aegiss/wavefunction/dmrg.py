"""DMRG wavefunction retrieval via block2/PySCF DMRGSCF."""

from __future__ import annotations

import logging
import os
import time
from typing import TYPE_CHECKING, Any

from pyblock2.driver.core import DMRGDriver, SymmetryTypes
from pyscf import ao2mo, mcscf

from aegiss.utils.dmrg_utils import (
    _casci_integrals,
    _default_dmrg_schedule,
    _refinement_schedule,
    compute_qmi,
)
from aegiss.utils.orbital_indices import resolve_nelecas

if TYPE_CHECKING:
    import numpy as np

logger = logging.getLogger(__name__)


def run_dmrg(
    mf: Any,
    ncas: int,
    nelecas: int | tuple[int, int],
    init_bond_dim: int = 10,
    symm_type: str = "SU2",
    verbosity: int = 0,
    *,
    get_qmi: bool = True,
) -> tuple[float, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """Run a DMRG calculation using block2 via a PySCF CASCI integral driver.

    Parameters
    ----------
    mf : pyscf.scf.hf.SCF
        Converged mean-field object.
    ncas : int
        Number of active orbitals.
    nelecas : int or tuple[int, int]
        Number of active electrons, either as a total count or an explicit
        ``(nalpha, nbeta)`` pair (same convention as PySCF's CASCI).
    init_bond_dim : int
        Initial bond dimension for the DMRG schedule.
    symm_type : {'SU2', 'SZ'}
        Spin symmetry type for block2.
    verbosity : int
        Verbosity level for block2 output.
    get_qmi : bool
        If ``True``, compute and return quantum mutual information.

    Returns:
    -------
    energy : float
        DMRG ground-state energy.
    soe1 : np.ndarray or None
        Single-orbital entropies (``None`` if *get_qmi* is ``False``).
    soe2 : np.ndarray or None
        Two-orbital entropies (``None`` if *get_qmi* is ``False``).
    minfo : np.ndarray or None
        Quantum mutual information matrix (``None`` if *get_qmi* is ``False``).
    """
    n_threads = int(os.environ.get("OMP_NUM_THREADS", "4"))
    scratch_dir = os.environ.get("SCRATCH_DIR", "./tmp")
    _symm_map = {"SU2": SymmetryTypes.SU2, "SZ": SymmetryTypes.SZ}
    if symm_type not in _symm_map:
        raise ValueError(
            f"Unsupported symmetry type: {symm_type!r}. Expected one of {list(_symm_map)}."
        )
    symm = _symm_map[symm_type]

    logger.info(
        "----------------DMRG Calculation (symmetry: %s)----------------\n", symm_type
    )
    driver = DMRGDriver(
        scratch=scratch_dir, symm_type=symm, n_threads=n_threads, n_mkl_threads=1
    )

    mycas = mcscf.CASCI(mf, ncas, nelecas)
    logger.info("Using pyscf.CASCI to obtain integrals for DMRG...")
    logger.info("CAS (ncas=%d, nelecas=%s)", ncas, nelecas)
    h1e, ecore = mycas.get_h1eff()
    g2e = ao2mo.restore(1, mycas.get_h2eff(), ncas)

    nalpha, nbeta = resolve_nelecas(nelecas)
    n_elec = nalpha + nbeta
    spin = nalpha - nbeta
    driver.initialize_system(n_sites=ncas, n_elec=n_elec, spin=spin)

    logger.info(
        "System parameters: n_sites=%d, n_elec=%d, spin=%d, symm_type=%s",
        driver.n_sites,
        driver.n_elec,
        spin,
        symm_type,
    )
    logger.info("Scratch directory: %s", scratch_dir)
    logger.info("Number of threads: %d", n_threads)

    bond_dims, noises, thrds = _default_dmrg_schedule(init_bond_dim=init_bond_dim)

    n_sweeps = len(bond_dims)
    twosite_to_onesite = max(n_sweeps - 2, 0)
    logger.info(
        "DMRG settings: n_sweeps = %d, init_bond_dim = %d, final_bond_dim = %d,"
        " twosite_to_onesite = %d",
        n_sweeps,
        init_bond_dim,
        bond_dims[-1],
        twosite_to_onesite,
    )
    logger.info("DMRG Noise: init = %s, final = %s", noises[0], noises[-1])
    logger.info("DMRG Threshold: init = %s, final = %s\n", thrds[0], thrds[-1])
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
        n_sweeps=n_sweeps,
        bond_dims=bond_dims,
        iprint=verbosity,
        noises=noises,
        thrds=thrds,
        twosite_to_onesite=twosite_to_onesite,
    )
    logger.info("DMRG completed in %.3f [s]", time.time() - t)
    logger.info("DMRG Energy (%s)\t : %.8f", symm_type, energy)

    soe1, soe2, minfo = None, None, None
    if symm_type == "SU2":
        logger.info("Mapping SU(2) MPS and MPO to SZ")

        driver.align_mps_center(ket, ref=0)
        zket = driver.mps_change_to_sz(ket, "ZKET")

        driver.symm_type = SymmetryTypes.SZ
        driver.initialize_system(n_sites=ncas, n_elec=n_elec, spin=spin)
        mpo = driver.get_qc_mpo(h1e=h1e, g2e=g2e, ecore=ecore, iprint=0)
        impo = driver.get_identity_mpo()
        norm = driver.expectation(zket, impo, zket)
        zket.iscale(1 / norm)
        logger.info("SZ Energy: %.8f", driver.expectation(zket, mpo, zket))
        if get_qmi:
            soe1, soe2, minfo = compute_qmi(driver, zket)

    elif get_qmi:
        soe1, soe2, minfo = compute_qmi(driver, ket)

    return energy, soe1, soe2, minfo


def run_sa_dmrg(
    mf: Any,
    ncas: int,
    nelecas: int | tuple[int, int],
    nroots: int,
    weights: tuple[float, ...] | None = None,
    init_bond_dim: int = 10,
    symm_type: str = "SZ",
    n_refinement_sweeps: int = 4,
    verbosity: int = 0,
    *,
    get_qmi: bool = True,
) -> tuple[
    list[float],
    list[np.ndarray | None],
    list[np.ndarray | None],
    list[np.ndarray | None],
]:
    """Run a state-averaged DMRG followed by state-specific refinement sweeps.

    The workflow is:

    1. **SA-DMRG**: all *nroots* states are optimized jointly using the supplied
       (or uniform) *weights* set on the multi-root MPS before sweeping.
    2. **SS refinement**: each root is extracted from the SA-MPS via
       ``driver.split_mps``, then *n_refinement_sweeps* single-state sweeps are
       run independently.  This step is skipped when *n_refinement_sweeps* is ``0``.
    3. **Entropy extraction**: single-orbital entropies are computed for each
       single-root MPS.

    Parameters
    ----------
    mf : pyscf.scf.hf.SCF
        Converged mean-field object.
    ncas : int
        Number of active orbitals.
    nelecas : int or tuple[int, int]
        Number of active electrons (same convention as PySCF's CASCI).
    nroots : int
        Number of electronic states to target (must be >= 2).
    weights : tuple of float or None
        Per-state SA weights.  ``None`` uses uniform ``1 / nroots`` weights.
    init_bond_dim : int
        Initial bond dimension for the SA sweep schedule.
    symm_type : {'SZ', 'SU2'}
        Spin symmetry type for block2.  ``'SZ'`` is strongly recommended for
        SA-DMRG; ``'SU2'`` triggers a per-root SU2 → SZ conversion before
        entropy extraction.
    n_refinement_sweeps : int
        Number of state-specific sweeps after SA convergence.  ``0`` skips this
        step and extracts entropies directly from the split SA-MPS roots.
    verbosity : int
        Verbosity level for block2 output.
    get_qmi : bool
        If ``True``, compute and return per-state quantum mutual information.

    Returns:
    -------
    energies : list of float
        Per-state energies (SS if refinement was run, SA otherwise).
    soe1_per_state : list of np.ndarray or None
        Per-state single-orbital entropies (``None`` if *get_qmi* is ``False``).
    soe2_per_state : list of np.ndarray or None
        Per-state two-orbital entropies (``None`` if *get_qmi* is ``False``).
    minfo_per_state : list of np.ndarray or None
        Per-state quantum mutual information matrices (``None`` if *get_qmi* is
        ``False``).
    """
    n_threads = int(os.environ.get("OMP_NUM_THREADS", "4"))
    scratch_dir = os.environ.get("SCRATCH_DIR", "./tmp")
    _symm_map = {"SU2": SymmetryTypes.SU2, "SZ": SymmetryTypes.SZ}
    if symm_type not in _symm_map:
        raise ValueError(
            f"Unsupported symmetry type: {symm_type!r}. Expected one of {list(_symm_map)}."
        )
    symm = _symm_map[symm_type]

    if nroots < 2:
        raise ValueError(f"nroots must be >= 2 for state-averaged DMRG, got {nroots}.")

    if weights is None:
        sa_weights = [1.0 / nroots] * nroots
    else:
        sa_weights = list(weights)
        if len(sa_weights) != nroots:
            raise ValueError(
                f"weights must have length {nroots}, got {len(sa_weights)}."
            )
        if any(w <= 0 for w in sa_weights):
            raise ValueError("weights must contain only positive values.")

    logger.info(
        "----------------SA-DMRG Calculation (symmetry: %s, nroots: %d)----------------\n",
        symm_type,
        nroots,
    )
    driver = DMRGDriver(
        scratch=scratch_dir, symm_type=symm, n_threads=n_threads, n_mkl_threads=1
    )

    ecore, h1e, g2e = _casci_integrals(mf, ncas, nelecas)

    nalpha, nbeta = resolve_nelecas(nelecas)
    n_elec = nalpha + nbeta
    spin = nalpha - nbeta
    driver.initialize_system(n_sites=ncas, n_elec=n_elec, spin=spin)

    logger.info(
        "System parameters: n_sites=%d, n_elec=%d, spin=%d, symm_type=%s",
        driver.n_sites,
        driver.n_elec,
        spin,
        symm_type,
    )
    logger.info("Scratch directory: %s", scratch_dir)
    logger.info("Number of threads: %d", n_threads)
    logger.info("SA weights: %s", sa_weights)

    bond_dims, noises, thrds = _default_dmrg_schedule(init_bond_dim=init_bond_dim)
    n_sweeps = len(bond_dims)
    twosite_to_onesite = max(n_sweeps - 2, 0)

    logger.info(
        "SA-DMRG settings: n_sweeps=%d, init_bond_dim=%d, final_bond_dim=%d,"
        " twosite_to_onesite=%d",
        n_sweeps,
        init_bond_dim,
        bond_dims[-1],
        twosite_to_onesite,
    )

    mpo = driver.get_qc_mpo(h1e=h1e, g2e=g2e, ecore=ecore, iprint=verbosity)
    ket = driver.get_random_mps(tag="SA_KET", bond_dim=init_bond_dim, nroots=nroots)
    # SA weights are set as an attribute on the MultiMPS object before sweeping.
    ket.weights = driver.bw.VectorFP(sa_weights)

    t = time.time()
    sa_result = driver.dmrg(
        mpo,
        ket,
        n_sweeps=n_sweeps,
        bond_dims=bond_dims,
        noises=noises,
        thrds=thrds,
        twosite_to_onesite=twosite_to_onesite,
        iprint=verbosity,
    )
    logger.info("SA-DMRG completed in %.3f [s]", time.time() - t)
    sa_energies: list[float] = (
        [float(e) for e in sa_result]
        if isinstance(sa_result, (list, tuple))
        else [float(sa_result)]
    )
    for k, e in enumerate(sa_energies):
        logger.info("SA-DMRG Energy (root %d, %s): %.8f", k, symm_type, e)

    # --- State-specific refinement: split SA-MPS, sweep each root independently ---
    ref_bond_dims, ref_noises, ref_thrds = (
        _refinement_schedule(n_refinement_sweeps, bond_dims[-1])
        if n_refinement_sweeps > 0
        else ([], [], [])
    )

    ss_energies: list[float] = []
    ket_roots: list[Any] = []
    ket_roots = [driver.split_mps(ket, k, f"SS_KET_{k}") for k in range(nroots)]
    for k in range(nroots):
        if n_refinement_sweeps > 0:
            logger.info("SS refinement: root %d / %d", k, nroots - 1)
            t = time.time()
            ss_result = driver.dmrg(
                mpo,
                ket_roots[k],
                n_sweeps=n_refinement_sweeps,
                bond_dims=ref_bond_dims,
                noises=ref_noises,
                thrds=ref_thrds,
                iprint=verbosity,
                proj_weights=[5.0] * k,
                proj_mpss=ket_roots[:k],
            )
            logger.info("  Root %d SS completed in %.3f [s]", k, time.time() - t)
            ss_energies.append(float(ss_result))

    energies = ss_energies if n_refinement_sweeps > 0 else sa_energies
    for k, e in enumerate(energies):
        logger.info("Final Energy (root %d, %s): %.8f", k, symm_type, e)

    # --- Entropy extraction ---
    soe1_per_state: list[np.ndarray | None] = [None] * nroots
    soe2_per_state: list[np.ndarray | None] = [None] * nroots
    minfo_per_state: list[np.ndarray | None] = [None] * nroots

    if not get_qmi:
        return energies, soe1_per_state, soe2_per_state, minfo_per_state

    if symm_type == "SU2":
        logger.info("Mapping SU(2) MPS roots to SZ for entropy extraction")
        driver.symm_type = SymmetryTypes.SZ
        driver.initialize_system(n_sites=ncas, n_elec=n_elec, spin=spin)
        mpo_sz = driver.get_qc_mpo(h1e=h1e, g2e=g2e, ecore=ecore, iprint=0)
        impo = driver.get_identity_mpo()
        for k, ket_k in enumerate(ket_roots):
            driver.align_mps_center(ket_k, ref=0)
            zket_k = driver.mps_change_to_sz(ket_k, f"ZKET_{k}")
            norm = driver.expectation(zket_k, impo, zket_k)
            zket_k.iscale(1 / norm)
            logger.info(
                "SZ Energy (root %d): %.8f",
                k,
                driver.expectation(zket_k, mpo_sz, zket_k),
            )
            soe1, soe2, minfo = compute_qmi(driver, zket_k)
            soe1_per_state[k] = soe1
            soe2_per_state[k] = soe2
            minfo_per_state[k] = minfo
    else:
        for k, ket_k in enumerate(ket_roots):
            soe1, soe2, minfo = compute_qmi(driver, ket_k)
            soe1_per_state[k] = soe1
            soe2_per_state[k] = soe2
            minfo_per_state[k] = minfo

    return energies, soe1_per_state, soe2_per_state, minfo_per_state
