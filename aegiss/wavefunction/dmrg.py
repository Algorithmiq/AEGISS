"""DMRG wavefunction retrieval via block2/PySCF DMRGSCF."""

from __future__ import annotations

import logging
import os
import time
from typing import TYPE_CHECKING, Any

from pyblock2.driver.core import DMRGDriver, SymmetryTypes
from pyscf import ao2mo, mcscf

from aegiss.utils.dmrg_utils import _default_dmrg_schedule, compute_qmi
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

    Notes:
    -----
    Three environment variables are read at call time:

    * ``OMP_NUM_THREADS`` (default ``4``) — number of OpenMP threads for block2.
    * ``SCRATCH_DIR`` (default ``"./tmp"``) — directory for DMRG scratch files.
    * ``MEM`` (default ``450``) — memory budget in **GiB** passed to block2 as
      ``stack_mem``.  The value is converted via ``MEM * 2**30`` bytes, so
      ``MEM=16`` allocates a 16 GiB stack.  Set this to roughly the amount of
      free RAM available to the process; block2 will raise an error at startup
      if the requested stack cannot be allocated.
    """
    n_threads = int(os.environ.get("OMP_NUM_THREADS", "4"))
    stack_mem = (int(os.environ.get("MEM", "450"))) * (1 << 30)

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
        scratch=scratch_dir,
        symm_type=symm,
        n_threads=n_threads,
        n_mkl_threads=1,
        stack_mem=stack_mem,
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
