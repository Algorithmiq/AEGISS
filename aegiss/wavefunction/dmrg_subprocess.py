"""DMRG wavefunction retrieval via block2main subprocess."""

from __future__ import annotations

import logging
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from pyscf import ao2mo, mcscf, tools

from aegiss.utils.dmrg_utils import _default_dmrg_schedule
from aegiss.utils.env import mem, omp_num_threads
from aegiss.utils.env import scratch_dir as get_scratch_dir
from aegiss.utils.orbital_indices import resolve_nelecas

logger = logging.getLogger(__name__)

_VALID_SYMM_TYPES = ("SU2", "SZ")


def _write_fcidump(
    path: Path,
    mf: Any,
    ncas: int,
    nelecas: int | tuple[int, int],
) -> None:
    """Write a PySCF FCIDUMP for the CAS active space to *path*."""
    mycas = mcscf.CASCI(mf, ncas, nelecas)
    h1e, ecore = mycas.get_h1eff()
    g2e = ao2mo.restore(1, mycas.get_h2eff(), ncas)
    nalpha, nbeta = resolve_nelecas(nelecas)
    tools.fcidump.from_integrals(
        str(path), h1e, g2e, ncas, nelecas, ecore, orbsym=[1] * ncas, ms=nalpha - nbeta
    )


def _write_input(
    path: Path,
    fcidump_path: Path,
    n_elec: int,
    spin: int,
    scratch_dir: Path,
    bond_dims: list[int],
    noises: list[float],
    thrds: list[float],
    n_threads: int,
    stack_mem_gb: float,
    symm_type: str = "SZ",
) -> None:
    """Write a StackBlock-compatible block2main input file to *path*."""
    n_sweeps = len(bond_dims)
    twosite_to_onesite = max(n_sweeps - 2, 0)

    sched_rows: list[str] = []
    prev = None
    for i, (m, thr, noise) in enumerate(zip(bond_dims, thrds, noises, strict=True)):
        if prev is None or (m, thr, noise) != prev:
            sched_rows.append(f"{i}  {m}  {thr}  {noise}")
            prev = (m, thr, noise)

    mem_gb = int(stack_mem_gb) if stack_mem_gb == int(stack_mem_gb) else stack_mem_gb

    lines: list[str] = [
        f"orbitals {fcidump_path}",
        f"nelec {n_elec}",
        f"spin {spin}",
        "hf_occ integral",
    ]
    if symm_type == "SZ":
        lines.append("nonspinadapted")
    lines += [
        "sym c1",
        "noreorder",
        f"prefix {scratch_dir}/",
        f"num_thrds {n_threads}",
        f"mem {mem_gb}G",
        "",
        "schedule",
        *sched_rows,
        "end",
        f"twodot_to_onedot {twosite_to_onesite}",
        f"maxiter {n_sweeps + 2}",
        f"sweep_tol {thrds[-1]}",
        "",
        "onepdm",
        "correlation",
    ]
    path.write_text("\n".join(lines))


def _read_outputs(scratch_dir: Path, ncas: int) -> tuple[float, np.ndarray, np.ndarray]:
    """Read block2main output files and return ``(energy, pdm1, npc1)``."""
    node0 = scratch_dir / "node0"
    missing = [
        f for f in ("E_dmrg.npy", "1pdm.npy", "1npc.npy") if not (node0 / f).exists()
    ]
    if missing:
        raise FileNotFoundError(
            f"block2main output missing in {node0}: {missing}. "
            "Check block2main stderr for errors."
        )
    energy = float(np.load(node0 / "E_dmrg.npy"))
    pdm1 = np.load(node0 / "1pdm.npy")  # (2, ncas, ncas): [0]=alpha, [1]=beta
    npc1 = np.load(node0 / "1npc.npy")  # (2, ncas, ncas)
    if pdm1.shape != (2, ncas, ncas) or npc1.shape != (2, ncas, ncas):
        raise ValueError(
            f"Unexpected block2main output shapes: pdm1={pdm1.shape}, npc1={npc1.shape}; expected (2, {ncas}, {ncas})."
        )
    return energy, pdm1, npc1


def _compute_soe(pdm1: np.ndarray, npc1: np.ndarray) -> np.ndarray:
    """Compute single-orbital von Neumann entropy from 1-PDM and 1-NPC.

    ``npc1[0,i,i] = n_i + 2*d_i``  →  ``d_i = (npc1[0,i,i] - n_i) / 2``.
    """
    ncas = pdm1.shape[-1]
    soe = np.zeros(ncas)
    for i in range(ncas):
        n_i = pdm1[0, i, i] + pdm1[1, i, i]
        d_i = (npc1[0, i, i] - n_i) / 2.0
        w = np.array([d_i, pdm1[0, i, i] - d_i, pdm1[1, i, i] - d_i, 1.0 - n_i + d_i])
        w = w[w > 1e-14]
        soe[i] = -np.dot(w, np.log(w))
    return soe


def run_dmrg_subprocess(
    mf: Any,
    ncas: int,
    nelecas: int | tuple[int, int],
    max_bond_dim: int = 500,
    symm_type: str = "SZ",
    verbosity: int = 0,
    *,
    get_qmi: bool = True,
) -> tuple[float, np.ndarray | None]:
    """Run a DMRG calculation via the ``block2main`` subprocess.

    Args:
        mf (pyscf.scf.hf.SCF): Converged mean-field object.
        ncas (int): Number of active orbitals.
        nelecas (int or tuple[int, int]): Number of active electrons (total
            count or ``(nalpha, nbeta)`` pair).
        max_bond_dim (int): Target bond dimension for the DMRG schedule (must
            be a positive multiple of 4).
        symm_type ({'SU2', 'SZ'}): Spin symmetry: ``'SZ'`` adds the
            ``nonspinadapted`` keyword so block2main runs without spin
            adaptation; ``'SU2'`` omits it.
        verbosity (int): If non-zero, block2main stdout/stderr are forwarded to
            the terminal.
        get_qmi (bool): If ``True``, compute and return single-orbital
            entropies.

    Returns:
        energy (float): DMRG ground-state energy.
        soe1 (np.ndarray or None): Single-orbital entropies (``None`` if
            *get_qmi* is ``False``). Two-orbital entropies and mutual
            information are not available from the ``1pdm`` + ``1npc`` output;
            pass them from an external source to any downstream plotting
            functions if needed.
    """
    if symm_type not in _VALID_SYMM_TYPES:
        raise ValueError(
            f"Unsupported symmetry type: {symm_type!r}. Expected one of {list(_VALID_SYMM_TYPES)}."
        )

    scratch = get_scratch_dir()
    n_threads = omp_num_threads()
    stack_mem_gb = mem() / (1024**3)

    bond_dims, noises, thrds = _default_dmrg_schedule(max_bond_dim)
    nalpha, nbeta = resolve_nelecas(nelecas)
    n_sweeps = len(bond_dims)
    twosite_to_onesite = max(n_sweeps - 2, 0)

    logger.info(
        "----------------DMRG Calculation (symmetry: %s)----------------\n", symm_type
    )
    logger.info("Using pyscf.CASCI to obtain integrals for DMRG...")
    logger.info("CAS (ncas=%d, nelecas=%s)", ncas, nelecas)
    logger.info(
        "System parameters: n_sites=%d, n_elec=%d, spin=%d, symm_type=%s",
        ncas,
        nalpha + nbeta,
        nalpha - nbeta,
        symm_type,
    )
    logger.info("Scratch directory: %s", scratch)
    logger.info("Number of threads: %d", n_threads)
    logger.info(
        "DMRG settings: n_sweeps = %d, max_bond_dim = %d, twosite_to_onesite = %d",
        n_sweeps,
        max_bond_dim,
        twosite_to_onesite,
    )
    logger.info("DMRG Noise: init = %s, final = %s", noises[0], noises[-1])
    logger.info("DMRG Threshold: init = %s, final = %s\n", thrds[0], thrds[-1])

    work_dir = Path(scratch) / f"dmrg_subprocess_{time.time_ns()}"
    work_dir.mkdir(parents=True, exist_ok=False)
    fcidump_path = work_dir / "FCIDUMP"
    _write_fcidump(fcidump_path, mf, ncas, nelecas)

    input_path = work_dir / "dmrg.conf"
    _write_input(
        input_path,
        fcidump_path,
        nalpha + nbeta,
        nalpha - nbeta,
        work_dir,
        bond_dims,
        noises,
        thrds,
        n_threads,
        stack_mem_gb,
        symm_type,
    )

    _venv_bin = Path(sys.executable).parent
    block2main_path = shutil.which("block2main") or (
        str(_venv_bin / "block2main") if (_venv_bin / "block2main").exists() else None
    )
    if block2main_path is None:
        raise RuntimeError(
            "block2main not found in PATH or venv bin directory. "
            "Activate the project venv or add block2main to PATH."
        )

    logger.info("Running block2main: %s %s", block2main_path, input_path)
    t = time.time()
    subprocess.run(  # noqa: S603
        [block2main_path, str(input_path)],
        capture_output=(verbosity == 0),
        check=True,
    )
    logger.info("DMRG completed in %.3f [s]", time.time() - t)

    energy, pdm1, npc1 = _read_outputs(work_dir, ncas)
    logger.info("DMRG Energy (%s)\t : %.8f", symm_type, energy)

    if not get_qmi:
        return energy, None

    soe1 = _compute_soe(pdm1, npc1)
    return energy, soe1
