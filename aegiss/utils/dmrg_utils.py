"""DMRG utilities: block2 input preparation (FCIDUMP, conf, sweep schedule)."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np
from pyscf import ao2mo, mcscf, tools
from pyscf.tools.fcidump import TOL, _convert_orbsym, write_eri, write_hcore

from aegiss.utils.orbital_indices import resolve_nelecas

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_FLOAT_FORMAT = "%24.16f"

_SCHEDULE_WAYPOINTS = [0, 4, 8, 12, 16]
_SCHEDULE_MAX_SWEEPS = 30
_SCHEDULE_NOISES = [1e-4, 1e-4, 5e-5, 5e-5, 0.0]
_SCHEDULE_THRDS = [1e-8, 1e-10, 1e-12, 1e-12, 1e-12]


def _default_dmrg_schedule(
    max_bond_dim: int = 500,
) -> tuple[list[int], list[float], list[float]]:
    """Define a default DMRG sweep schedule for the Block2 ``DMRGDriver.dmrg`` method.

    Builds a 30-sweep schedule from fixed waypoints at sweeps ``[0, 4, 8, 12, 16]``.
    The first segment uses ``max_bond_dim // 2``; all subsequent segments use
    ``max_bond_dim``.  Noise and thresholds are tightened at each waypoint.

    Parameters
    ----------
    max_bond_dim : int
        Target bond dimension.  Must be a positive multiple of 4
        so that both ``max_bond_dim`` and ``max_bond_dim // 2`` are even.

    Returns:
    -------
    bond_dims : list of int
        Bond dimension for each sweep.
    noises : list of float
        Perturbative noise injected at each sweep.
    thrds : list of float
        Davidson convergence threshold for each sweep.

    Note:
        Suggested in: https://github.com/block-hczhai/block2-preview/issues/63
    """
    if max_bond_dim <= 0:
        raise ValueError("max_bond_dim must be a positive multiple of 4")
    if max_bond_dim % 4 != 0:
        raise ValueError(
            "max_bond_dim must be a multiple of 4 to keep all bond dimensions even"
        )

    bd_at_waypoint = [max_bond_dim // 2] + [max_bond_dim] * (
        len(_SCHEDULE_WAYPOINTS) - 1
    )

    bond_dims: list[int] = []
    noises: list[float] = []
    thrds: list[float] = []
    for i, start in enumerate(_SCHEDULE_WAYPOINTS):
        end = (
            _SCHEDULE_WAYPOINTS[i + 1]
            if i + 1 < len(_SCHEDULE_WAYPOINTS)
            else _SCHEDULE_MAX_SWEEPS
        )
        n = end - start
        bond_dims.extend([bd_at_waypoint[i]] * n)
        noises.extend([_SCHEDULE_NOISES[i]] * n)
        thrds.extend([_SCHEDULE_THRDS[i]] * n)

    return bond_dims, noises, thrds


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


def _write_header(orb_sym: list[int], ncas: int, na: int, nb: int) -> str:
    """Helper function to write the FCIDUMP header for UHF integrals."""
    header = f" &FCI NORB= {ncas:3d},NELEC= {na + nb:4d},MS2= {na - nb:4d},\n"
    header += f"  ORBSYM={','.join(str(i) for i in orb_sym)},\n"
    header += "  ISYM=   1,\n"
    header += "  IUHF=1,\n"
    header += " &END\n"
    return header


def _write_uhf_fcidump(
    path: Path,
    mf: Any,
    ncas: int,
    nelecas: tuple[int, int],
    orb_sym: list[int] | None = None,
    tol: float | Any = TOL,
) -> None:
    """Write a PySCF FCIDUMP for the Unrestricted CAS active space to *path*."""
    na, nb = nelecas
    mycas = mcscf.UCASCI(mf, ncas, nelecas)
    h1e, ecore = mycas.get_h1eff()
    g2e = mycas.get_h2eff()

    if not isinstance(h1e, tuple) or len(h1e) != 2:
        raise TypeError("expected UHF h1e as a 2-tuple (alpha, beta)")
    if not isinstance(g2e, tuple) or len(g2e) != 3:
        raise TypeError("expected UHF g2e as a 3-tuple (aa, ab, bb)")

    mh1e_a = h1e[0]
    mh1e_b = h1e[1]
    mh1e_a[np.abs(mh1e_a) < tol] = 0.0
    mh1e_b[np.abs(mh1e_b) < tol] = 0.0

    g2e_aa = ao2mo.restore(8, g2e[0], ncas)
    g2e_ab = ao2mo.restore(4, g2e[1], ncas)
    g2e_bb = ao2mo.restore(8, g2e[2], ncas)

    g2e_aa[np.abs(g2e_aa) < tol] = 0.0
    g2e_ab[np.abs(g2e_ab) < tol] = 0.0
    g2e_bb[np.abs(g2e_bb) < tol] = 0.0

    orb_sym = (
        _convert_orbsym(mf.mol, orb_sym, molpro_orbsym=True)
        if orb_sym is not None
        else [1] * ncas
    )

    header = _write_header(orb_sym, ncas, na, nb)
    with path.open("w") as f:
        f.write(header)
        output_format = DEFAULT_FLOAT_FORMAT + "  0  0  0  0\n"
        # block2's UHF reader uses a "0.0  0 0 0 0" sentinel line to advance from
        # one spin sector to the next, so each block MUST be followed by one.
        # Omitting them makes block2 dump everything into the aa sector (segfault /
        # zero bb/ab/h1e) even though the per-value text diff looks identical.
        write_eri(f, g2e_aa, ncas, tol=tol, float_format=DEFAULT_FLOAT_FORMAT)
        f.write(output_format % 0.0)
        write_eri(f, g2e_bb, ncas, tol=tol, float_format=DEFAULT_FLOAT_FORMAT)
        f.write(output_format % 0.0)
        write_eri(f, g2e_ab, ncas, tol=tol, float_format=DEFAULT_FLOAT_FORMAT)
        f.write(output_format % 0.0)
        write_hcore(f, mh1e_a, ncas, tol=tol, float_format=DEFAULT_FLOAT_FORMAT)
        f.write(output_format % 0.0)
        write_hcore(f, mh1e_b, ncas, tol=tol, float_format=DEFAULT_FLOAT_FORMAT)
        f.write(output_format % 0.0)
        f.write(output_format % ecore)


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
