# Copyright 2026 Algorithmiq
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
    """Define a default DMRG sweep schedule for the block2main input file.

    Builds a 30-sweep schedule from fixed waypoints at sweeps ``[0, 4, 8, 12, 16]``.
    The first segment uses ``max_bond_dim // 2``; all subsequent segments use
    ``max_bond_dim``.  Noise and thresholds are tightened at each waypoint.

    Args:
        max_bond_dim (int): Target bond dimension.  Must be a positive multiple of 4
            so that both ``max_bond_dim`` and ``max_bond_dim // 2`` are even.

    Returns:
        bond_dims (list of int): Bond dimension for each sweep.
        noises (list of float): Perturbative noise injected at each sweep.
        thrds (list of float): Davidson convergence threshold for each sweep.

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

    mh1e_a = h1e[0].copy()
    mh1e_b = h1e[1].copy()
    mh1e_a[np.abs(mh1e_a) < tol] = 0.0
    mh1e_b[np.abs(mh1e_b) < tol] = 0.0

    g2e_aa = ao2mo.restore(8, g2e[0], ncas).copy()
    g2e_ab = ao2mo.restore(4, g2e[1], ncas).copy()
    g2e_bb = ao2mo.restore(8, g2e[2], ncas).copy()

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


def _aufbau_hf_occ(n_elec: int, spin: int, ncas: int) -> str:
    """Spin-summed aufbau occupation string for block2main's ``occ`` warmup.

    Returns ``ncas`` space-separated integers (each 0/1/2): the lowest
    ``min(na, nb)`` orbitals doubly occupied, the next ``|na - nb|`` singly
    occupied, the rest empty (built by adding one per alpha and one per beta
    electron from the lowest orbital up).

    Fed to block2main's ``occ`` keyword (with ``warmup occ``), this seeds the
    initial MPS bond-dimension distribution at the reference determinant, which
    avoids a local-minimum trap that the bare random init falls into for
    near-degenerate / multireference active spaces. The ``occ`` line must have
    exactly ``ncas`` *spatial* values; the per-spin (2*ncas) form is not used by
    this path, so the spin-summed spatial form is required.
    """
    na = (n_elec + spin) // 2
    nb = (n_elec - spin) // 2
    return " ".join(
        str((1 if i < na else 0) + (1 if i < nb else 0)) for i in range(ncas)
    )


def _write_input(
    path: Path,
    fcidump_path: Path,
    n_elec: int,
    spin: int,
    ncas: int,
    scratch_dir: Path,
    bond_dims: list[int],
    noises: list[float],
    thrds: list[float],
    n_threads: int,
    stack_mem_gb: float,
    symm_type: str = "SZ",
    reorder: str = "noreorder",
) -> None:
    """Write a StackBlock-compatible block2main input file to *path*.

    ``reorder`` selects orbital ordering: ``'noreorder'`` (default) keeps the
    FCIDUMP order and pairs with ``full_fci_space``; ``'fiedler'`` lets
    block2main reorder by the Fiedler vector and omits ``full_fci_space``
    (that combination can land in a higher local minimum). The ``occ`` warmup is
    written in both cases, and block2main maps ``1pdm``/``1npc`` back to the
    input orbital order regardless, so downstream per-orbital indexing is
    unaffected by the choice.
    """
    no_reorder = reorder == "noreorder"
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
        # Keep hf_occ=integral so block2main stays in StackBlock-compatible mode
        # (it writes outputs under node0/, which _read_outputs reads). Drive the
        # occupation warmup explicitly instead: seed the initial MPS at the
        # reference determinant (warmup occ), which guards against a local-minimum
        # trap the bare random init hits on near-degenerate active spaces.
        "hf_occ integral",
        "warmup occ",
        f"occ {_aufbau_hf_occ(n_elec, spin, ncas)}",
        "cbias 0.2",
    ]
    if no_reorder:
        # full-FCI initial distribution; pairs with noreorder (omitted under
        # fiedler, where it can drive convergence to a higher local minimum).
        lines.append("full_fci_space")
    if symm_type == "SZ":
        lines.append("nonspinadapted")
    lines.append("sym c1")
    if no_reorder:
        lines.append("noreorder")
    # else: omit -> block2main defaults to fiedler reordering
    lines += [
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
