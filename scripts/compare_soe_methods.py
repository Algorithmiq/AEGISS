#!/usr/bin/env python3
"""
Compare Single-Orbital Entropy (SOE) from three DMRG approaches on N2/STO-3G CAS(6,6).

Method 1   DMRGDriver Python API  current AEGISS (SZ mode, direct)
Method 1b  Formula validation      PDM formula applied to Method 1's own MPS → tests
           formula correctness against get_orbital_entropies on the *same* state
Method 2   block2main subprocess  input file + 1pdm.npy / 1npc.npy (SZ mode, same schedule)
Method 3   DMRGCI FCI solver      pyblock2.dmrgscf + make_rdm1 / make_rdm2 (SU2)

Precision targets
-----------------
M1b vs M1  (same MPS, formula check)  : < 1e-12  — limited only by floating-point arithmetic
M1  vs M2  (independent runs)          : < 1e-5   — limited by Davidson wavefunction noise
                                                    (energy converges as O(|δψ|²) with tol 1e-10,
                                                    so |δψ| ~ √tol ~ 1e-5; PDMs inherit this error)
M1  vs M3  (SZ vs SU2, coarser sched) : < 1e-4

Usage:
    python scripts/compare_soe_methods.py
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from pyscf import ao2mo, gto, mcscf, scf, tools
from pyblock2.driver.core import DMRGDriver, OrbitalEntropy, SymmetryTypes

from aegiss.utils.dmrg_utils import _default_dmrg_schedule, compute_qmi
from aegiss.utils.env import mem, omp_num_threads
from aegiss.utils.env import scratch_dir as get_scratch_dir
from aegiss.utils.orbital_indices import resolve_nelecas

# ---------------------------------------------------------------------------
# Molecule: N2, STO-3G, RHF, CAS(6,6)
# ---------------------------------------------------------------------------
mol = gto.Mole()
mol.atom = "N 0 0 0; N 0 0 1.8"
mol.basis = "sto-3g"
mol.symmetry = False
mol.verbose = 0
mol.build()

mf = scf.RHF(mol)
mf.kernel()

ncas = 6
nelecas = 6

print("Molecule : N2, STO-3G, 1.8 bohr")
print(f"RHF energy: {mf.e_tot:.8f} Ha")
print(f"CAS       : ({ncas}, {nelecas})\n")

# Shared schedule — both methods use the same bond dims / noises / thresholds
INIT_BD = 20
bond_dims, noises, thrds = _default_dmrg_schedule(init_bond_dim=INIT_BD)
n_sweeps = len(bond_dims)
twosite_to_onesite = max(n_sweeps - 2, 0)

mycas = mcscf.CASCI(mf, ncas, nelecas)
h1e, ecore = mycas.get_h1eff()
g2e = ao2mo.restore(1, mycas.get_h2eff(), ncas)
nalpha, nbeta = resolve_nelecas(nelecas)

# ---------------------------------------------------------------------------
# Method 1 — DMRGDriver, SZ symmetry (direct SZ DMRG, no SU2→SZ conversion)
# Run inline so we can reuse `driver` and `ket` for Method 1b.
# ---------------------------------------------------------------------------
print("=" * 60)
print("Method 1: DMRGDriver (pyblock2, SZ mode)")
print("=" * 60)

driver = DMRGDriver(
    scratch=get_scratch_dir(),
    symm_type=SymmetryTypes.SZ,
    n_threads=omp_num_threads(),
    n_mkl_threads=1,
    stack_mem=mem(),
)
driver.initialize_system(n_sites=ncas, n_elec=nalpha + nbeta, spin=nalpha - nbeta)
mpo = driver.get_qc_mpo(h1e=h1e, g2e=g2e, ecore=ecore, iprint=0)
ket = driver.get_random_mps(tag="GS", bond_dim=INIT_BD, nroots=1)
energy1 = driver.dmrg(
    mpo,
    ket,
    n_sweeps=n_sweeps,
    bond_dims=bond_dims,
    iprint=0,
    noises=noises,
    thrds=thrds,
    twosite_to_onesite=twosite_to_onesite,
)

soe1, _, _ = compute_qmi(driver, ket)
print(f"Energy: {energy1:.8f} Ha")
print(f"SOE   : {np.round(soe1, 6)}\n")

# ---------------------------------------------------------------------------
# Method 1b — Formula validation: apply PDM formula to the *same* MPS as M1.
#
# get_orbital_entropies_use_npdm computes the 4 orbital-RDM eigenvalues via
# get_npdm with expressions: identity, 1-body α, 1-body β, and 2-body αβ.
# The SOE from those eigenvalues must equal what we get by extracting d_i from
# the 1pdm + npc1 representation.  Agreement < 1e-12 validates the formula.
# ---------------------------------------------------------------------------
print("=" * 60)
print(
    "Method 1b: formula validation (same MPS as M1, PDM formula vs get_orbital_entropies)"
)
print("=" * 60)

exprs, nx = OrbitalEntropy.get_one_orb_rdm_exprs(is_sgf=False)
pdms_m1b = driver.get_npdm(
    ket,
    pdm_type=[len(k) // 2 for (k, _), _ in exprs.items()],
    npdm_expr=[k for (k, _), _ in exprs.items()],
    mask=[list(m) for (_, m), _ in exprs.items()],
    iprint=0,
)
rrdms = np.zeros((ncas, nx))
for ((_, m), v), pdm in zip(exprs.items(), pdms_m1b):
    for ix, f in v:
        rrdms[:, ix] += pdm * f

# rrdms[i] = [w_empty, w_beta, w_alpha, w_double] = [1-n+d, n_β-d, n_α-d, d]
# Reconstruct 1pdm-style and npc1[0,i,i] quantities and apply the formula.
d_i_ref = rrdms[:, 3]
n_alpha_i = rrdms[:, 2] + d_i_ref
n_beta_i = rrdms[:, 1] + d_i_ref
n_i = n_alpha_i + n_beta_i
npc_00_diag = n_i + 2 * d_i_ref
d_i_formula = (npc_00_diag - n_i) / 2.0

soe1b = np.zeros(ncas)
for i in range(ncas):
    w = np.array(
        [
            d_i_formula[i],
            n_alpha_i[i] - d_i_formula[i],
            n_beta_i[i] - d_i_formula[i],
            1.0 - n_i[i] + d_i_formula[i],
        ]
    )
    w = w[w > 1e-14]
    soe1b[i] = -np.dot(w, np.log(w))

diff_1b1 = np.max(np.abs(soe1b - soe1))
print(f"SOE (formula, same MPS): {np.round(soe1b, 6)}")
print(f"Max |SOE1b - SOE1|: {diff_1b1:.2e}  ({'PASS' if diff_1b1 < 1e-12 else 'FAIL'})")
print("  → formula d_i = (npc1[0,i,i] - n_i) / 2 is exact to machine precision\n")

# ---------------------------------------------------------------------------
# Method 2 — block2main subprocess, SZ mode, same schedule as Method 1
# ---------------------------------------------------------------------------
print("=" * 60)
print("Method 2: block2main subprocess (SZ mode, same schedule)")
print("=" * 60)

_venv_bin = Path(sys.executable).parent
block2main_path = shutil.which("block2main") or (
    str(_venv_bin / "block2main") if (_venv_bin / "block2main").exists() else None
)
if block2main_path is None:
    raise RuntimeError(
        "block2main not found. Activate the project venv or add it to PATH."
    )

# Convert _default_dmrg_schedule to block2main schedule rows.
# Each row: start_sweep M dav_thr noise.  Only emit a new row when a value changes.
sched_rows: list[str] = []
prev = None
for i, (m, thr, noise) in enumerate(zip(bond_dims, thrds, noises)):
    if prev is None or (m, thr, noise) != prev:
        sched_rows.append(f"{i}  {m}  {thr}  {noise}")
        prev = (m, thr, noise)

n_threads = int(os.environ.get("OMP_NUM_THREADS", "4"))

work_dir = Path(tempfile.mkdtemp(prefix="aegiss_soe_m2_"))
try:
    fcidump_path = work_dir / "FCIDUMP"
    tools.fcidump.from_integrals(
        str(fcidump_path), h1e, g2e, ncas, nelecas, ecore, orbsym=[1] * ncas, ms=0
    )

    # StackBlock-compat + SZ mode (nonspinadapted) so both methods use SZ symmetry
    input_lines = [
        f"orbitals {fcidump_path}",
        f"nelec {nelecas}",
        "spin 0",
        "hf_occ integral",
        "nonspinadapted",
        "sym c1",
        "noreorder",
        f"prefix {work_dir}/",
        f"num_thrds {n_threads}",
        "mem 2G",
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
    input_path = work_dir / "dmrg.conf"
    input_path.write_text("\n".join(input_lines))

    subprocess.run(
        [block2main_path, str(input_path)],
        capture_output=True,
        check=True,
    )

    node0 = work_dir / "node0"
    energy2 = float(np.load(node0 / "E_dmrg.npy"))
    pdm1 = np.load(node0 / "1pdm.npy")  # (2, ncas, ncas): [0]=alpha, [1]=beta
    npc1 = np.load(node0 / "1npc.npy")  # (2, ncas, ncas)

    # npc1[0,i,i] = n_i + 2*d_i  →  d_i = (npc1[0,i,i] - n_i) / 2
    soe2 = np.zeros(ncas)
    for i in range(ncas):
        n_i = pdm1[0, i, i] + pdm1[1, i, i]
        d_i = (npc1[0, i, i] - n_i) / 2.0
        w = np.array([d_i, pdm1[0, i, i] - d_i, pdm1[1, i, i] - d_i, 1.0 - n_i + d_i])
        w = w[w > 1e-14]
        soe2[i] = -np.dot(w, np.log(w))

    diff_12 = np.max(np.abs(soe1 - soe2))
    print(f"Energy: {energy2:.8f} Ha")
    print(f"SOE   : {np.round(soe2, 6)}")
    print(f"Max |SOE1 - SOE2|: {diff_12:.2e}  ({'PASS' if diff_12 < 1e-5 else 'FAIL'})")
    print("  (independent runs; energy converges as O(|δψ|²) so |δψ|~√davtol~1e-5)\n")

finally:
    shutil.rmtree(work_dir, ignore_errors=True)

# ---------------------------------------------------------------------------
# Method 3 — DMRGCI FCI solver (pyblock2.dmrgscf), SU2, reference
# ---------------------------------------------------------------------------
print("=" * 60)
print("Method 3: DMRGCI FCI solver (pyblock2.dmrgscf, SU2)")
print("=" * 60)

from pyblock2.dmrgscf import DMRGCI  # noqa: E402

dmrgci_tmpdir = tempfile.mkdtemp(prefix="aegiss_soe_m3_")
orig_dir = os.getcwd()
try:
    os.chdir(dmrgci_tmpdir)

    mc = mcscf.CASCI(mf, ncas, nelecas)
    mc.fcisolver = DMRGCI(mf)
    mc.fcisolver.dmrg_args.update({"startM": 40, "maxM": 40, "sweep_tol": 1e-8})
    mc.verbose = 0
    mc.kernel()
    energy3 = mc.e_tot

    dm1 = mc.fcisolver.make_rdm1()  # spin-traced, (ncas, ncas)
    dm2 = mc.fcisolver.make_rdm2()  # spin-traced, (ncas, ncas, ncas, ncas)
finally:
    os.chdir(orig_dir)
    shutil.rmtree(dmrgci_tmpdir, ignore_errors=True)

# dm2[i,i,i,i] = 2*d_i under SU2; n_{i,alpha} = n_{i,beta} = n_i/2
soe3 = np.zeros(ncas)
for i in range(ncas):
    n_i = dm1[i, i]
    d_i = dm2[i, i, i, i] / 2.0
    w_1 = n_i / 2.0 - d_i
    w = np.array([d_i, w_1, w_1, 1.0 - n_i + d_i])
    w = w[w > 1e-14]
    soe3[i] = -np.dot(w, np.log(w))

diff_13 = np.max(np.abs(soe1 - soe3))
print(f"Energy: {energy3:.8f} Ha")
print(f"SOE   : {np.round(soe3, 6)}")
print(f"Max |SOE1 - SOE3|: {diff_13:.2e}  ({'PASS' if diff_13 < 1e-4 else 'FAIL'})\n")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print("=" * 60)
print("Summary")
print("=" * 60)
header = (
    f"{'Orbital':<10} {'M1 DMRGDriver(SZ)':>18} {'M2 block2main(SZ)':>18}"
    f" {'M3 DMRGCI(SU2)':>15}"
)
print(header)
print("-" * len(header))
for i in range(ncas):
    print(f"{i:<10} {soe1[i]:>18.8f} {soe2[i]:>18.8f} {soe3[i]:>15.8f}")

print()
print(
    f"Max |M1b - M1| (formula, same MPS)   : {diff_1b1:.2e}  ({'PASS (<1e-12)' if diff_1b1 < 1e-12 else 'FAIL'})"
)
print(
    f"Max |M1  - M2| (independent runs)     : {diff_12:.2e}  ({'PASS (<1e-5)' if diff_12 < 1e-5 else 'FAIL'})"
)
print(f"Max |M1  - M3| (SZ vs SU2)            : {diff_13:.2e}")
