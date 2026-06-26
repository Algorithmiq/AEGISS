"""End-to-end DMRG regression tests (require a working ``block2main``).

These guard the local-minimum trap that the bare ``hf_occ integral`` (random)
MPS init falls into on near-degenerate / multireference active spaces. The
aufbau ``hf_occ`` warmup + ``full_fci_space`` now written by ``_write_input``
must recover the exact CASCI ground state on cases that previously got stuck
~0.04-0.14 Ha too high.

Skipped automatically when pyscf/pyblock2/block2main are unavailable. Marked
``slow`` (each case runs a real UHF + stability search + DMRG).
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pytest

pytest.importorskip("pyscf")
pytest.importorskip("pyblock2")

from pyscf import gto, mcscf, scf

from aegiss.wavefunction.dmrg_subprocess import run_dmrg_subprocess


def _has_block2main() -> bool:
    if shutil.which("block2main") is not None:
        return True
    return (Path(sys.executable).parent / "block2main").exists()


pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(not _has_block2main(), reason="block2main not available"),
]


def _broken_symmetry_uhf(atom: str, spin: int, charge: int = 0):
    """Converge UHF, then follow the internal instability to the broken-symmetry
    (multireference) solution."""
    mol = gto.M(
        atom=atom, basis="cc-pvdz", spin=spin, charge=charge, symmetry=False, verbose=0
    )
    mf = scf.UHF(mol).run()
    mo = mf.stability()[0]
    return mf.run(mf.make_rdm1(mo, mf.mo_occ))


@pytest.mark.parametrize("reorder", ["noreorder", "fiedler"])
@pytest.mark.parametrize(
    ("label", "atom", "spin", "charge", "ncas", "nelecas"),
    [
        # closed-shell singlet: stuck at -108.637 (+0.139 Ha) with the old init
        ("N2_singlet_BS", "N 0 0 0; N 0 0 2.1", 0, 0, 6, (3, 3)),
        # open-shell doublet: stuck at -108.247 (+0.036 Ha) with the old init
        ("N2cat_doublet_BS", "N 0 0 0; N 0 0 2.1", 1, 1, 6, (3, 2)),
    ],
)
def test_uhf_dmrg_recovers_exact_casci_on_trap(
    label, atom, spin, charge, ncas, nelecas, reorder, monkeypatch, tmp_path
) -> None:
    monkeypatch.setenv("SCRATCH_DIR", str(tmp_path))
    monkeypatch.setenv("OMP_NUM_THREADS", "1")  # deterministic

    mf = _broken_symmetry_uhf(atom, spin, charge)
    e_exact = mcscf.UCASCI(mf, ncas, nelecas).kernel()[0]

    energy, _ = run_dmrg_subprocess(
        mf,
        ncas=ncas,
        nelecas=nelecas,
        max_bond_dim=100,
        symm_type="SZ",
        unrestricted=True,
        get_qmi=False,
        reorder=reorder,
    )
    # 6-orbital CAS at bond dim 100 is FCI-exact; the only way to miss the exact
    # energy is the init trap this test guards against. The occ warmup must
    # recover it under both orbital orderings.
    assert energy == pytest.approx(e_exact, abs=1e-5), (
        f"{label}/{reorder}: DMRG {energy:.8f} != exact CASCI {e_exact:.8f} "
        f"(diff {energy - e_exact:+.2e}) — init trap regression"
    )
