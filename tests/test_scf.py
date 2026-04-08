"""Tests for aegiss.initial_guess (build_mol, run_scf)."""
from __future__ import annotations

import pytest
from aegiss.initial_guess import build_mol, run_scf


# ---------------------------------------------------------------------------
# build_mol
# ---------------------------------------------------------------------------

class TestBuildMol:

    def test_basic_h2(self):
        mol = build_mol("H 0 0 0; H 0 0 0.74", basis="sto-3g", verbose=0)
        assert mol.nao == 2  # STO-3G: 1 AO per H

    def test_basis_kwarg_applied(self):
        mol = build_mol("H 0 0 0; H 0 0 0.74", basis="cc-pvdz", verbose=0)
        assert mol.basis == "cc-pvdz"

    def test_verbose_kwarg_applied(self):
        mol = build_mol("H 0 0 0; H 0 0 0.74", basis="sto-3g", verbose=3)
        assert mol.verbose == 3

    def test_spin_kwarg_applied(self):
        # H atom: 1 electron → spin=1
        mol = build_mol("H 0 0 0", basis="sto-3g", spin=1, verbose=0)
        assert mol.spin == 1

    def test_water_nao(self):
        mol = build_mol(
            "O 0 0 0; H 0 0.96 0; H 0.96 0 0", basis="sto-3g", verbose=0
        )
        assert mol.nao == 7  # STO-3G: O has 5 AOs, each H has 1


# ---------------------------------------------------------------------------
# run_scf
# ---------------------------------------------------------------------------

class TestRunScf:

    def test_rhf_converges(self, h2_rhf):
        _, mf = h2_rhf
        assert mf.converged

    def test_rhf_returns_mo_coeff(self, h2_rhf):
        mol, mf = h2_rhf
        assert mf.mo_coeff.shape == (mol.nao, mol.nao)

    def test_rhf_occupation_sums_to_nelec(self, h2_rhf):
        mol, mf = h2_rhf
        import numpy as np
        assert int(mf.mo_occ.sum()) == mol.tot_electrons()

    def test_rhf_energy_h2_sto3g(self, h2_rhf):
        # Known RHF/STO-3G energy for H2 at r=0.74 Å ≈ -1.117 Ha
        _, mf = h2_rhf
        assert mf.e_tot == pytest.approx(-1.117, abs=0.01)

    def test_invalid_method_raises(self):
        mol = build_mol("H 0 0 0; H 0 0 0.74", basis="sto-3g", verbose=0)
        with pytest.raises(ValueError, match="Unknown method"):
            run_scf(mol, method="INVALID")

    def test_uhf(self):
        # UHF on H2 (closed-shell → same energy as RHF)
        mol = build_mol("H 0 0 0; H 0 0 0.74", basis="sto-3g", verbose=0)
        mf = run_scf(mol, method="UHF")
        assert mf.converged
        assert mf.e_tot == pytest.approx(-1.117, abs=0.01)

    def test_rks_converges(self):
        mol = build_mol("H 0 0 0; H 0 0 0.74", basis="sto-3g", verbose=0)
        mf = run_scf(mol, method="RKS", xc="lda,vwn")
        assert mf.converged
