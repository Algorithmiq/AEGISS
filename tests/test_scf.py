"""Tests for aegiss.initial_guess.scf (build_mol / run_scf)."""

from __future__ import annotations

import numpy as np
import pytest

from aegiss.initial_guess import build_mol, run_scf


class TestBuildMol:
    def test_returns_mol_with_nao(self) -> None:
        mol = build_mol("H 0 0 0; H 0 0 0.74", basis="sto-3g", verbose=0)
        assert mol.nao > 0

    def test_nao_for_h2_sto3g(self) -> None:
        mol = build_mol("H 0 0 0; H 0 0 0.74", basis="sto-3g", verbose=0)
        assert mol.nao == 2

    def test_kwargs_forwarded(self) -> None:
        mol = build_mol(
            "H 0 0 0; H 0 0 0.74", basis="sto-3g", verbose=0, symmetry=False
        )
        assert not mol.symmetry

    def test_nao_changes_with_basis(self) -> None:
        mol_small = build_mol("H 0 0 0; H 0 0 0.74", basis="sto-3g", verbose=0)
        mol_large = build_mol("H 0 0 0; H 0 0 0.74", basis="cc-pvdz", verbose=0)
        assert mol_large.nao > mol_small.nao


class TestRunScf:
    def test_h2_rhf_converges(self, h2_rhf) -> None:
        assert h2_rhf.converged

    def test_h2_rhf_energy_sign(self, h2_rhf) -> None:
        assert h2_rhf.e_tot < 0.0

    def test_h2_rhf_mo_coeff_shape(self, h2_rhf) -> None:
        # H2 sto-3g: 2 AOs → 2 MOs
        assert h2_rhf.mo_coeff.shape == (2, 2)

    def test_h2_rhf_mo_occ_sum(self, h2_rhf) -> None:
        # H2 has 2 electrons
        assert abs(float(np.sum(h2_rhf.mo_occ)) - 2.0) < 1e-10

    def test_unknown_method_raises(self) -> None:
        mol = build_mol("H 0 0 0; H 0 0 0.74", basis="sto-3g", verbose=0)
        with pytest.raises(ValueError, match="Unknown method"):
            run_scf(mol, method="INVALID")

    def test_rks_converges(self) -> None:
        mol = build_mol("H 0 0 0; H 0 0 0.74", basis="sto-3g", verbose=0)
        mf = run_scf(mol, method="RKS", xc="lda", verbose=0)
        assert mf.converged

    def test_rks_energy_lower_than_rhf(self) -> None:
        """LDA energy for H2 should differ from RHF."""
        mol = build_mol("H 0 0 0; H 0 0 0.74", basis="sto-3g", verbose=0)
        mf_rhf = run_scf(mol, method="RHF", verbose=0)
        mf_rks = run_scf(mol, method="RKS", xc="lda", verbose=0)
        assert mf_rhf.e_tot != pytest.approx(mf_rks.e_tot, rel=1e-4)
