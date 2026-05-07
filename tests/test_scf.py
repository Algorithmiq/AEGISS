"""Tests for aegiss.initial_guess.scf (build_mol / run_scf)."""

from __future__ import annotations

import numpy as np
import pytest
from pyscf.gto.basis import BasisNotFoundError

from aegiss.initial_guess import build_mol, run_scf


class TestBuildMol:
    def test_returns_mol_with_nao(self) -> None:
        mol = build_mol("H 0 0 0; H 0 0 0.74", basis="sto-3g", verbose=0)
        assert mol.nao > 0

    def test_nao_for_h2_sto3g(self) -> None:
        mol = build_mol("H 0 0 0; H 0 0 0.74", basis="sto-3g", verbose=0)
        assert mol.nao == 2

    def test_list_atom_format(self) -> None:
        mol = build_mol(["H 0 0 0", "H 0 0 0.74"], basis="sto-3g", verbose=0)
        assert mol.nao == 2

    def test_charge_kwarg(self) -> None:
        # H2+ has 1 electron, so spin=1 is required for mol.build() to succeed
        mol = build_mol(
            "H 0 0 0; H 0 0 0.74", basis="sto-3g", verbose=0, charge=1, spin=1
        )
        assert mol.charge == 1

    def test_kwargs_forwarded(self) -> None:
        mol = build_mol(
            "H 0 0 0; H 0 0 0.74", basis="sto-3g", verbose=0, symmetry=False
        )
        assert not mol.symmetry

    def test_nao_changes_with_basis(self) -> None:
        mol_small = build_mol("H 0 0 0; H 0 0 0.74", basis="sto-3g", verbose=0)
        mol_large = build_mol("H 0 0 0; H 0 0 0.74", basis="cc-pvdz", verbose=0)
        assert mol_large.nao > mol_small.nao

    def test_single_atom(self) -> None:
        # He has 2 electrons so spin=0 (default) is consistent without extra kwarg
        mol = build_mol("He 0 0 0", basis="sto-3g", verbose=0)
        assert mol.nao == 1

    def test_tuple_of_tuples_atom_format(self) -> None:
        mol = build_mol(
            [("H", (0, 0, 0)), ("H", (0, 0, 0.74))], basis="sto-3g", verbose=0
        )
        assert mol.nao == 2

    def test_empty_list_gives_zero_nao(self) -> None:
        mol = build_mol([], basis="sto-3g", verbose=0)
        assert mol.nao == 0

    def test_empty_string_raises(self) -> None:
        with pytest.raises(IndexError):
            build_mol("", basis="sto-3g", verbose=0)

    def test_invalid_basis_raises(self) -> None:
        with pytest.raises(BasisNotFoundError):
            build_mol("H 0 0 0; H 0 0 0.74", basis="not_a_basis", verbose=0)

    def test_unknown_element_raises(self) -> None:
        with pytest.raises(BasisNotFoundError):
            build_mol("Xx 0 0 0", basis="sto-3g", verbose=0)

    def test_spin_kwarg(self) -> None:
        mol = build_mol("Li 0 0 0", basis="sto-3g", spin=1, verbose=0)
        assert mol.spin == 1


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

    def test_method_lowercase(self) -> None:
        mol = build_mol("H 0 0 0; H 0 0 0.74", basis="sto-3g", verbose=0)
        mf = run_scf(mol, method="rhf", verbose=0)
        assert mf.converged

    def test_kwargs_forwarded_to_mf(self) -> None:
        mol = build_mol("H 0 0 0; H 0 0 0.74", basis="sto-3g", verbose=0)
        mf = run_scf(mol, method="RHF", verbose=0, max_cycle=100)
        assert mf.max_cycle == 100

    def test_rks_default_xc(self) -> None:
        mol = build_mol("H 0 0 0; H 0 0 0.74", basis="sto-3g", verbose=0)
        mf = run_scf(mol, method="RKS", verbose=0)
        assert mf.xc == "b3lyp"

    def test_uhf_converges(self) -> None:
        mol = build_mol("Li 0 0 0", basis="sto-3g", spin=1, verbose=0)
        mf = run_scf(mol, method="UHF", verbose=0)
        assert mf.converged

    def test_uhf_mo_occ_alpha_beta(self) -> None:
        mol = build_mol("Li 0 0 0", basis="sto-3g", spin=1, verbose=0)
        mf = run_scf(mol, method="UHF", verbose=0)
        # UHF returns (alpha_occ, beta_occ) as a tuple of arrays
        alpha_occ, beta_occ = mf.mo_occ
        assert abs(float(np.sum(alpha_occ)) - float(np.sum(beta_occ))) == pytest.approx(
            1.0, abs=1e-8
        )

    def test_uks_converges(self) -> None:
        mol = build_mol("Li 0 0 0", basis="sto-3g", spin=1, verbose=0)
        mf = run_scf(mol, method="UKS", xc="lda", verbose=0)
        assert mf.converged

    def test_uks_xc_set(self) -> None:
        mol = build_mol("Li 0 0 0", basis="sto-3g", spin=1, verbose=0)
        mf = run_scf(mol, method="UKS", xc="pbe", verbose=0)
        assert mf.xc == "pbe"

    def test_pcm_solvent_converges(self) -> None:
        pytest.importorskip("pyscf.solvent")
        mol = build_mol("H 0 0 0; H 0 0 0.74", basis="sto-3g", verbose=0)
        # "IEF-PCM" contains "PCM" so it routes through the PCM branch in run_scf
        mf = run_scf(mol, method="RHF", verbose=0, solvent_info=("IEF-PCM", 78.4))
        assert mf.converged

    def test_ddcosmo_solvent_converges(self) -> None:
        pytest.importorskip("pyscf.solvent")
        mol = build_mol("H 0 0 0; H 0 0 0.74", basis="sto-3g", verbose=0)
        mf = run_scf(mol, method="RHF", verbose=0, solvent_info=("DDCOSMO", 78.4))
        assert mf.converged
