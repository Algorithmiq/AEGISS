"""Tests for aegiss.active_space.ao_projection."""
from __future__ import annotations

import numpy as np
import pytest
from aegiss.active_space.ao_projection import (
    _compute_contribution,
    _get_s22_from_ao_labels,
    _get_secondary_basis_s21,
)


# ---------------------------------------------------------------------------
# _get_secondary_basis_s21  (requires PySCF)
# ---------------------------------------------------------------------------

class TestGetSecondaryBasisS21:

    def test_shape_minao(self, h2_rhf):
        mol, _ = h2_rhf
        s21 = _get_secondary_basis_s21(mol, secondary_basis="minao")
        # rows = n_minao AOs, cols = n_original AOs
        assert s21.ndim == 2
        assert s21.shape[1] == mol.nao  # columns match original basis

    def test_identity_basis_is_square(self, h2_rhf):
        """When projected_basis matches the mol basis (sto-3g), result is square."""
        mol, _ = h2_rhf
        s21 = _get_secondary_basis_s21(mol, secondary_basis="sto-3g")
        assert s21.shape[0] == s21.shape[1]

    def test_none_basis_uses_original_mol(self, h2_rhf):
        """secondary_basis=None returns integrals against mol itself → square."""
        mol, _ = h2_rhf
        s21 = _get_secondary_basis_s21(mol, secondary_basis=None)
        assert s21.shape[0] == mol.nao
        assert s21.shape[1] == mol.nao


# ---------------------------------------------------------------------------
# _get_s22_from_ao_labels  (requires PySCF)
# ---------------------------------------------------------------------------

class TestGetS22FromAoLabels:

    def test_valid_label_returns_matrix(self, h2_rhf):
        mol, mf = h2_rhf
        s21 = _get_secondary_basis_s21(mol, secondary_basis="minao")
        # H 1s labels are present in minao for H2
        s22 = _get_s22_from_ao_labels(mol, "H 1s", s21, mf.mo_coeff)
        assert s22.ndim == 2
        assert s22.shape[1] == mf.mo_coeff.shape[1]  # one col per MO

    def test_output_shape_matches_n_mos(self, h2_rhf):
        mol, mf = h2_rhf
        s21 = _get_secondary_basis_s21(mol, secondary_basis="minao")
        s22 = _get_s22_from_ao_labels(mol, "H 1s", s21, mf.mo_coeff)
        assert s22.shape[1] == mf.mo_coeff.shape[1]


# ---------------------------------------------------------------------------
# _compute_contribution  (pure numpy, no PySCF needed)
# ---------------------------------------------------------------------------

class TestComputeContribution:
    """_compute_contribution currently always returns (None, None).
    Tests document the expected interface and verify the current behaviour.
    """

    def _make_s22(self, n_aos=3, n_mos=4):
        rng = np.random.default_rng(0)
        return rng.random((n_aos, n_mos))

    def test_returns_two_values(self):
        s22 = self._make_s22()
        reduced = [(i, 0.5) for i in range(s22.shape[1])]
        result = _compute_contribution(s22, "H 1s", reduced, 0.3, "a")
        assert len(result) == 2

    def test_currently_returns_none_none(self):
        # NOTE: _compute_contribution does not yet return selection results.
        # This test documents the known incomplete state; update when implemented.
        s22 = self._make_s22()
        reduced = [(i, 0.5) for i in range(s22.shape[1])]
        s, above = _compute_contribution(s22, "test", reduced, 0.3, "a")
        assert s is None
        assert above is None

    def test_relative_mode_normalises_by_max(self, capsys):
        # With contribution_type='r', contributions are normalised by the max.
        # The max normalised value should be 1.0 (printed to stdout).
        s22 = np.array([[1.0, 0.0, 0.5]])  # 1 AO row, 3 MOs
        reduced = [(0, 0.8), (1, 0.6), (2, 0.4)]
        _compute_contribution(s22, "test", reduced, 0.5, "r")
        captured = capsys.readouterr().out
        # Orbital 0 has max contribution → normalised to 1.0
        assert "1.0000" in captured

    def test_absolute_mode_uses_raw_contribution(self, capsys):
        # Raw sum of |j| for orbital 0 is 1.0; should appear in output
        s22 = np.array([[1.0, 0.2]])
        reduced = [(0, 0.8), (1, 0.6)]
        _compute_contribution(s22, "test", reduced, 0.5, "a")
        captured = capsys.readouterr().out
        assert "1.0000" in captured

    def test_above_below_separator_printed(self, capsys):
        s22 = np.array([[0.8, 0.3]])
        reduced = [(0, 0.5), (1, 0.5)]
        _compute_contribution(s22, "O 2p", reduced, 0.5, "a")
        out = capsys.readouterr().out
        assert "above" in out
        assert "below" in out
