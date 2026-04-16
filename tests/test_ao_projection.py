"""Tests for aegiss.active_space.ao_projection."""

from __future__ import annotations

import numpy as np
import pytest

from aegiss.active_space.ao_projection import (
    _compute_contribution,
    select_by_ao_projection,
    select_by_label,
)


class TestComputeContribution:
    def test_returns_list_of_tuples(self, h2_rhf) -> None:
        space = [(0, 0.0), (1, 0.0)]
        result = _compute_contribution(
            mf=h2_rhf,
            label="H 1s",
            active_mo_coeff=h2_rhf.mo_coeff,
            reduced_space_indices=space,
            contribution=0.0,
            contribution_type="r",
            secondary_basis="minao",
        )
        assert isinstance(result, list)
        for item in result:
            assert len(item) == 2

    def test_returns_empty_for_bad_label(self, h2_rhf) -> None:
        """A label matching no AOs in the projection mol returns []."""
        space = [(0, 0.0), (1, 0.0)]
        result = _compute_contribution(
            mf=h2_rhf,
            label="Xe 1s",  # no Xe in H2
            active_mo_coeff=h2_rhf.mo_coeff,
            reduced_space_indices=space,
            contribution=0.0,
            contribution_type="a",
            secondary_basis="minao",
        )
        assert result == []

    def test_high_threshold_filters_all(self, h2_rhf) -> None:
        """Contribution threshold of 2.0 (relative) should select nothing."""
        space = [(0, 0.0), (1, 0.0)]
        result = _compute_contribution(
            mf=h2_rhf,
            label="H 1s",
            active_mo_coeff=h2_rhf.mo_coeff,
            reduced_space_indices=space,
            contribution=2.0,
            contribution_type="r",
            secondary_basis="minao",
        )
        assert result == []

    def test_zero_threshold_selects_all(self, h2_rhf) -> None:
        """Contribution threshold of 0.0 selects every orbital."""
        space = [(0, 0.0), (1, 0.0)]
        result = _compute_contribution(
            mf=h2_rhf,
            label="H 1s",
            active_mo_coeff=h2_rhf.mo_coeff,
            reduced_space_indices=space,
            contribution=0.0,
            contribution_type="a",
            secondary_basis="minao",
        )
        assert len(result) == 2

    def test_no_secondary_basis_uses_original(self, h2_rhf) -> None:
        space = [(0, 0.0), (1, 0.0)]
        result = _compute_contribution(
            mf=h2_rhf,
            label="H 1s",
            active_mo_coeff=h2_rhf.mo_coeff,
            reduced_space_indices=space,
            contribution=0.0,
            contribution_type="a",
            secondary_basis=None,
        )
        assert isinstance(result, list)

    def test_mismatched_mo_coeff_raises(self, h2_rhf) -> None:
        """Wrong-shape active_mo_coeff should raise ValueError."""
        bad_coeff = np.zeros((5, 2))  # 5 rows instead of 2
        space = [(0, 0.0), (1, 0.0)]
        with pytest.raises(ValueError, match="Mismatch"):
            _compute_contribution(
                mf=h2_rhf,
                label="H 1s",
                active_mo_coeff=bad_coeff,
                reduced_space_indices=space,
                contribution=0.0,
                contribution_type="a",
                secondary_basis="minao",
            )

    def test_weights_are_non_negative(self, h2_rhf) -> None:
        space = [(0, 0.0), (1, 0.0)]
        result = _compute_contribution(
            mf=h2_rhf,
            label="H 1s",
            active_mo_coeff=h2_rhf.mo_coeff,
            reduced_space_indices=space,
            contribution=0.0,
            contribution_type="a",
            secondary_basis="minao",
        )
        for _, w in result:
            assert w >= 0.0


class TestSelectByLabel:
    """select_by_label is a thin wrapper — just verify it delegates correctly."""

    def test_returns_list(self, h2_rhf) -> None:
        space = [(0, 0.0), (1, 0.0)]
        result = select_by_label(
            label="H 1s",
            mf=h2_rhf,
            active_mo_coeff=h2_rhf.mo_coeff,
            space=space,
            contribution=0.0,
            projected_basis="minao",
            contribution_type="a",
        )
        assert isinstance(result, list)

    def test_consistent_with_compute_contribution(self, h2_rhf) -> None:
        space = [(0, 0.0), (1, 0.0)]
        direct = _compute_contribution(
            mf=h2_rhf,
            label="H 1s",
            active_mo_coeff=h2_rhf.mo_coeff,
            reduced_space_indices=space,
            contribution=0.5,
            contribution_type="r",
            secondary_basis="minao",
        )
        via_label = select_by_label(
            label="H 1s",
            mf=h2_rhf,
            active_mo_coeff=h2_rhf.mo_coeff,
            space=space,
            contribution=0.5,
            projected_basis="minao",
            contribution_type="r",
        )
        assert direct == via_label


class TestSelectByAoProjection:
    def test_returns_ndarray(self, h2_rhf) -> None:
        result = select_by_ao_projection(
            mf=h2_rhf,
            ao_labels=["H 1s"],
            projected_basis="minao",
            contribution=0.0,
            contribution_type="a",
        )
        assert isinstance(result, np.ndarray)

    def test_indices_in_valid_range(self, h2_rhf) -> None:
        result = select_by_ao_projection(
            mf=h2_rhf,
            ao_labels=["H 1s"],
            projected_basis="minao",
            contribution=0.0,
            contribution_type="a",
        )
        n_mo = h2_rhf.mo_coeff.shape[1]
        assert all(0 <= idx < n_mo for idx in result)

    def test_sorted_output(self, h2_rhf) -> None:
        result = select_by_ao_projection(
            mf=h2_rhf,
            ao_labels=["H 1s"],
            projected_basis="minao",
            contribution=0.0,
            contribution_type="a",
        )
        assert list(result) == sorted(result)

    def test_no_duplicates(self, h2_rhf) -> None:
        result = select_by_ao_projection(
            mf=h2_rhf,
            ao_labels=["H 1s"],
            projected_basis="minao",
            contribution=0.0,
            contribution_type="a",
        )
        assert len(result) == len(set(result.tolist()))

    def test_high_threshold_gives_fewer_orbitals(self, h2_rhf) -> None:
        low = select_by_ao_projection(
            mf=h2_rhf,
            ao_labels=["H 1s"],
            projected_basis="minao",
            contribution=0.0,
            contribution_type="a",
        )
        high = select_by_ao_projection(
            mf=h2_rhf,
            ao_labels=["H 1s"],
            projected_basis="minao",
            contribution=0.9,
            contribution_type="r",
        )
        assert len(high) <= len(low)

    def test_top_n_limits_selection(self, h2_rhf) -> None:
        result = select_by_ao_projection(
            mf=h2_rhf,
            ao_labels=["H 1s"],
            projected_basis="minao",
            top_n=1,
        )
        assert len(result) <= 1

    def test_restricted_space_tuple(self, h2_rhf) -> None:
        """Tuple restricted_space (occ, virt) is combined internally."""
        occ_space = [(0, 0.5)]
        virt_space = [(1, 0.2)]
        result = select_by_ao_projection(
            mf=h2_rhf,
            ao_labels=["H 1s"],
            projected_basis="minao",
            contribution=0.0,
            contribution_type="a",
            restricted_space=(occ_space, virt_space),
        )
        assert isinstance(result, np.ndarray)
        # All returned indices must be from the restricted space
        allowed = {0, 1}
        for idx in result:
            assert idx in allowed

    def test_restricted_space_list(self, h2_rhf) -> None:
        """List restricted_space is used directly."""
        restricted = [(0, 0.5), (1, 0.2)]
        result = select_by_ao_projection(
            mf=h2_rhf,
            ao_labels=["H 1s"],
            projected_basis="minao",
            contribution=0.0,
            contribution_type="a",
            restricted_space=restricted,
        )
        allowed = {0, 1}
        for idx in result:
            assert idx in allowed

    def test_no_projection_basis(self, h2_rhf) -> None:
        result = select_by_ao_projection(
            mf=h2_rhf,
            ao_labels=["H 1s"],
            projected_basis=None,
            contribution=0.0,
            contribution_type="a",
        )
        assert isinstance(result, np.ndarray)
