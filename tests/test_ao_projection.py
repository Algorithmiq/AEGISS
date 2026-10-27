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

"""Tests for aegiss.active_space.ao_projection."""

from __future__ import annotations

from typing import ClassVar
from unittest.mock import patch

import numpy as np
import pytest

from aegiss.active_space.ao_projection import (
    _compute_contribution,
    _resolve_contribution,
    _resolve_top_n,
    select_by_ao_projection,
    select_by_label,
)
from aegiss.utils.plotting import plot_ao_contribution


class TestComputeContributionValues:
    """Pin AO-projection weights against reference values to catch pyscf version regressions.

    Each case: (fixture_name, label, sec_basis, expected_weights_by_mo_index)
    expected_weights are sorted by MO index (ascending) and represent absolute contributions.
    Tolerance is 1e-6; values were generated with the project's pinned pyscf version.
    """

    _CASES: ClassVar = [
        # He atom — single 1s MO, minao projection
        ("he_rhf", "He 1s", "minao", [0.99832646]),
        # Be atom — 5 MOs (1s, 2s, 2px, 2py, 2pz), minao projection
        ("be_rhf", "Be 1s", "minao", [0.99968735, 0.00548893, 0.0, 0.0, 0.0]),
        ("be_rhf", "Be 2s", "minao", [0.00760731, 0.98117295, 0.0, 0.0, 0.0]),
        # Ne atom — 5 MOs, minao projection
        ("ne_rhf", "Ne 2s", "minao", [0.00229484, 0.99861592, 0.0, 0.0, 0.0]),
        ("ne_rhf", "Ne 2p", "minao", [0.0, 0.0, 0.98561349, 0.98561349, 0.98561349]),
        # No secondary basis (projection onto molecule's own AOs)
        ("he_rhf", "He 1s", None, [1.0]),
        ("be_rhf", "Be 2s", None, [0.28381188, 0.95887998, 0.0, 0.0, 0.0]),
    ]

    @pytest.mark.parametrize(("fixture_name", "label", "sec_basis", "expected"), _CASES)
    def test_weight_values(
        self, request, fixture_name, label, sec_basis, expected
    ) -> None:
        mf = request.getfixturevalue(fixture_name)
        mo = mf.mo_coeff if mf.mo_coeff.ndim == 2 else mf.mo_coeff[0]
        n_mo = mo.shape[1]
        space = [(i, 0.0) for i in range(n_mo)]
        _, all_weights = _compute_contribution(
            mf=mf,
            label=label,
            full_mo_coeff=mo,
            reduced_space_indices=space,
            contribution=0.0,
            contribution_type="a",
            secondary_basis=sec_basis,
        )
        weights = [w for _, w in sorted(all_weights, key=lambda x: x[0])]
        assert weights == pytest.approx(expected, abs=1e-6)


class TestComputeContribution:
    def test_returns_tuple_of_lists(self, h2_rhf) -> None:
        space = [(0, 0.0), (1, 0.0)]
        selected, all_weights = _compute_contribution(
            mf=h2_rhf,
            label="H 1s",
            full_mo_coeff=h2_rhf.mo_coeff,
            reduced_space_indices=space,
            contribution=0.0,
            contribution_type="r",
            secondary_basis="minao",
        )
        assert isinstance(selected, list)
        assert isinstance(all_weights, list)
        for item in selected:
            assert len(item) == 2

    def test_returns_empty_tuple_for_bad_label(self, h2_rhf) -> None:
        """A label matching no AOs in the projection mol returns ([], [])."""
        space = [(0, 0.0), (1, 0.0)]
        selected, all_weights = _compute_contribution(
            mf=h2_rhf,
            label="Xe 1s",  # no Xe in H2
            full_mo_coeff=h2_rhf.mo_coeff,
            reduced_space_indices=space,
            contribution=0.0,
            contribution_type="a",
            secondary_basis="minao",
        )
        assert selected == []
        assert all_weights == []

    def test_high_threshold_filters_selected_but_keeps_all_weights(
        self, h2_rhf
    ) -> None:
        """Threshold of 2.0 (relative) selects nothing but all_weights still covers all orbitals."""
        space = [(0, 0.0), (1, 0.0)]
        selected, all_weights = _compute_contribution(
            mf=h2_rhf,
            label="H 1s",
            full_mo_coeff=h2_rhf.mo_coeff,
            reduced_space_indices=space,
            contribution=2.0,
            contribution_type="r",
            secondary_basis="minao",
        )
        assert selected == []
        assert len(all_weights) == 2

    def test_zero_threshold_selects_all(self, h2_rhf) -> None:
        """Contribution threshold of 0.0 selects every orbital."""
        space = [(0, 0.0), (1, 0.0)]
        selected, _ = _compute_contribution(
            mf=h2_rhf,
            label="H 1s",
            full_mo_coeff=h2_rhf.mo_coeff,
            reduced_space_indices=space,
            contribution=0.0,
            contribution_type="a",
            secondary_basis="minao",
        )
        assert len(selected) == 2

    def test_all_weights_covers_full_space(self, h2_rhf) -> None:
        """all_weights always has one entry per orbital in the space regardless of threshold."""
        space = [(0, 0.0), (1, 0.0)]
        _, all_weights = _compute_contribution(
            mf=h2_rhf,
            label="H 1s",
            full_mo_coeff=h2_rhf.mo_coeff,
            reduced_space_indices=space,
            contribution=0.9,
            contribution_type="r",
            secondary_basis="minao",
        )
        assert len(all_weights) == len(space)

    def test_no_secondary_basis_uses_original(self, h2_rhf) -> None:
        space = [(0, 0.0), (1, 0.0)]
        selected, all_weights = _compute_contribution(
            mf=h2_rhf,
            label="H 1s",
            full_mo_coeff=h2_rhf.mo_coeff,
            reduced_space_indices=space,
            contribution=0.0,
            contribution_type="a",
            secondary_basis=None,
        )
        assert isinstance(selected, list)
        assert isinstance(all_weights, list)

    def test_mismatched_mo_coeff_raises(self, h2_rhf) -> None:
        """Wrong-shape active_mo_coeff should raise ValueError."""
        bad_coeff = np.zeros((5, 2))  # 5 rows instead of 2
        space = [(0, 0.0), (1, 0.0)]
        with pytest.raises(ValueError, match="Mismatch"):
            _compute_contribution(
                mf=h2_rhf,
                label="H 1s",
                full_mo_coeff=bad_coeff,
                reduced_space_indices=space,
                contribution=0.0,
                contribution_type="a",
                secondary_basis="minao",
            )

    def test_weights_are_non_negative(self, h2_rhf) -> None:
        space = [(0, 0.0), (1, 0.0)]
        selected, all_weights = _compute_contribution(
            mf=h2_rhf,
            label="H 1s",
            full_mo_coeff=h2_rhf.mo_coeff,
            reduced_space_indices=space,
            contribution=0.0,
            contribution_type="a",
            secondary_basis="minao",
        )
        for _, w in selected:
            assert w >= 0.0
        for _, w in all_weights:
            assert w >= 0.0


class TestSelectByLabel:
    """select_by_label is a thin wrapper — just verify it delegates correctly."""

    def test_returns_tuple_of_lists(self, h2_rhf) -> None:
        space = [(0, 0.0), (1, 0.0)]
        selected, all_weights = select_by_label(
            label="H 1s",
            mf=h2_rhf,
            full_mo_coeff=h2_rhf.mo_coeff,
            space=space,
            contribution=0.0,
            projected_basis="minao",
            contribution_type="a",
        )
        assert isinstance(selected, list)
        assert isinstance(all_weights, list)

    def test_consistent_with_compute_contribution(self, h2_rhf) -> None:
        space = [(0, 0.0), (1, 0.0)]
        direct = _compute_contribution(
            mf=h2_rhf,
            label="H 1s",
            full_mo_coeff=h2_rhf.mo_coeff,
            reduced_space_indices=space,
            contribution=0.5,
            contribution_type="r",
            secondary_basis="minao",
        )
        via_label = select_by_label(
            label="H 1s",
            mf=h2_rhf,
            full_mo_coeff=h2_rhf.mo_coeff,
            space=space,
            contribution=0.5,
            projected_basis="minao",
            contribution_type="r",
        )
        assert direct == via_label


class TestResolveContribution:
    def test_scalar_broadcast(self) -> None:
        result = _resolve_contribution(0.4, 3)
        assert result == [0.4, 0.4, 0.4]

    def test_scalar_single_label(self) -> None:
        assert _resolve_contribution(0.7, 1) == [0.7]

    def test_list_exact_length_returned_as_is(self) -> None:
        thresholds = [0.1, 0.5, 0.9]
        assert _resolve_contribution(thresholds, 3) == thresholds

    def test_list_wrong_length_broadcasts_first(self) -> None:
        result = _resolve_contribution([0.3, 0.6], 4)
        assert result == [0.3, 0.3, 0.3, 0.3]

    def test_list_too_short_by_one_broadcasts_first(self) -> None:
        result = _resolve_contribution([0.2, 0.8], 3)
        assert result == [0.2, 0.2, 0.2]

    def test_list_too_long_broadcasts_first(self) -> None:
        result = _resolve_contribution([0.5, 0.6, 0.7], 2)
        assert result == [0.5, 0.5]

    def test_zero_threshold_broadcast(self) -> None:
        assert _resolve_contribution(0.0, 2) == [0.0, 0.0]

    def test_output_length_matches_n_labels(self) -> None:
        for n in (1, 2, 5):
            assert len(_resolve_contribution(0.5, n)) == n
            assert len(_resolve_contribution([0.1, 0.2], n)) == n


class TestResolveTopN:
    def test_scalar_broadcast(self) -> None:
        result = _resolve_top_n(3, 4)
        assert result == [3, 3, 3, 3]

    def test_scalar_single_label(self) -> None:
        assert _resolve_top_n(5, 1) == [5]

    def test_list_exact_length_returned_as_is(self) -> None:
        counts = [2, 4, 6]
        assert _resolve_top_n(counts, 3) == counts

    def test_list_wrong_length_broadcasts_first(self) -> None:
        result = _resolve_top_n([3, 7], 4)
        assert result == [3, 3, 3, 3]

    def test_list_too_short_by_one_broadcasts_first(self) -> None:
        result = _resolve_top_n([6, 3], 3)
        assert result == [6, 6, 6]

    def test_list_too_long_broadcasts_first(self) -> None:
        result = _resolve_top_n([2, 4, 8], 2)
        assert result == [2, 2]

    def test_output_length_matches_n_labels(self) -> None:
        for n in (1, 2, 5):
            assert len(_resolve_top_n(1, n)) == n
            assert len(_resolve_top_n([1, 2], n)) == n

    def test_output_values_are_ints(self) -> None:
        for v in _resolve_top_n(4, 3):
            assert isinstance(v, int)

        for v in _resolve_top_n([2, 6], 2):
            assert isinstance(v, int)


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

    def test_top_n_int_limits_selection(self, h2_rhf) -> None:
        result = select_by_ao_projection(
            mf=h2_rhf,
            ao_labels=["H 1s"],
            projected_basis="minao",
            top_n=1,
        )
        assert len(result) <= 1

    def test_top_n_list_limits_per_group(self, h2_rhf) -> None:
        """top_n as a list must apply per-group caps — regression test for list bug."""
        result = select_by_ao_projection(
            mf=h2_rhf,
            ao_labels=["H 1s", "H 1s"],
            projected_basis="minao",
            top_n=[1, 1],
        )
        # Each group picks at most 1 orbital; union is at most 2
        assert len(result) <= 2

    def test_top_n_list_does_not_return_all_orbitals(self, h2_rhf) -> None:
        """Ensure the list-top_n bug (returning all MOs) does not regress."""
        n_mo = h2_rhf.mo_coeff.shape[1]
        result = select_by_ao_projection(
            mf=h2_rhf,
            ao_labels=["H 1s", "H 1s"],
            projected_basis="minao",
            top_n=[1, 1],
        )
        assert len(result) < n_mo

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

    def test_grouped_ao_labels_accepted(self, h2_rhf) -> None:
        """Mixed list[str | list[str]] is handled without error."""
        result = select_by_ao_projection(
            mf=h2_rhf,
            ao_labels=["H 1s", ["H 1s"]],
            projected_basis="minao",
            contribution=0.0,
            contribution_type="a",
        )
        assert isinstance(result, np.ndarray)

    def test_save_path_none_does_not_call_plot(self, h2_rhf) -> None:
        """No heatmap is saved when save_path is None."""
        with patch(
            "aegiss.active_space.ao_projection.plot_ao_contribution"
        ) as mock_plot:
            select_by_ao_projection(
                mf=h2_rhf,
                ao_labels=["H 1s"],
                projected_basis="minao",
                contribution=0.0,
                contribution_type="a",
                save_path=None,
            )
        mock_plot.assert_not_called()

    def test_save_path_calls_plot_and_creates_pdf(self, h2_rhf, tmp_path) -> None:
        """Passing save_path triggers plot_ao_contribution and writes a PDF."""
        save_stem = str(tmp_path / "ao_test")
        select_by_ao_projection(
            mf=h2_rhf,
            ao_labels=["H 1s"],
            projected_basis="minao",
            contribution=0.0,
            contribution_type="a",
            save_path=save_stem,
        )
        assert (tmp_path / "ao_test_heatmap.pdf").exists()


class TestSelectByAoProjectionUnrestricted:
    """UHF (3D ``mo_coeff``) → independent split alpha/beta selection (issue #52).

    The AO overlaps are spin-independent, so each channel is screened with its
    own ``mo_coeff[spin]`` block and the result is a ``{"alpha", "beta"}`` dict.
    """

    def test_returns_split_dict(self, c_sto3g_uhf) -> None:
        result = select_by_ao_projection(
            mf=c_sto3g_uhf,
            ao_labels=["C 2p"],
            projected_basis="minao",
            contribution=0.0,
            contribution_type="a",
        )
        assert isinstance(result, dict)
        assert set(result) == {"alpha", "beta"}
        for arr in result.values():
            assert isinstance(arr, np.ndarray)

    def test_each_channel_valid_sorted_unique(self, c_sto3g_uhf) -> None:
        result = select_by_ao_projection(
            mf=c_sto3g_uhf,
            ao_labels=["C 2p"],
            projected_basis="minao",
            contribution=0.0,
            contribution_type="a",
        )
        n_mo = c_sto3g_uhf.mo_coeff.shape[-1]
        for arr in result.values():
            assert all(0 <= idx < n_mo for idx in arr)
            assert list(arr) == sorted(arr)
            assert len(arr) == len(set(arr.tolist()))

    def test_restricted_space_none_uses_nmo_not_spin_axis(self, c_sto3g_uhf) -> None:
        """``_resolve_space`` must count MOs (shape[-1]), not the spin axis."""
        result = select_by_ao_projection(
            mf=c_sto3g_uhf,
            ao_labels=["C 1s"],
            projected_basis="minao",
            contribution=0.0,
            contribution_type="a",
            restricted_space=None,
        )
        n_mo = c_sto3g_uhf.mo_coeff.shape[-1]
        assert n_mo > 2  # guards against slicing the size-2 spin axis
        for arr in result.values():
            assert all(0 <= idx < n_mo for idx in arr)

    def test_top_n_applies_per_channel(self, o_sto3g_uhf) -> None:
        result = select_by_ao_projection(
            mf=o_sto3g_uhf,
            ao_labels=["O 2p"],
            projected_basis="minao",
            top_n=2,
        )
        assert len(result["alpha"]) <= 2
        assert len(result["beta"]) <= 2

    def test_per_channel_plot_paths(self, c_sto3g_uhf, tmp_path) -> None:
        """Alpha/beta heatmaps are written to separate spin-tagged files."""
        save_stem = str(tmp_path / "ao_uhf")
        select_by_ao_projection(
            mf=c_sto3g_uhf,
            ao_labels=["C 2p"],
            projected_basis="minao",
            contribution=0.0,
            contribution_type="a",
            save_path=save_stem,
        )
        assert (tmp_path / "ao_uhf_alpha_heatmap.pdf").exists()
        assert (tmp_path / "ao_uhf_beta_heatmap.pdf").exists()


class TestSelectByAoProjectionSpinResolvedSpace:
    """Spin-resolved (per-channel) ``restricted_space`` dict for a UHF reference."""

    def test_each_channel_restricted_to_own_space(self, c_sto3g_uhf) -> None:
        """A ``{"alpha", "beta"}`` restricted_space confines each spin channel."""
        restricted_space = {
            "alpha": ([(0, 0.0), (1, 0.0)], []),
            "beta": ([(2, 0.0), (3, 0.0)], []),
        }
        result = select_by_ao_projection(
            mf=c_sto3g_uhf,
            ao_labels=["C 2p"],
            projected_basis="minao",
            contribution=0.0,
            contribution_type="a",
            restricted_space=restricted_space,
        )
        assert isinstance(result, dict)
        assert set(result["alpha"].tolist()) <= {0, 1}
        assert set(result["beta"].tolist()) <= {2, 3}

    def test_empty_channel_gives_empty_array(self, c_sto3g_uhf) -> None:
        """A channel with an empty window selects nothing (no ``max()`` crash)."""
        restricted_space = {
            "alpha": ([], []),
            "beta": ([(2, 0.0)], []),
        }
        result = select_by_ao_projection(
            mf=c_sto3g_uhf,
            ao_labels=["C 2p"],
            projected_basis="minao",
            contribution=0.9,
            contribution_type="r",
            restricted_space=restricted_space,
        )
        assert isinstance(result, dict)
        assert result["alpha"].tolist() == []
        assert set(result["beta"].tolist()) <= {2}

    def test_dict_space_requires_unrestricted(self, h2_rhf) -> None:
        """A spin-resolved restricted_space on a restricted mf is rejected."""
        with pytest.raises(ValueError, match="unrestricted"):
            select_by_ao_projection(
                mf=h2_rhf,
                ao_labels=["H 1s"],
                projected_basis="minao",
                contribution=0.0,
                contribution_type="a",
                restricted_space={"alpha": ([(0, 0.0)], []), "beta": ([(1, 0.0)], [])},
            )


class TestSelectByAoProjectionSpinResolvedContribution:
    """Spin-resolved (per-channel) ``contribution`` dict for a UHF reference."""

    def test_per_channel_thresholds_applied(self, c_sto3g_uhf) -> None:
        """Each channel screens against its own contribution threshold."""
        loose = select_by_ao_projection(
            mf=c_sto3g_uhf,
            ao_labels=["C 2p"],
            projected_basis="minao",
            contribution={"alpha": 0.0, "beta": 0.0},
            contribution_type="r",
        )
        split = select_by_ao_projection(
            mf=c_sto3g_uhf,
            ao_labels=["C 2p"],
            projected_basis="minao",
            # A strict beta cut should not select more than the loose one.
            contribution={"alpha": 0.0, "beta": 0.99},
            contribution_type="r",
        )
        assert set(split["alpha"].tolist()) == set(loose["alpha"].tolist())
        assert set(split["beta"].tolist()) <= set(loose["beta"].tolist())

    def test_dict_matches_scalar_when_channels_equal(self, c_sto3g_uhf) -> None:
        """A dict with equal entries reproduces the shared-threshold result."""
        shared = select_by_ao_projection(
            mf=c_sto3g_uhf,
            ao_labels=["C 2p"],
            projected_basis="minao",
            contribution=0.5,
            contribution_type="r",
        )
        as_dict = select_by_ao_projection(
            mf=c_sto3g_uhf,
            ao_labels=["C 2p"],
            projected_basis="minao",
            contribution={"alpha": 0.5, "beta": 0.5},
            contribution_type="r",
        )
        for channel in ("alpha", "beta"):
            assert as_dict[channel].tolist() == shared[channel].tolist()

    def test_per_channel_list_thresholds(self, c_sto3g_uhf) -> None:
        """Each channel's value may itself be a per-AO-label list."""
        result = select_by_ao_projection(
            mf=c_sto3g_uhf,
            ao_labels=["C 2s", "C 2p"],
            projected_basis="minao",
            contribution={"alpha": [0.0, 0.0], "beta": [0.5, 0.9]},
            contribution_type="r",
        )
        assert isinstance(result, dict)
        assert set(result) == {"alpha", "beta"}

    def test_dict_contribution_requires_unrestricted(self, h2_rhf) -> None:
        """A spin-resolved contribution on a restricted mf is rejected."""
        with pytest.raises(ValueError, match="unrestricted"):
            select_by_ao_projection(
                mf=h2_rhf,
                ao_labels=["H 1s"],
                projected_basis="minao",
                contribution={"alpha": 0.0, "beta": 0.5},
                contribution_type="r",
            )


class TestSelectByAoProjectionReturnContributions:
    """``return_contributions=True`` exposes the heatmap input (issue #68)."""

    def test_default_return_unchanged(self, h2_rhf) -> None:
        result = select_by_ao_projection(
            mf=h2_rhf, ao_labels=["H 1s"], projected_basis="minao"
        )
        assert isinstance(result, np.ndarray)

    def test_restricted_returns_indices_and_contributions(self, h2_rhf) -> None:
        indices, contributions = select_by_ao_projection(
            mf=h2_rhf,
            ao_labels=["H 1s", ["H 1s"]],
            projected_basis="minao",
            contribution=0.0,
            contribution_type="a",
            return_contributions=True,
        )
        plain = select_by_ao_projection(
            mf=h2_rhf,
            ao_labels=["H 1s", ["H 1s"]],
            projected_basis="minao",
            contribution=0.0,
            contribution_type="a",
        )
        assert indices.tolist() == plain.tolist()
        # One entry per AO group, each covering the whole search space.
        n_mo = h2_rhf.mo_coeff.shape[1]
        assert set(contributions) == {0, 1}
        for weights in contributions.values():
            assert [idx for idx, _ in weights] == list(range(n_mo))

    def test_contributions_feed_plot_standalone(self, h2_rhf, tmp_path) -> None:
        """The returned dict plugs straight into ``plot_ao_contribution``."""
        _, contributions = select_by_ao_projection(
            mf=h2_rhf,
            ao_labels=["H 1s"],
            projected_basis="minao",
            return_contributions=True,
        )
        plot_ao_contribution(
            ao_contributions=contributions, folder=str(tmp_path), filename="ao"
        )
        assert (tmp_path / "ao_heatmap.pdf").exists()

    def test_unrestricted_returns_per_channel_contributions(self, c_sto3g_uhf) -> None:
        selection, contributions = select_by_ao_projection(
            mf=c_sto3g_uhf,
            ao_labels=["C 2p"],
            projected_basis="minao",
            contribution=0.0,
            contribution_type="a",
            return_contributions=True,
        )
        assert set(selection) == {"alpha", "beta"}
        assert set(contributions) == {"alpha", "beta"}
        for channel_contributions in contributions.values():
            assert set(channel_contributions) == {0}

    def test_empty_channel_gives_empty_contributions(self, c_sto3g_uhf) -> None:
        _, contributions = select_by_ao_projection(
            mf=c_sto3g_uhf,
            ao_labels=["C 2p"],
            projected_basis="minao",
            restricted_space={"alpha": ([], []), "beta": ([(2, 0.0)], [])},
            return_contributions=True,
        )
        assert contributions["alpha"] == {}
        assert [idx for idx, _ in contributions["beta"][0]] == [2]
