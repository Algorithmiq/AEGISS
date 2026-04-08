"""Tests for aegiss.utils.orbital_indices."""
from __future__ import annotations

import numpy as np
import pytest
from aegiss.utils.orbital_indices import (
    combine_candidate_entropies,
    get_orbital_spaces,
    get_subspaces_from_cas,
)


# ---------------------------------------------------------------------------
# combine_candidate_entropies
# ---------------------------------------------------------------------------

class TestCombineCandidateEntropies:
    def test_basic_merge_sorted_by_index(self):
        occ = [(0, 0.8), (2, 0.5)]
        virt = [(1, 0.6), (3, 0.4)]
        result = combine_candidate_entropies(occ, virt)
        assert result == [(0, 0.8), (1, 0.6), (2, 0.5), (3, 0.4)]

    def test_empty_occ(self):
        result = combine_candidate_entropies([], [(5, 0.3), (1, 0.7)])
        assert result == [(1, 0.7), (5, 0.3)]

    def test_empty_virt(self):
        result = combine_candidate_entropies([(2, 0.9)], [])
        assert result == [(2, 0.9)]

    def test_both_empty(self):
        assert combine_candidate_entropies([], []) == []

    def test_preserves_entropy_values(self):
        occ = [(0, 0.123)]
        virt = [(3, 0.456)]
        result = combine_candidate_entropies(occ, virt)
        assert result[0][1] == pytest.approx(0.123)
        assert result[1][1] == pytest.approx(0.456)


# ---------------------------------------------------------------------------
# get_orbital_spaces
# ---------------------------------------------------------------------------

class TestGetOrbitalSpaces:
    """
    Fixture baseline: mo_occ = [2,2,2,2,2,0,0]
    MOs 0-4 occupied, 5-6 virtual.
    """

    def test_partition_with_mixed_orbital_list(self, mock_mf_7mo):
        # orbital_list covers some occ and some virt
        frozen_occ, active_occ, active_virt, frozen_virt = get_orbital_spaces(
            mock_mf_7mo, [1, 2, 5]
        )
        np.testing.assert_array_equal(frozen_occ, [0, 3, 4])
        np.testing.assert_array_equal(active_occ, [1, 2])
        np.testing.assert_array_equal(active_virt, [5])
        np.testing.assert_array_equal(frozen_virt, [6])

    def test_all_occ_in_list(self, mock_mf_7mo):
        frozen_occ, active_occ, active_virt, frozen_virt = get_orbital_spaces(
            mock_mf_7mo, [0, 1, 2, 3, 4]
        )
        assert len(frozen_occ) == 0
        np.testing.assert_array_equal(active_occ, [0, 1, 2, 3, 4])
        assert len(active_virt) == 0
        np.testing.assert_array_equal(frozen_virt, [5, 6])

    def test_only_virt_in_list(self, mock_mf_7mo):
        frozen_occ, active_occ, active_virt, frozen_virt = get_orbital_spaces(
            mock_mf_7mo, [5, 6]
        )
        np.testing.assert_array_equal(frozen_occ, [0, 1, 2, 3, 4])
        assert len(active_occ) == 0
        np.testing.assert_array_equal(active_virt, [5, 6])
        assert len(frozen_virt) == 0

    def test_empty_orbital_list_uses_all(self, mock_mf_7mo):
        # Empty list → all orbitals selected (no frozen)
        frozen_occ, active_occ, active_virt, frozen_virt = get_orbital_spaces(
            mock_mf_7mo, []
        )
        assert len(frozen_occ) == 0
        np.testing.assert_array_equal(active_occ, [0, 1, 2, 3, 4])
        np.testing.assert_array_equal(active_virt, [5, 6])
        assert len(frozen_virt) == 0

    def test_unique_virt_space_returns_none_active_virt(self, mock_mf_7mo):
        # unique_virt_space=True: active_virt is None, frozen_virt contains all virtuals
        frozen_occ, active_occ, active_virt, all_virt = get_orbital_spaces(
            mock_mf_7mo, [1, 2], unique_virt_space=True
        )
        assert active_virt is None
        np.testing.assert_array_equal(all_virt, [5, 6])
        np.testing.assert_array_equal(active_occ, [1, 2])
        np.testing.assert_array_equal(frozen_occ, [0, 3, 4])

    def test_return_types_are_arrays(self, mock_mf_7mo):
        result = get_orbital_spaces(mock_mf_7mo, [1, 5])
        for arr in result:
            if arr is not None:
                assert isinstance(arr, np.ndarray)


# ---------------------------------------------------------------------------
# get_subspaces_from_cas
# ---------------------------------------------------------------------------

class TestGetSubspacesFromCas:
    """
    Fixture baseline: 7 MOs, 5 occ (indices 0-4), 2 virt (indices 5-6).
    neletot = 10.
    """

    def test_standard_cas_selection(self, mock_mf_7mo):
        # ncas=4, nelecas=6 → n_active_occ=3, n_active_virt=1
        # active_occ = arange(5-3, 5) = [2,3,4]; active_virt = arange(5,6) = [5]
        active_occ, active_virt = get_subspaces_from_cas(mock_mf_7mo, ncas=4, nelecas=6)
        np.testing.assert_array_equal(active_occ, [2, 3, 4])
        np.testing.assert_array_equal(active_virt, [5])

    def test_full_cas(self, mock_mf_7mo):
        # ncas=7, nelecas=10 → n_active_occ=5, n_active_virt=2
        active_occ, active_virt = get_subspaces_from_cas(mock_mf_7mo, ncas=7, nelecas=10)
        np.testing.assert_array_equal(active_occ, [0, 1, 2, 3, 4])
        np.testing.assert_array_equal(active_virt, [5, 6])

    def test_one_active_occ_one_active_virt(self, mock_mf_7mo):
        # ncas=2, nelecas=2 → n_active_occ=1, n_active_virt=1
        active_occ, active_virt = get_subspaces_from_cas(mock_mf_7mo, ncas=2, nelecas=2)
        np.testing.assert_array_equal(active_occ, [4])
        np.testing.assert_array_equal(active_virt, [5])

    def test_ncas_exceeds_nmos_raises(self, mock_mf_7mo):
        with pytest.raises(AssertionError):
            get_subspaces_from_cas(mock_mf_7mo, ncas=8, nelecas=6)

    def test_negative_nelecas_raises(self, mock_mf_7mo):
        with pytest.raises(AssertionError):
            get_subspaces_from_cas(mock_mf_7mo, ncas=4, nelecas=-2)

    def test_nelecas_exceeds_neletot_raises(self, mock_mf_7mo):
        with pytest.raises(AssertionError):
            get_subspaces_from_cas(mock_mf_7mo, ncas=7, nelecas=12)

    def test_active_occ_contiguous_from_homo(self, mock_mf_7mo):
        # Active occ must be the LAST n_active_occ occupied MOs (HOMO and below)
        active_occ, _ = get_subspaces_from_cas(mock_mf_7mo, ncas=3, nelecas=4)
        # n_active_occ=2 → [3,4]
        np.testing.assert_array_equal(active_occ, [3, 4])

    def test_active_virt_contiguous_from_lumo(self, mock_mf_7mo):
        # Active virt must be the FIRST n_active_virt virtual MOs (LUMO and above)
        _, active_virt = get_subspaces_from_cas(mock_mf_7mo, ncas=3, nelecas=2)
        # n_active_occ=1, n_active_virt=2 → [5,6]
        np.testing.assert_array_equal(active_virt, [5, 6])
