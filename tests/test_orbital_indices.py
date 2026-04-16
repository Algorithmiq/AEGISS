"""Tests for aegiss.utils.orbital_indices."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from aegiss.utils.orbital_indices import (
    combine_candidate_entropies,
    get_orbital_spaces,
    get_subspaces_from_cas,
)


class TestCombineCandidateEntropies:
    def test_combines_and_sorts(self) -> None:
        occ = [(1, 0.8), (3, 0.5)]
        virt = [(5, 0.3), (2, 0.9)]
        result = combine_candidate_entropies(occ, virt)
        indices = [idx for idx, _ in result]
        assert indices == sorted(indices)

    def test_empty_inputs(self) -> None:
        result = combine_candidate_entropies([], [])
        assert result == []

    def test_only_occ(self) -> None:
        occ = [(0, 0.1), (2, 0.9)]
        result = combine_candidate_entropies(occ, [])
        assert len(result) == 2

    def test_only_virt(self) -> None:
        virt = [(4, 0.2), (6, 0.7)]
        result = combine_candidate_entropies([], virt)
        assert len(result) == 2

    def test_entropy_values_preserved(self) -> None:
        occ = [(1, 0.8)]
        virt = [(3, 0.4)]
        result = combine_candidate_entropies(occ, virt)
        assert result[0] == (1, 0.8)
        assert result[1] == (3, 0.4)


class TestGetOrbitalSpaces:
    def test_partition_counts(self, mock_mf_7mo: MagicMock) -> None:
        """4 occ (indices 0-3), 3 virt (4-6). Select [2,3,4,5]."""
        frozen_occ, active_occ, active_virt, frozen_virt = get_orbital_spaces(
            mock_mf_7mo, [2, 3, 4, 5]
        )
        assert len(frozen_occ) == 2   # 0,1
        assert len(active_occ) == 2   # 2,3
        assert len(active_virt) == 2  # 4,5
        assert len(frozen_virt) == 1  # 6

    def test_frozen_occ_values(self, mock_mf_7mo: MagicMock) -> None:
        frozen_occ, _, _, _ = get_orbital_spaces(mock_mf_7mo, [2, 3, 4, 5])
        np.testing.assert_array_equal(frozen_occ, [0, 1])

    def test_active_occ_values(self, mock_mf_7mo: MagicMock) -> None:
        _, active_occ, _, _ = get_orbital_spaces(mock_mf_7mo, [2, 3, 4, 5])
        np.testing.assert_array_equal(active_occ, [2, 3])

    def test_active_virt_values(self, mock_mf_7mo: MagicMock) -> None:
        _, _, active_virt, _ = get_orbital_spaces(mock_mf_7mo, [2, 3, 4, 5])
        np.testing.assert_array_equal(active_virt, [4, 5])

    def test_frozen_virt_values(self, mock_mf_7mo: MagicMock) -> None:
        _, _, _, frozen_virt = get_orbital_spaces(mock_mf_7mo, [2, 3, 4, 5])
        np.testing.assert_array_equal(frozen_virt, [6])

    def test_unique_virt_space_active_virt_is_empty(self, mock_mf_7mo: MagicMock) -> None:
        _, _, active_virt, _ = get_orbital_spaces(
            mock_mf_7mo, [2, 3], unique_virt_space=True
        )
        assert len(active_virt) == 0

    def test_unique_virt_space_frozen_virt_is_all_virt(self, mock_mf_7mo: MagicMock) -> None:
        _, _, _, frozen_virt = get_orbital_spaces(
            mock_mf_7mo, [2, 3], unique_virt_space=True
        )
        np.testing.assert_array_equal(frozen_virt, [4, 5, 6])

    def test_empty_orbital_list_selects_all(self, mock_mf_7mo: MagicMock) -> None:
        """Empty orbital_list triggers fallback: all MOs selected."""
        frozen_occ, active_occ, active_virt, frozen_virt = get_orbital_spaces(
            mock_mf_7mo, []
        )
        assert len(frozen_occ) == 0
        assert len(active_occ) == 4
        assert len(active_virt) == 3
        assert len(frozen_virt) == 0

    def test_returns_ndarrays(self, mock_mf_7mo: MagicMock) -> None:
        result = get_orbital_spaces(mock_mf_7mo, [1, 2, 4])
        for arr in result:
            assert isinstance(arr, np.ndarray)


class TestGetSubspacesFromCas:
    def test_basic_cas(self, mock_mf_7mo: MagicMock) -> None:
        """ncas=4, nelecas=4 → 2 active occ + 2 active virt."""
        mock_mf_7mo.mol.nao = 7
        active_occ, active_virt = get_subspaces_from_cas(
            mock_mf_7mo, ncas=4, nelecas=4
        )
        assert len(active_occ) == 2
        assert len(active_virt) == 2

    def test_active_occ_are_homo_region(self, mock_mf_7mo: MagicMock) -> None:
        """Active occ should be the last 2 occupied (HOMO-1, HOMO)."""
        mock_mf_7mo.mol.nao = 7
        active_occ, _ = get_subspaces_from_cas(mock_mf_7mo, ncas=4, nelecas=4)
        np.testing.assert_array_equal(active_occ, [2, 3])

    def test_active_virt_are_lumo_region(self, mock_mf_7mo: MagicMock) -> None:
        mock_mf_7mo.mol.nao = 7
        _, active_virt = get_subspaces_from_cas(mock_mf_7mo, ncas=4, nelecas=4)
        np.testing.assert_array_equal(active_virt, [4, 5])

    def test_ncas_exceeds_nmos_raises(self, mock_mf_7mo: MagicMock) -> None:
        mock_mf_7mo.mol.nao = 7
        with pytest.raises(ValueError, match="active orbitals cannot exceed"):
            get_subspaces_from_cas(mock_mf_7mo, ncas=10, nelecas=4)

    def test_negative_nelecas_raises(self, mock_mf_7mo: MagicMock) -> None:
        mock_mf_7mo.mol.nao = 7
        with pytest.raises(ValueError, match="cannot be negative"):
            get_subspaces_from_cas(mock_mf_7mo, ncas=4, nelecas=-2)

    def test_nelecas_exceeds_total_raises(self, mock_mf_7mo: MagicMock) -> None:
        mock_mf_7mo.mol.nao = 7
        with pytest.raises(ValueError, match="cannot exceed total"):
            get_subspaces_from_cas(mock_mf_7mo, ncas=4, nelecas=20)

    def test_not_enough_virt_raises(self, mock_mf_7mo: MagicMock) -> None:
        """ncas=10 virt needed > available virt raises ValueError."""
        mock_mf_7mo.mol.nao = 7
        with pytest.raises(ValueError, match="Not enough virtual orbitals"):
            # 4 occ, 3 virt — requesting ncas=6, nelecas=4 → 4 virt needed
            get_subspaces_from_cas(mock_mf_7mo, ncas=6, nelecas=4)

    def test_returns_ndarrays(self, mock_mf_7mo: MagicMock) -> None:
        mock_mf_7mo.mol.nao = 7
        active_occ, active_virt = get_subspaces_from_cas(
            mock_mf_7mo, ncas=4, nelecas=4
        )
        assert isinstance(active_occ, np.ndarray)
        assert isinstance(active_virt, np.ndarray)
