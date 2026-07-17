"""Tests for aegiss.utils.orbital_indices."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from aegiss.utils.orbital_indices import (
    combine_candidate_entropies,
    get_orbital_spaces,
    get_subspaces_from_cas,
    resolve_nelecas,
    valence_occupied_orbitals,
)


class TestResolveNelecas:
    def test_even_int(self) -> None:
        assert resolve_nelecas(10) == (5, 5)

    def test_odd_int_high_spin(self) -> None:
        assert resolve_nelecas(5) == (3, 2)

    def test_odd_int_one_electron(self) -> None:
        assert resolve_nelecas(1) == (1, 0)

    def test_tuple_passthrough(self) -> None:
        assert resolve_nelecas((3, 2)) == (3, 2)

    def test_tuple_closed_shell(self) -> None:
        assert resolve_nelecas((5, 5)) == (5, 5)


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
        assert len(frozen_occ) == 2  # 0,1
        assert len(active_occ) == 2  # 2,3
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

    def test_unique_virt_space_active_virt_is_empty(
        self, mock_mf_7mo: MagicMock
    ) -> None:
        _, _, active_virt, _ = get_orbital_spaces(
            mock_mf_7mo, [2, 3], unique_virt_space=True
        )
        assert len(active_virt) == 0

    def test_unique_virt_space_frozen_virt_is_all_virt(
        self, mock_mf_7mo: MagicMock
    ) -> None:
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


class TestValenceOccupiedOrbitals:
    def test_h2o_excludes_core(self, h2o_rhf: Any) -> None:
        """H2O/sto-3g: 5 occupied, O contributes 1 core 1s → 4 valence."""
        assert valence_occupied_orbitals(h2o_rhf) == [1, 2, 3, 4]

    def test_h2o_keeps_all_when_core_kept(self, h2o_rhf: Any) -> None:
        assert valence_occupied_orbitals(h2o_rhf, exclude_core=False) == [0, 1, 2, 3, 4]

    def test_h2_has_no_core(self, h2_rhf: Any) -> None:
        """Pure-hydrogen system has zero chemical core → nothing dropped."""
        assert valence_occupied_orbitals(h2_rhf) == valence_occupied_orbitals(
            h2_rhf, exclude_core=False
        )

    def test_returns_plain_python_ints(self, h2o_rhf: Any) -> None:
        result = valence_occupied_orbitals(h2o_rhf)
        assert all(type(i) is int for i in result)


class TestGetSubspacesFromCas:
    def test_basic_cas(self, mock_mf_7mo: MagicMock) -> None:
        """ncas=4, nelecas=4 → 2 active occ + 2 active virt."""
        mock_mf_7mo.mol.nao = 7
        active_occ, active_virt = get_subspaces_from_cas(mock_mf_7mo, ncas=4, nelecas=4)
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
        with pytest.raises(ValueError, match="nalpha >= nbeta >= 0"):
            get_subspaces_from_cas(mock_mf_7mo, ncas=4, nelecas=-2)

    def test_open_shell_tuple_active_occ_count(self, mock_mf_7mo: MagicMock) -> None:
        """nelecas=(3, 2) → nalpha=3 active occupied orbitals."""
        active_occ, _ = get_subspaces_from_cas(mock_mf_7mo, ncas=5, nelecas=(3, 2))
        assert len(active_occ) == 3

    def test_open_shell_tuple_active_occ_indices(self, mock_mf_7mo: MagicMock) -> None:
        active_occ, _ = get_subspaces_from_cas(mock_mf_7mo, ncas=5, nelecas=(3, 2))
        np.testing.assert_array_equal(active_occ, [1, 2, 3])

    def test_open_shell_tuple_active_virt_indices(self, mock_mf_7mo: MagicMock) -> None:
        _, active_virt = get_subspaces_from_cas(mock_mf_7mo, ncas=5, nelecas=(3, 2))
        np.testing.assert_array_equal(active_virt, [4, 5])

    def test_invalid_tuple_nalpha_lt_nbeta_raises(self, mock_mf_7mo: MagicMock) -> None:
        with pytest.raises(ValueError, match="nalpha >= nbeta >= 0"):
            get_subspaces_from_cas(mock_mf_7mo, ncas=4, nelecas=(2, 4))

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
        active_occ, active_virt = get_subspaces_from_cas(mock_mf_7mo, ncas=4, nelecas=4)
        assert isinstance(active_occ, np.ndarray)
        assert isinstance(active_virt, np.ndarray)

    # --- spin-aware placement (restricted + unrestricted references) ----------

    @staticmethod
    def _mock_mf(mo_occ, nmos: int) -> MagicMock:
        """Mock mf with the given occupation (1-D restricted or 2-D UHF)."""
        mf = MagicMock()
        occ = np.asarray(mo_occ, dtype=float)
        mf.mo_occ = occ
        mf.mo_coeff = (
            np.zeros((2, nmos, nmos)) if occ.ndim == 2 else np.zeros((nmos, nmos))
        )
        return mf

    def test_restricted_window_placed_by_total_active_electrons(self) -> None:
        # closed-shell reference (12 e in 10 MOs); window must sit at
        # ncore = (12 - 6)/2 = 3 -> active [3..8], split at HOMO (6).
        mf = self._mock_mf([2] * 6 + [0] * 4, nmos=10)
        active_occ, active_virt = get_subspaces_from_cas(mf, ncas=6, nelecas=(3, 3))
        np.testing.assert_array_equal(active_occ, [3, 4, 5])
        np.testing.assert_array_equal(active_virt, [6, 7, 8])

    def test_restricted_open_shell_target_does_not_shift_window(self) -> None:
        # Same closed-shell reference, but an open-shell (triplet) DMRG target.
        # The window must NOT shift with the nalpha/nbeta split: still [3..8],
        # split at the RHF HOMO -> identical partition to the (3,3) case.
        mf = self._mock_mf([2] * 6 + [0] * 4, nmos=10)
        active_occ, active_virt = get_subspaces_from_cas(mf, ncas=6, nelecas=(4, 2))
        np.testing.assert_array_equal(active_occ, [3, 4, 5])
        np.testing.assert_array_equal(active_virt, [6, 7, 8])

    def test_uhf_triplet_places_somos_in_active_occ(self) -> None:
        # UHF triplet reference (Na=7, Nb=5, 2S=2) in 10 MOs. occ_per_orb =
        # [2,2,2,2,2,1,1,0,0,0]; window ncore=(12-6)/2=3 -> [3..8]; SOMOs (5,6)
        # are occupied so they land in active_occ.
        mo_occ = np.array([[1] * 7 + [0] * 3, [1] * 5 + [0] * 5], dtype=float)
        mf = self._mock_mf(mo_occ, nmos=10)
        active_occ, active_virt = get_subspaces_from_cas(mf, ncas=6, nelecas=(4, 2))
        np.testing.assert_array_equal(active_occ, [3, 4, 5, 6])
        np.testing.assert_array_equal(active_virt, [7, 8])

    def test_uhf_tuple_mo_coeff_and_occ(self) -> None:
        # pyscf UHF exposes mo_coeff/mo_occ as *tuples* of per-spin blocks, not
        # a stacked 3-D array; the partition must be identical to the array form.
        mf = MagicMock()
        mf.mo_occ = (np.array([1.0] * 7 + [0.0] * 3), np.array([1.0] * 5 + [0.0] * 5))
        mf.mo_coeff = (np.zeros((10, 10)), np.zeros((10, 10)))
        active_occ, active_virt = get_subspaces_from_cas(mf, ncas=6, nelecas=(4, 2))
        np.testing.assert_array_equal(active_occ, [3, 4, 5, 6])
        np.testing.assert_array_equal(active_virt, [7, 8])

    def test_uhf_closed_core_violation_raises(self) -> None:
        # UHF reference with 2S=2 but a closed-shell (3,3) active target:
        # reference spin (2) != active spin (0) -> no single CAS window.
        mo_occ = np.array([[1] * 7 + [0] * 3, [1] * 5 + [0] * 5], dtype=float)
        mf = self._mock_mf(mo_occ, nmos=10)
        with pytest.raises(ValueError, match="active spin"):
            get_subspaces_from_cas(mf, ncas=6, nelecas=(3, 3))
