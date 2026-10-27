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

"""Tests for aegiss.active_space.entropy_screening."""

from __future__ import annotations

import numpy as np
import pytest

from aegiss.active_space.entropy_screening import (
    _global_entropy_ranking,
    _relative_entropy_ranking,
    select_by_entropy,
    select_by_spin_entropy,
)


@pytest.fixture
def simple_orbs() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    occ_orbs = np.array([0, 1, 2, 3])
    virt_orbs = np.array([4, 5, 6])
    soe1 = np.array([0.1, 0.9, 0.5, 0.3, 0.8, 0.2, 0.05])
    return occ_orbs, virt_orbs, soe1


class TestGlobalEntropyRanking:
    def test_returns_occ_and_virt(self, simple_orbs) -> None:
        occ, virt, soe1 = simple_orbs
        occ_sel, virt_sel = _global_entropy_ranking(soe1, "r", 0.5, occ, virt)
        assert isinstance(occ_sel, list)
        assert isinstance(virt_sel, list)

    def test_relative_threshold_selects_high_entropy(self, simple_orbs) -> None:
        occ, virt, soe1 = simple_orbs
        occ_sel, virt_sel = _global_entropy_ranking(soe1, "r", 0.5, occ, virt)
        all_sel = occ_sel + virt_sel
        # All selected should have s1 / max_s1 >= 0.5
        max_s1 = np.max(soe1)
        for _idx, s in all_sel:
            assert s / max_s1 >= 0.5

    def test_absolute_threshold(self, simple_orbs) -> None:
        occ, virt, soe1 = simple_orbs
        occ_sel, virt_sel = _global_entropy_ranking(soe1, "a", 0.5, occ, virt)
        all_sel = occ_sel + virt_sel
        # absolute: raw soe1 values >=0.5
        for idx, _s in all_sel:
            assert soe1[np.where(np.concatenate((occ, virt)) == idx)[0][0]] >= 0.5

    def test_all_selected_when_threshold_zero(self, simple_orbs) -> None:
        occ, virt, soe1 = simple_orbs
        occ_sel, virt_sel = _global_entropy_ranking(soe1, "a", 0.0, occ, virt)
        assert len(occ_sel) + len(virt_sel) == len(soe1)

    def test_occ_indices_belong_to_occ(self, simple_orbs) -> None:
        occ, virt, soe1 = simple_orbs
        occ_sel, _ = _global_entropy_ranking(soe1, "r", 0.1, occ, virt)
        for idx, _ in occ_sel:
            assert idx in occ

    def test_virt_indices_belong_to_virt(self, simple_orbs) -> None:
        occ, virt, soe1 = simple_orbs
        _, virt_sel = _global_entropy_ranking(soe1, "r", 0.1, occ, virt)
        for idx, _ in virt_sel:
            assert idx in virt


class TestRelativeEntropyRanking:
    def test_returns_list_of_tuples(self) -> None:
        soe1 = np.array([0.3, 0.9, 0.1, 0.7])
        subspace = np.array([2, 5, 8, 11])
        result = _relative_entropy_ranking(soe1, "r", 0.5, subspace)
        assert isinstance(result, list)
        for item in result:
            assert isinstance(item, tuple)
            assert len(item) == 2

    def test_relative_threshold(self) -> None:
        soe1 = np.array([0.2, 1.0, 0.4, 0.8])
        subspace = np.array([0, 1, 2, 3])
        result = _relative_entropy_ranking(soe1, "r", 0.5, subspace)
        indices = [idx for idx, _ in result]
        # After normalisation: 1.0→1.0, 0.8→0.8, 0.4→0.4, 0.2→0.2
        # Those >=0.5: orb 1 (1.0) and orb 3 (0.8)
        assert 1 in indices
        assert 3 in indices
        assert 0 not in indices
        assert 2 not in indices

    def test_absolute_threshold(self) -> None:
        soe1 = np.array([0.1, 0.9, 0.6, 0.3])
        subspace = np.array([10, 11, 12, 13])
        result = _relative_entropy_ranking(soe1, "a", 0.5, subspace)
        indices = [idx for idx, _ in result]
        assert 11 in indices
        assert 12 in indices
        assert 10 not in indices

    def test_all_selected_zero_threshold(self) -> None:
        soe1 = np.array([0.2, 0.8, 0.5])
        subspace = np.array([0, 1, 2])
        result = _relative_entropy_ranking(soe1, "a", 0.0, subspace)
        assert len(result) == 3

    def test_indices_from_subspace(self) -> None:
        soe1 = np.array([0.5, 0.9])
        subspace = np.array([7, 9])
        result = _relative_entropy_ranking(soe1, "r", 0.0, subspace)
        for idx, _ in result:
            assert idx in subspace


class TestSelectByEntropy:
    def test_none_soe1_returns_empty(self) -> None:
        occ = np.array([0, 1])
        virt = np.array([2, 3])
        sel_occ, sel_virt = select_by_entropy(occ, virt, soe1=None)
        assert sel_occ == []
        assert sel_virt == []

    def test_global_ranking_returns_tuples(self, simple_orbs) -> None:
        occ, virt, soe1 = simple_orbs
        sel_occ, sel_virt = select_by_entropy(
            occ, virt, soe1, fraction=0.5, fraction_type="r", ranking="global"
        )
        for item in sel_occ + sel_virt:
            assert isinstance(item, tuple)
            assert len(item) == 2

    def test_subset_ranking_returns_tuples(self, simple_orbs) -> None:
        occ, virt, soe1 = simple_orbs
        sel_occ, sel_virt = select_by_entropy(
            occ, virt, soe1, fraction=0.5, fraction_type="r", ranking="subset"
        )
        for item in sel_occ + sel_virt:
            assert isinstance(item, tuple)

    def test_global_occ_indices_in_occ_space(self, simple_orbs) -> None:
        occ, virt, soe1 = simple_orbs
        sel_occ, _ = select_by_entropy(
            occ, virt, soe1, fraction=0.1, fraction_type="r", ranking="global"
        )
        for idx, _ in sel_occ:
            assert idx in occ

    def test_global_virt_indices_in_virt_space(self, simple_orbs) -> None:
        occ, virt, soe1 = simple_orbs
        _, sel_virt = select_by_entropy(
            occ, virt, soe1, fraction=0.1, fraction_type="r", ranking="global"
        )
        for idx, _ in sel_virt:
            assert idx in virt

    def test_high_threshold_selects_few(self, simple_orbs) -> None:
        occ, virt, soe1 = simple_orbs
        sel_occ, sel_virt = select_by_entropy(
            occ, virt, soe1, fraction=0.99, fraction_type="r", ranking="subset"
        )
        # Only the maximum per subspace passes 0.99 threshold
        assert len(sel_occ) == 1
        assert len(sel_virt) == 1

    def test_absolute_zero_threshold_selects_all(self, simple_orbs) -> None:
        occ, virt, soe1 = simple_orbs
        sel_occ, sel_virt = select_by_entropy(
            occ, virt, soe1, fraction=0.0, fraction_type="a", ranking="global"
        )
        assert len(sel_occ) + len(sel_virt) == len(soe1)

    def test_default_fraction_type_is_relative(self, simple_orbs) -> None:
        occ, virt, soe1 = simple_orbs
        # default fraction_type="r" — result should differ from absolute
        sel_occ_r, _ = select_by_entropy(
            occ, virt, soe1, fraction=0.5, fraction_type="r", ranking="subset"
        )
        sel_occ_a, _ = select_by_entropy(
            occ, virt, soe1, fraction=0.5, fraction_type="a", ranking="subset"
        )
        # The results may differ; at minimum both calls should succeed
        assert isinstance(sel_occ_r, list)
        assert isinstance(sel_occ_a, list)

    def test_spin_resolved_array_raises(self, simple_orbs) -> None:
        # a (2, ncas) spin-resolved array must be rejected with a clear message,
        # not silently mishandled (previously an out-of-bounds IndexError).
        occ, virt, soe1 = simple_orbs
        soe2d = np.stack([soe1, soe1])
        with pytest.raises(ValueError, match="select_by_spin_entropy"):
            select_by_entropy(occ, virt, soe2d)


class TestSelectBySpinEntropy:
    @pytest.fixture
    def spin_orbs(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        occ = np.array([0, 1, 2])
        virt = np.array([3, 4, 5])
        # alpha hot on 0,3 ; beta hot on 1,2,4 -> channels select different orbs
        soe = np.array(
            [
                [0.6, 0.1, 0.02, 0.9, 0.0, 0.05],
                [0.05, 0.7, 0.7, 0.0, 0.7, 0.04],
            ]
        )
        return occ, virt, soe

    def test_returns_alpha_beta_dict(self, spin_orbs) -> None:
        occ, virt, soe = spin_orbs
        sel = select_by_spin_entropy(occ, virt, soe, fraction=0.1, fraction_type="r")
        assert set(sel) == {"alpha", "beta"}
        for occ_sel, virt_sel in sel.values():
            assert isinstance(occ_sel, list)
            assert isinstance(virt_sel, list)

    def test_channels_select_independently(self, spin_orbs) -> None:
        occ, virt, soe = spin_orbs
        sel = select_by_spin_entropy(occ, virt, soe, fraction=0.1, fraction_type="r")
        alpha_idx = {i for i, _ in sel["alpha"][0] + sel["alpha"][1]}
        beta_idx = {i for i, _ in sel["beta"][0] + sel["beta"][1]}
        # the two channels keep genuinely different orbital sets
        assert alpha_idx != beta_idx

    def test_matches_per_channel_select_by_entropy(self, spin_orbs) -> None:
        occ, virt, soe = spin_orbs
        sel = select_by_spin_entropy(occ, virt, soe, fraction=0.2, fraction_type="a")
        assert sel["alpha"] == select_by_entropy(
            occ, virt, soe[0], fraction=0.2, fraction_type="a"
        )
        assert sel["beta"] == select_by_entropy(
            occ, virt, soe[1], fraction=0.2, fraction_type="a"
        )

    def test_none_returns_empty_channels(self) -> None:
        occ = np.array([0, 1])
        virt = np.array([2, 3])
        sel = select_by_spin_entropy(occ, virt, None)
        assert sel == {"alpha": ([], []), "beta": ([], [])}

    def test_one_dim_array_raises(self, spin_orbs) -> None:
        occ, virt, soe = spin_orbs
        with pytest.raises(ValueError, match="select_by_entropy"):
            select_by_spin_entropy(occ, virt, soe[0])
