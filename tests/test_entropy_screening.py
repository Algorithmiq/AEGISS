"""Tests for aegiss.active_space.entropy_screening."""
from __future__ import annotations

import numpy as np
import pytest
from aegiss.active_space.entropy_screening import select_by_entropy


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _all_indices(occ_selected, virt_selected):
    return [idx for idx, _ in occ_selected] + [idx for idx, _ in virt_selected]


def _all_values(occ_selected, virt_selected):
    return [v for _, v in occ_selected] + [v for _, v in virt_selected]


# ---------------------------------------------------------------------------
# Relative threshold ('r')
# ---------------------------------------------------------------------------

class TestRelativeThreshold:
    """fraction_type='r': threshold applied to entropy / max_entropy."""

    def test_basic_selection(self):
        # soe1=[0.8, 0.6, 0.4, 0.1] → normalised=[1.0, 0.75, 0.5, 0.125]
        # fraction=0.5 keeps indices 0,1,2 (values 1.0, 0.75, 0.5 >= 0.5)
        occ = np.array([0, 1])
        virt = np.array([2, 3])
        soe1 = np.array([0.8, 0.6, 0.4, 0.1])

        occ_sel, virt_sel = select_by_entropy(occ, virt, soe1, fraction=0.5, fraction_type="r")

        assert _all_indices(occ_sel, virt_sel) == [0, 1, 2]

    def test_occ_virt_split(self):
        occ = np.array([0, 1])
        virt = np.array([2, 3])
        soe1 = np.array([0.8, 0.6, 0.4, 0.1])

        occ_sel, virt_sel = select_by_entropy(occ, virt, soe1, fraction=0.5, fraction_type="r")

        assert [idx for idx, _ in occ_sel] == [0, 1]
        assert [idx for idx, _ in virt_sel] == [2]

    def test_returned_values_are_normalised(self):
        # With fraction_type='r', stored values are entropy / max_entropy
        occ = np.array([0])
        virt = np.array([1])
        soe1 = np.array([0.8, 0.4])

        occ_sel, virt_sel = select_by_entropy(occ, virt, soe1, fraction=0.4, fraction_type="r")

        # normalised: [1.0, 0.5]; both >= 0.4 → both selected
        all_vals = _all_values(occ_sel, virt_sel)
        assert len(all_vals) == 2
        np.testing.assert_allclose(sorted(all_vals, reverse=True), [1.0, 0.5])

    def test_all_selected_when_below_threshold(self):
        # fraction=0.1 → all normalised values >= 0.1
        occ = np.array([0, 1])
        virt = np.array([2, 3])
        soe1 = np.array([0.8, 0.6, 0.4, 0.2])

        occ_sel, virt_sel = select_by_entropy(occ, virt, soe1, fraction=0.1, fraction_type="r")

        assert len(_all_indices(occ_sel, virt_sel)) == 4

    def test_none_selected_when_all_below_threshold(self):
        # fraction=1.1 → max normalised value is 1.0 < 1.1, all cut
        occ = np.array([0, 1])
        virt = np.array([2])
        soe1 = np.array([0.8, 0.6, 0.4])

        occ_sel, virt_sel = select_by_entropy(occ, virt, soe1, fraction=1.1, fraction_type="r")

        assert occ_sel == []
        assert virt_sel == []

    def test_unsorted_soe1_sorted_by_entropy(self):
        # soe1 in non-descending order; result must still rank highest first
        occ = np.array([0, 1, 2])
        virt = np.array([3, 4, 5])
        soe1 = np.array([0.3, 0.9, 0.1, 0.7, 0.5, 0.2])
        # max=0.9; normalised=[0.333,1.0,0.111,0.778,0.556,0.222]
        # sorted desc: [1.0,0.778,0.556,0.333,0.222,0.111] → orbitals [1,3,4,0,5,2]
        # fraction=0.4 → first below is 0.333 at k=3; keep [1,3,4]

        occ_sel, virt_sel = select_by_entropy(occ, virt, soe1, fraction=0.4, fraction_type="r")

        assert set(_all_indices(occ_sel, virt_sel)) == {1, 3, 4}
        assert [idx for idx, _ in occ_sel] == [1]
        assert {idx for idx, _ in virt_sel} == {3, 4}


# ---------------------------------------------------------------------------
# Absolute threshold ('a')
# ---------------------------------------------------------------------------

class TestAbsoluteThreshold:
    """fraction_type='a': threshold applied directly to entropy values."""

    def test_basic_selection(self):
        # soe1=[0.8, 0.6, 0.4, 0.1]; fraction=0.5 → first below is 0.4 at k=2
        occ = np.array([0, 1])
        virt = np.array([2, 3])
        soe1 = np.array([0.8, 0.6, 0.4, 0.1])

        occ_sel, virt_sel = select_by_entropy(occ, virt, soe1, fraction=0.5, fraction_type="a")

        assert _all_indices(occ_sel, virt_sel) == [0, 1]
        assert virt_sel == []

    def test_returned_values_are_raw_entropy(self):
        occ = np.array([0])
        virt = np.array([1])
        soe1 = np.array([0.8, 0.4])

        occ_sel, virt_sel = select_by_entropy(occ, virt, soe1, fraction=0.5, fraction_type="a")

        # only orbital 0 (entropy=0.8) passes; value stored must be 0.8
        assert len(occ_sel) == 1
        np.testing.assert_allclose(occ_sel[0][1], 0.8)

    def test_all_selected(self):
        occ = np.array([0, 1])
        virt = np.array([2])
        soe1 = np.array([0.9, 0.8, 0.7])

        occ_sel, virt_sel = select_by_entropy(occ, virt, soe1, fraction=0.5, fraction_type="a")

        assert len(_all_indices(occ_sel, virt_sel)) == 3

    def test_none_selected(self):
        # fraction=0.95 is above all values
        occ = np.array([0])
        virt = np.array([1, 2])
        soe1 = np.array([0.8, 0.6, 0.4])

        occ_sel, virt_sel = select_by_entropy(occ, virt, soe1, fraction=0.95, fraction_type="a")

        assert occ_sel == []
        assert virt_sel == []

    def test_boundary_value_is_included(self):
        # entropy value exactly equal to fraction → NOT below → included
        occ = np.array([0])
        virt = np.array([1])
        soe1 = np.array([0.5, 0.3])

        occ_sel, virt_sel = select_by_entropy(occ, virt, soe1, fraction=0.5, fraction_type="a")

        assert [idx for idx, _ in occ_sel] == [0]

    def test_only_virt_selected(self):
        # occ orbitals all below threshold; virt above
        occ = np.array([0, 1])
        virt = np.array([2, 3])
        soe1 = np.array([0.1, 0.2, 0.9, 0.8])  # virt orbs have high entropy

        occ_sel, virt_sel = select_by_entropy(occ, virt, soe1, fraction=0.5, fraction_type="a")

        assert occ_sel == []
        assert {idx for idx, _ in virt_sel} == {2, 3}
