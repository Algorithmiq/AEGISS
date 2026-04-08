"""Tests for aegiss.utils.dmrg_utils."""
from __future__ import annotations

import numpy as np
import pytest
from aegiss.utils.dmrg_utils import _default_dmrg_schedule


class TestDefaultDmrgSchedule:

    def test_returns_three_lists(self):
        result = _default_dmrg_schedule(20)
        assert len(result) == 3

    def test_bond_dims_length(self):
        bond_dims, _, _ = _default_dmrg_schedule(20)
        assert len(bond_dims) == 16

    def test_noises_length(self):
        _, noises, _ = _default_dmrg_schedule(20)
        assert len(noises) == 14

    def test_thrds_length(self):
        _, _, thrds = _default_dmrg_schedule(20)
        assert len(thrds) == 14

    def test_bond_dims_scaling_with_init(self):
        bond_dims, _, _ = _default_dmrg_schedule(10)
        # multipliers: [1,1,1,1, 2,2,2,2, 4,4,4,4, 6,6,6,6]
        assert bond_dims[0] == 10
        assert bond_dims[4] == 20
        assert bond_dims[8] == 40
        assert bond_dims[12] == 60

    def test_bond_dims_non_decreasing(self):
        bond_dims, _, _ = _default_dmrg_schedule(20)
        for a, b in zip(bond_dims, bond_dims[1:]):
            assert b >= a

    def test_noises_non_increasing(self):
        _, noises, _ = _default_dmrg_schedule(20)
        for a, b in zip(noises, noises[1:]):
            assert b <= a

    def test_thrds_non_increasing(self):
        _, _, thrds = _default_dmrg_schedule(20)
        for a, b in zip(thrds, thrds[1:]):
            assert b <= a

    def test_noises_end_at_zero(self):
        _, noises, _ = _default_dmrg_schedule(20)
        assert noises[-1] == 0.0

    def test_zero_init_bond_dim_raises(self):
        with pytest.raises(ValueError):
            _default_dmrg_schedule(0)

    def test_negative_init_bond_dim_raises(self):
        with pytest.raises(ValueError):
            _default_dmrg_schedule(-5)

    def test_large_init_bond_dim(self):
        bond_dims, noises, thrds = _default_dmrg_schedule(500)
        assert bond_dims[0] == 500
        assert bond_dims[-1] == 3000  # 500 * 6
