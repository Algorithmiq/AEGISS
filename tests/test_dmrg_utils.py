"""Tests for aegiss.utils.dmrg_utils."""

from __future__ import annotations

import numpy as np
import pytest

from aegiss.utils.dmrg_utils import _default_dmrg_schedule


class TestDefaultDmrgSchedule:
    def test_returns_three_lists(self) -> None:
        bond_dims, noises, thrds = _default_dmrg_schedule()
        assert isinstance(bond_dims, list)
        assert isinstance(noises, list)
        assert isinstance(thrds, list)

    def test_bond_dims_length(self) -> None:
        bond_dims, _, _ = _default_dmrg_schedule()
        assert len(bond_dims) == 16

    def test_noises_length(self) -> None:
        _, noises, _ = _default_dmrg_schedule()
        assert len(noises) == 16

    def test_thrds_length(self) -> None:
        _, _, thrds = _default_dmrg_schedule()
        assert len(thrds) == 16

    def test_bond_dims_monotone(self) -> None:
        bond_dims, _, _ = _default_dmrg_schedule(init_bond_dim=10)
        assert bond_dims[0] <= bond_dims[-1]

    def test_bond_dims_scale_with_init(self) -> None:
        bond_dims_10, _, _ = _default_dmrg_schedule(init_bond_dim=10)
        bond_dims_20, _, _ = _default_dmrg_schedule(init_bond_dim=20)
        assert bond_dims_20[0] == 2 * bond_dims_10[0]
        assert bond_dims_20[-1] == 2 * bond_dims_10[-1]

    def test_noises_non_negative(self) -> None:
        _, noises, _ = _default_dmrg_schedule()
        assert all(n >= 0.0 for n in noises)

    def test_thrds_positive(self) -> None:
        _, _, thrds = _default_dmrg_schedule()
        assert all(t > 0.0 for t in thrds)

    def test_invalid_init_bond_dim_raises(self) -> None:
        with pytest.raises(
            ValueError, match="init_bond_dim must be a positive integer"
        ):
            _default_dmrg_schedule(init_bond_dim=0)

    def test_negative_init_bond_dim_raises(self) -> None:
        with pytest.raises(
            ValueError, match="init_bond_dim must be a positive integer"
        ):
            _default_dmrg_schedule(init_bond_dim=-5)

    def test_all_lists_same_length(self) -> None:
        bond_dims, noises, thrds = _default_dmrg_schedule()
        assert len(bond_dims) == len(noises) == len(thrds)

    def test_bond_dims_first_equals_init(self) -> None:
        bond_dims, _, _ = _default_dmrg_schedule(init_bond_dim=15)
        assert bond_dims[0] == 15

    def test_bond_dims_last_is_6x_init(self) -> None:
        bond_dims, _, _ = _default_dmrg_schedule(init_bond_dim=10)
        assert bond_dims[-1] == 60

    def test_noises_non_increasing(self) -> None:
        _, noises, _ = _default_dmrg_schedule()
        assert all(noises[i] >= noises[i + 1] for i in range(len(noises) - 1))

    def test_thrds_non_increasing(self) -> None:
        _, _, thrds = _default_dmrg_schedule()
        assert all(thrds[i] >= thrds[i + 1] for i in range(len(thrds) - 1))

    def test_noises_end_with_zeros(self) -> None:
        _, noises, _ = _default_dmrg_schedule()
        np.testing.assert_allclose(noises[-2:], 0.0, atol=1e-15)


