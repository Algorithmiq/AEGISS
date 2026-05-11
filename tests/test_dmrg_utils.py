"""Tests for aegiss.utils.dmrg_utils."""

from __future__ import annotations

import numpy as np
import pytest

from aegiss.utils.dmrg_utils import _default_dmrg_schedule, compute_qmi


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


class TestComputeQmi:
    def test_returns_three_arrays(self) -> None:
        soe1 = np.array([0.1, 0.5, 0.4, 0.2])
        soe2 = np.outer(soe1, soe1) * 2.0

        driver = type("FakeDriver", (), {})()

        def _get_orbital_entropies(_ket: object, orb_type: int) -> np.ndarray:
            return soe1 if orb_type == 1 else soe2

        driver.get_orbital_entropies = _get_orbital_entropies  # type: ignore[attr-defined]

        ordm1, ordm2, minfo = compute_qmi(driver, ket=None)

        assert ordm1.shape == (4,)
        assert ordm2.shape == (4, 4)
        assert minfo.shape == (4, 4)

    def test_minfo_diagonal_is_zero(self) -> None:
        soe1 = np.array([0.1, 0.5, 0.4, 0.2])
        soe2 = np.outer(soe1, soe1) * 2.0

        driver = type("FakeDriver", (), {})()

        def _get_orbital_entropies(_ket: object, orb_type: int) -> np.ndarray:
            return soe1 if orb_type == 1 else soe2

        driver.get_orbital_entropies = _get_orbital_entropies  # type: ignore[attr-defined]

        _, _, minfo = compute_qmi(driver, ket=None)
        np.testing.assert_allclose(np.diag(minfo), 0.0, atol=1e-12)

    def test_minfo_non_negative(self) -> None:
        """QMI values should be >= 0 for physically sensible inputs."""
        n = 3
        soe1 = np.array([0.3, 0.6, 0.1])
        # soe2[i,j] <= soe1[i] + soe1[j] ensures non-negative QMI
        soe2 = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                soe2[i, j] = soe1[i] + soe1[j]
        np.fill_diagonal(soe2, 0.0)

        driver = type("FakeDriver", (), {})()

        def _get_orbital_entropies(_ket: object, orb_type: int) -> np.ndarray:
            return soe1 if orb_type == 1 else soe2

        driver.get_orbital_entropies = _get_orbital_entropies  # type: ignore[attr-defined]

        _, _, minfo = compute_qmi(driver, ket=None)
        assert np.all(minfo >= -1e-14)

    def test_minfo_symmetric(self) -> None:
        soe1 = np.array([0.1, 0.5, 0.4, 0.2])
        soe2 = np.outer(soe1, soe1) * 1.5

        driver = type("FakeDriver", (), {})()
        driver.get_orbital_entropies = (  # type: ignore[attr-defined]
            lambda _ket, orb_type: soe1 if orb_type == 1 else soe2
        )

        _, _, minfo = compute_qmi(driver, ket=None)
        np.testing.assert_allclose(minfo, minfo.T, atol=1e-14)

    def test_minfo_single_orbital_is_zero(self) -> None:
        soe1 = np.array([0.5])
        soe2 = np.array([[0.0]])

        driver = type("FakeDriver", (), {})()
        driver.get_orbital_entropies = (  # type: ignore[attr-defined]
            lambda _ket, orb_type: soe1 if orb_type == 1 else soe2
        )

        _, _, minfo = compute_qmi(driver, ket=None)
        np.testing.assert_allclose(minfo, [[0.0]], atol=1e-14)

    def test_minfo_exact_values_n2(self) -> None:
        # minfo[i,j] = 0.5 * (soe1[i] + soe1[j] - soe2[i,j]) for i != j
        soe1 = np.array([0.2, 0.4])
        soe2 = np.array([[0.0, 0.3], [0.3, 0.0]])

        driver = type("FakeDriver", (), {})()
        driver.get_orbital_entropies = (  # type: ignore[attr-defined]
            lambda _ket, orb_type: soe1 if orb_type == 1 else soe2
        )

        _, _, minfo = compute_qmi(driver, ket=None)
        expected = np.array([[0.0, 0.15], [0.15, 0.0]])
        np.testing.assert_allclose(minfo, expected, atol=1e-14)
