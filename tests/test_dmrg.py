"""Tests for aegiss.wavefunction.dmrg."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

pytest.importorskip("pyblock2")

from aegiss.wavefunction.dmrg import run_dmrg


@pytest.fixture
def mocked_run_dmrg():
    """Patch all heavy dependencies of run_dmrg and yield a dict of mocks."""
    fake_energy = -2.345
    soe1 = np.array([0.1, 0.2])
    soe2 = np.zeros((2, 2))
    minfo = np.zeros((2, 2))

    mock_driver = MagicMock()
    mock_driver.dmrg.return_value = fake_energy
    mock_driver.expectation.return_value = 1.0

    mock_casci = MagicMock()
    mock_casci.get_h1eff.return_value = (np.zeros((2, 2)), 0.0)
    mock_casci.get_h2eff.return_value = np.zeros((4,))

    with (
        patch(
            "aegiss.wavefunction.dmrg.DMRGDriver", return_value=mock_driver
        ) as MockDriver,
        patch(
            "aegiss.wavefunction.dmrg.mcscf.CASCI", return_value=mock_casci
        ) as MockCASCI,
        patch(
            "aegiss.wavefunction.dmrg.ao2mo.restore",
            return_value=np.zeros((2, 2, 2, 2)),
        ),
        patch("aegiss.wavefunction.dmrg.resolve_nelecas", return_value=(1, 1)),
        patch(
            "aegiss.wavefunction.dmrg.compute_qmi",
            return_value=(soe1, soe2, minfo),
        ) as mock_qmi,
    ):
        yield {
            "driver": mock_driver,
            "MockDriver": MockDriver,
            "MockCASCI": MockCASCI,
            "casci": mock_casci,
            "compute_qmi": mock_qmi,
            "energy": fake_energy,
            "soe1": soe1,
            "soe2": soe2,
            "minfo": minfo,
        }


class TestRunDmrgSetup:
    """Tests for the integral-fetching setup that runs before the DMRG sweep."""

    def test_casci_called_with_mf_ncas_nelecas(self, h2_rhf, mocked_run_dmrg) -> None:
        run_dmrg(mf=h2_rhf, ncas=2, nelecas=2, symm_type="SZ", get_qmi=False)
        mocked_run_dmrg["MockCASCI"].assert_called_once_with(h2_rhf, 2, 2)

    def test_h1eff_fetched_from_casci(self, mocked_run_dmrg) -> None:
        run_dmrg(mf=None, ncas=2, nelecas=2, symm_type="SZ", get_qmi=False)
        mocked_run_dmrg["casci"].get_h1eff.assert_called_once()

    def test_h2eff_fetched_from_casci(self, mocked_run_dmrg) -> None:
        run_dmrg(mf=None, ncas=2, nelecas=2, symm_type="SZ", get_qmi=False)
        mocked_run_dmrg["casci"].get_h2eff.assert_called_once()

    def test_initialize_system_n_sites_matches_ncas(self, mocked_run_dmrg) -> None:
        run_dmrg(mf=None, ncas=3, nelecas=2, symm_type="SZ", get_qmi=False)
        call_kwargs = (
            mocked_run_dmrg["driver"].initialize_system.call_args_list[0].kwargs
        )
        assert call_kwargs["n_sites"] == 3

    def test_initialize_system_nelec_derived_from_resolve_nelecas(
        self, mocked_run_dmrg
    ) -> None:
        # resolve_nelecas is mocked to return (1, 1) → n_elec = 1 + 1 = 2
        run_dmrg(mf=None, ncas=2, nelecas=2, symm_type="SZ", get_qmi=False)
        call_kwargs = (
            mocked_run_dmrg["driver"].initialize_system.call_args_list[0].kwargs
        )
        assert call_kwargs["n_elec"] == 2

    def test_initialize_system_spin_derived_from_resolve_nelecas(
        self, mocked_run_dmrg
    ) -> None:
        # resolve_nelecas is mocked to return (1, 1) → spin = 1 - 1 = 0
        run_dmrg(mf=None, ncas=2, nelecas=2, symm_type="SZ", get_qmi=False)
        call_kwargs = (
            mocked_run_dmrg["driver"].initialize_system.call_args_list[0].kwargs
        )
        assert call_kwargs["spin"] == 0


class TestRunDmrgSymmTypeValidation:
    def test_invalid_symm_type_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="Unsupported symmetry type"):
            run_dmrg(mf=None, ncas=4, nelecas=4, symm_type="SU3")

    def test_invalid_symm_type_message_contains_valid_options(self) -> None:
        with pytest.raises(ValueError, match="SU2"):
            run_dmrg(mf=None, ncas=4, nelecas=4, symm_type="invalid")

    def test_invalid_symm_type_is_included_in_message(self) -> None:
        with pytest.raises(ValueError, match="'bad'"):
            run_dmrg(mf=None, ncas=4, nelecas=4, symm_type="bad")


class TestRunDmrgBehavior:
    def test_energy_returned(self, mocked_run_dmrg) -> None:
        e, *_ = run_dmrg(mf=None, ncas=2, nelecas=2, symm_type="SZ", get_qmi=False)
        assert e == mocked_run_dmrg["energy"]

    @pytest.mark.usefixtures("mocked_run_dmrg")
    def test_sz_get_qmi_false_returns_nones(self) -> None:
        _, s1, s2, mi = run_dmrg(
            mf=None, ncas=2, nelecas=2, symm_type="SZ", get_qmi=False
        )
        assert s1 is None
        assert s2 is None
        assert mi is None

    def test_sz_get_qmi_true_returns_qmi_arrays(self, mocked_run_dmrg) -> None:
        _, s1, s2, mi = run_dmrg(
            mf=None, ncas=2, nelecas=2, symm_type="SZ", get_qmi=True
        )
        assert s1 is mocked_run_dmrg["soe1"]
        assert s2 is mocked_run_dmrg["soe2"]
        assert mi is mocked_run_dmrg["minfo"]

    def test_sz_get_qmi_true_calls_compute_qmi_once(self, mocked_run_dmrg) -> None:
        run_dmrg(mf=None, ncas=2, nelecas=2, symm_type="SZ", get_qmi=True)
        mocked_run_dmrg["compute_qmi"].assert_called_once()

    def test_sz_get_qmi_false_skips_compute_qmi(self, mocked_run_dmrg) -> None:
        run_dmrg(mf=None, ncas=2, nelecas=2, symm_type="SZ", get_qmi=False)
        mocked_run_dmrg["compute_qmi"].assert_not_called()

    @pytest.mark.usefixtures("mocked_run_dmrg")
    def test_su2_get_qmi_false_returns_nones(self) -> None:
        _, s1, s2, mi = run_dmrg(
            mf=None, ncas=2, nelecas=2, symm_type="SU2", get_qmi=False
        )
        assert s1 is None
        assert s2 is None
        assert mi is None

    def test_su2_get_qmi_true_passes_zket_to_compute_qmi(self, mocked_run_dmrg) -> None:
        driver = mocked_run_dmrg["driver"]
        run_dmrg(mf=None, ncas=2, nelecas=2, symm_type="SU2", get_qmi=True)
        # SU2 path converts ket → zket via mps_change_to_sz before calling compute_qmi
        expected_zket = driver.mps_change_to_sz.return_value
        called_ket = mocked_run_dmrg["compute_qmi"].call_args[0][1]
        assert called_ket is expected_zket

    def test_omp_num_threads_env_var(self, mocked_run_dmrg, monkeypatch) -> None:
        monkeypatch.setenv("OMP_NUM_THREADS", "8")
        run_dmrg(mf=None, ncas=2, nelecas=2, symm_type="SZ", get_qmi=False)
        assert mocked_run_dmrg["MockDriver"].call_args.kwargs["n_threads"] == 8

    def test_scratch_dir_env_var(self, mocked_run_dmrg, monkeypatch) -> None:
        monkeypatch.setenv("SCRATCH_DIR", "/my/scratch")
        run_dmrg(mf=None, ncas=2, nelecas=2, symm_type="SZ", get_qmi=False)
        assert (
            mocked_run_dmrg["MockDriver"].call_args.kwargs["scratch"] == "/my/scratch"
        )
