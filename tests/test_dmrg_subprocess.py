"""Tests for aegiss.wavefunction.dmrg_subprocess."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from pyscf import scf

from aegiss.wavefunction.dmrg_subprocess import (
    _compute_soe,
    _compute_spin_soe,
    _read_outputs,
    run_dmrg_subprocess,
)

# ---------------------------------------------------------------------------
# Shared fake PDM data
#
# Half-filled orbital: n_alpha = n_beta= 0.5, d = 0.25
# npc1[0,i,i] = n_i + 2*d_i = 1.0 + 0.5 = 1.5
# Eigenvalues: [0.25, 0.25, 0.25, 0.25]  →  s_i = log(4)
# ---------------------------------------------------------------------------
_NCAS = 2
_FAKE_ENERGY = -2.345
_FAKE_PDM1 = np.full((2, _NCAS, _NCAS), 0.0)
for _i in range(_NCAS):
    _FAKE_PDM1[0, _i, _i] = 0.5
    _FAKE_PDM1[1, _i, _i] = 0.5
_FAKE_NPC1 = np.full((2, _NCAS, _NCAS), 0.0)
for _i in range(_NCAS):
    _FAKE_NPC1[0, _i, _i] = 1.5
    _FAKE_NPC1[1, _i, _i] = 0.5
_EXPECTED_SOE = np.full(_NCAS, np.log(4.0))


@pytest.fixture
def mocked_run_dmrg_subprocess():
    """Patch all external dependencies of run_dmrg_subprocess."""
    mock_casci = MagicMock()
    mock_casci.get_h1eff.return_value = (np.zeros((_NCAS, _NCAS)), 0.0)
    mock_casci.get_h2eff.return_value = np.zeros((4,))

    with (
        patch("aegiss.wavefunction.dmrg_subprocess.subprocess.run") as mock_subproc,
        patch(
            "aegiss.wavefunction.dmrg_subprocess._read_outputs",
            return_value=(_FAKE_ENERGY, _FAKE_PDM1, _FAKE_NPC1),
        ) as mock_read,
        patch(
            "aegiss.utils.dmrg_utils.mcscf.CASCI",
            return_value=mock_casci,
        ) as MockCASCI,
        patch(
            "aegiss.utils.dmrg_utils.ao2mo.restore",
            return_value=np.zeros((_NCAS, _NCAS, _NCAS, _NCAS)),
        ),
        patch("aegiss.utils.dmrg_utils.tools.fcidump.from_integrals"),
        patch(
            "aegiss.wavefunction.dmrg_subprocess.resolve_nelecas",
            return_value=(1, 1),
        ),
    ):
        yield {
            "mock_subproc": mock_subproc,
            "mock_read": mock_read,
            "MockCASCI": MockCASCI,
            "casci": mock_casci,
        }


class TestRunDmrgSubprocessSetup:
    """The subprocess is called and integrals are fetched correctly."""

    def test_subprocess_called_once(self, mocked_run_dmrg_subprocess) -> None:
        run_dmrg_subprocess(mf=None, ncas=_NCAS, nelecas=2, get_qmi=False)
        mocked_run_dmrg_subprocess["mock_subproc"].assert_called_once()

    def test_casci_called_with_mf_ncas_nelecas(
        self, mocked_run_dmrg_subprocess
    ) -> None:
        sentinel = object()
        run_dmrg_subprocess(mf=sentinel, ncas=_NCAS, nelecas=2, get_qmi=False)
        mocked_run_dmrg_subprocess["MockCASCI"].assert_called_once_with(
            sentinel, _NCAS, 2
        )

    def test_h1eff_fetched(self, mocked_run_dmrg_subprocess) -> None:
        run_dmrg_subprocess(mf=None, ncas=_NCAS, nelecas=2, get_qmi=False)
        mocked_run_dmrg_subprocess["casci"].get_h1eff.assert_called_once()

    def test_h2eff_fetched(self, mocked_run_dmrg_subprocess) -> None:
        run_dmrg_subprocess(mf=None, ncas=_NCAS, nelecas=2, get_qmi=False)
        mocked_run_dmrg_subprocess["casci"].get_h2eff.assert_called_once()

    def test_read_outputs_called_once(self, mocked_run_dmrg_subprocess) -> None:
        run_dmrg_subprocess(mf=None, ncas=_NCAS, nelecas=2, get_qmi=False)
        mocked_run_dmrg_subprocess["mock_read"].assert_called_once()


@pytest.mark.usefixtures("mocked_run_dmrg_subprocess")
class TestRunDmrgSubprocessBehavior:
    """Return-value contract."""

    def test_energy_returned(self) -> None:
        e, _ = run_dmrg_subprocess(mf=None, ncas=_NCAS, nelecas=2, get_qmi=False)
        assert e == _FAKE_ENERGY

    def test_get_qmi_false_returns_none_soe1(self) -> None:
        _, s1 = run_dmrg_subprocess(mf=None, ncas=_NCAS, nelecas=2, get_qmi=False)
        assert s1 is None

    def test_get_qmi_true_returns_soe1_array(self) -> None:
        _, s1 = run_dmrg_subprocess(mf=None, ncas=_NCAS, nelecas=2, get_qmi=True)
        assert s1 is not None
        assert s1.shape == (_NCAS,)
        np.testing.assert_allclose(s1, _EXPECTED_SOE, atol=1e-12)


class TestRunDmrgSubprocessSymmTypeValidation:
    def test_invalid_symm_type_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="Unsupported symmetry type"):
            run_dmrg_subprocess(mf=None, ncas=4, nelecas=4, symm_type="SU3")

    def test_invalid_symm_type_message_contains_valid_options(self) -> None:
        with pytest.raises(ValueError, match="SU2"):
            run_dmrg_subprocess(mf=None, ncas=4, nelecas=4, symm_type="invalid")

    def test_invalid_symm_type_is_included_in_message(self) -> None:
        with pytest.raises(ValueError, match="'bad'"):
            run_dmrg_subprocess(mf=None, ncas=4, nelecas=4, symm_type="bad")


@pytest.mark.usefixtures("mocked_run_dmrg_subprocess")
class TestRunDmrgSubprocessUnrestrictedDispatch:
    """`unrestricted` selects the UHF writer; autodetect + override + SZ guard."""

    @staticmethod
    def _run_with_patched_writers(**kwargs):
        with (
            patch("aegiss.wavefunction.dmrg_subprocess._write_uhf_fcidump") as mock_uhf,
            patch("aegiss.wavefunction.dmrg_subprocess._write_fcidump") as mock_rhf,
        ):
            run_dmrg_subprocess(ncas=_NCAS, get_qmi=False, **kwargs)
        return mock_uhf, mock_rhf

    def test_unrestricted_true_calls_uhf_writer(self) -> None:
        mock_uhf, mock_rhf = self._run_with_patched_writers(
            mf=None, nelecas=2, unrestricted=True
        )
        mock_uhf.assert_called_once()
        mock_rhf.assert_not_called()

    def test_unrestricted_false_calls_restricted_writer(self) -> None:
        mock_uhf, mock_rhf = self._run_with_patched_writers(
            mf=None, nelecas=2, unrestricted=False
        )
        mock_rhf.assert_called_once()
        mock_uhf.assert_not_called()

    def test_default_none_autodetects_uhf_from_mf(self) -> None:
        mock_uhf, mock_rhf = self._run_with_patched_writers(
            mf=MagicMock(spec=scf.uhf.UHF), nelecas=2
        )
        mock_uhf.assert_called_once()
        mock_rhf.assert_not_called()

    def test_default_none_uses_restricted_for_non_uhf_mf(self) -> None:
        mock_uhf, mock_rhf = self._run_with_patched_writers(mf=None, nelecas=2)
        mock_rhf.assert_called_once()
        mock_uhf.assert_not_called()

    def test_explicit_false_overrides_uhf_mf(self) -> None:
        mock_uhf, mock_rhf = self._run_with_patched_writers(
            mf=MagicMock(spec=scf.uhf.UHF), nelecas=2, unrestricted=False
        )
        mock_rhf.assert_called_once()
        mock_uhf.assert_not_called()

    def test_uhf_writer_receives_nalpha_nbeta_tuple(self) -> None:
        # resolve_nelecas is patched to return (1, 1) by the fixture.
        mock_uhf, _ = self._run_with_patched_writers(
            mf=None, nelecas=2, unrestricted=True
        )
        assert mock_uhf.call_args[0][3] == (1, 1)

    def test_unrestricted_with_su2_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="require symm_type='SZ'"):
            run_dmrg_subprocess(
                mf=None, ncas=_NCAS, nelecas=2, symm_type="SU2", unrestricted=True
            )


@pytest.mark.usefixtures("mocked_run_dmrg_subprocess")
class TestRunDmrgSubprocessEnvVars:
    """Environment variable handling for the subprocess backend."""

    def test_omp_num_threads(self, monkeypatch) -> None:
        monkeypatch.setenv("OMP_NUM_THREADS", "8")
        with patch("aegiss.wavefunction.dmrg_subprocess._write_input") as mock_wi:
            run_dmrg_subprocess(mf=None, ncas=_NCAS, nelecas=2, get_qmi=False)
        n_threads_arg = mock_wi.call_args[0][9]  # positional: n_threads
        assert n_threads_arg == 8

    def test_scratch_dir(self, monkeypatch, tmp_path) -> None:
        monkeypatch.setenv("SCRATCH_DIR", str(tmp_path))
        with patch("aegiss.wavefunction.dmrg_subprocess._write_input") as mock_wi:
            run_dmrg_subprocess(mf=None, ncas=_NCAS, nelecas=2, get_qmi=False)
        scratch_arg = mock_wi.call_args[0][5]  # positional: scratch_dir (Path)
        assert str(scratch_arg).startswith(str(tmp_path))

    def test_mem_default(self, monkeypatch) -> None:
        monkeypatch.delenv("MEM", raising=False)
        with patch("aegiss.wavefunction.dmrg_subprocess._write_input") as mock_wi:
            run_dmrg_subprocess(mf=None, ncas=_NCAS, nelecas=2, get_qmi=False)
        mem_gb_arg = mock_wi.call_args[0][10]  # positional: stack_mem_gb
        assert mem_gb_arg == pytest.approx(4.0)

    def test_mem_env_var(self, monkeypatch) -> None:
        monkeypatch.setenv("MEM", "16")
        with patch("aegiss.wavefunction.dmrg_subprocess._write_input") as mock_wi:
            run_dmrg_subprocess(mf=None, ncas=_NCAS, nelecas=2, get_qmi=False)
        mem_gb_arg = mock_wi.call_args[0][10]
        assert mem_gb_arg == pytest.approx(16.0)

    def test_mem_env_var_non_integer_falls_back_to_default(self, monkeypatch) -> None:
        monkeypatch.setenv("MEM", "not_an_int")
        with patch("aegiss.wavefunction.dmrg_subprocess._write_input") as mock_wi:
            run_dmrg_subprocess(mf=None, ncas=_NCAS, nelecas=2, get_qmi=False)
        mem_gb_arg = mock_wi.call_args[0][10]
        assert mem_gb_arg == pytest.approx(4.0)


class TestComputeSOE:
    """Unit-test _compute_soe against hand-computed entropies."""

    def _make_pdm_npc(self, n_alpha: float, n_beta: float, d: float, ncas: int = 1):
        pdm1 = np.zeros((2, ncas, ncas))
        npc1 = np.zeros((2, ncas, ncas))
        for i in range(ncas):
            pdm1[0, i, i] = n_alpha
            pdm1[1, i, i] = n_beta
            n_i = n_alpha + n_beta
            npc1[0, i, i] = n_i + 2 * d
        return pdm1, npc1

    def test_empty_orbital_entropy_is_zero(self) -> None:
        pdm1, npc1 = self._make_pdm_npc(0.0, 0.0, 0.0)
        soe = _compute_soe(pdm1, npc1)
        assert soe[0] == pytest.approx(0.0)

    def test_doubly_occupied_entropy_is_zero(self) -> None:
        pdm1, npc1 = self._make_pdm_npc(1.0, 1.0, 1.0)
        soe = _compute_soe(pdm1, npc1)
        assert soe[0] == pytest.approx(0.0)

    def test_singly_occupied_alpha_entropy_is_zero(self) -> None:
        pdm1, npc1 = self._make_pdm_npc(1.0, 0.0, 0.0)
        soe = _compute_soe(pdm1, npc1)
        assert soe[0] == pytest.approx(0.0)

    def test_half_filled_equal_weight_entropy_is_log4(self) -> None:
        # w = [0.25, 0.25, 0.25, 0.25]  →  s = log(4)
        pdm1, npc1 = self._make_pdm_npc(0.5, 0.5, 0.25)
        soe = _compute_soe(pdm1, npc1)
        assert soe[0] == pytest.approx(np.log(4.0))

    def test_output_shape_matches_ncas(self) -> None:
        pdm1, npc1 = self._make_pdm_npc(0.5, 0.5, 0.25, ncas=4)
        soe = _compute_soe(pdm1, npc1)
        assert soe.shape == (4,)


class TestComputeSpinSOE:
    """Unit-test _compute_spin_soe: per-spin entropy = binary entropy of occupation."""

    @staticmethod
    def _pdm(n_alpha: float, n_beta: float, ncas: int = 1) -> np.ndarray:
        pdm1 = np.zeros((2, ncas, ncas))
        for i in range(ncas):
            pdm1[0, i, i] = n_alpha
            pdm1[1, i, i] = n_beta
        return pdm1

    def test_shape_is_two_by_ncas(self) -> None:
        soe = _compute_spin_soe(self._pdm(0.5, 0.5, ncas=4))
        assert soe.shape == (2, 4)

    def test_empty_and_full_spin_orbitals_are_zero(self) -> None:
        soe = _compute_spin_soe(self._pdm(1.0, 0.0))
        # alpha fully occupied, beta empty -> both binary entropies vanish
        assert soe[0, 0] == pytest.approx(0.0)
        assert soe[1, 0] == pytest.approx(0.0)

    def test_half_filled_spin_orbital_is_log2(self) -> None:
        soe = _compute_spin_soe(self._pdm(0.5, 0.5))
        assert soe[0, 0] == pytest.approx(np.log(2.0))
        assert soe[1, 0] == pytest.approx(np.log(2.0))

    def test_alpha_and_beta_independent(self) -> None:
        # alpha half-filled (log2), beta fully occupied (0)
        soe = _compute_spin_soe(self._pdm(0.5, 1.0))
        assert soe[0, 0] == pytest.approx(np.log(2.0))
        assert soe[1, 0] == pytest.approx(0.0)

    def test_subadditive_vs_spin_summed(self) -> None:
        # spin-summed orbital entropy must not exceed S_alpha + S_beta
        pdm1 = self._pdm(0.5, 0.5)
        npc1 = np.zeros((2, 1, 1))
        npc1[0, 0, 0] = (0.5 + 0.5) + 2 * 0.25  # d = 0.25 -> equal-weight 4 states
        summed = _compute_soe(pdm1, npc1)
        spin = _compute_spin_soe(pdm1)
        assert summed[0] <= spin.sum(axis=0)[0] + 1e-12


class TestReadOutputs:
    """Unit-test _read_outputs by writing fake .npy files to a tmp directory."""

    def _write_fake_outputs(
        self,
        tmp_path,
        ncas: int,
        energy: float = -1.0,
        pdm_ncas: int | None = None,
        npc_ncas: int | None = None,
    ) -> None:
        node0 = tmp_path / "node0"
        node0.mkdir(parents=True)
        np.save(node0 / "E_dmrg.npy", np.array(energy))
        np.save(node0 / "1pdm.npy", np.zeros((2, pdm_ncas or ncas, pdm_ncas or ncas)))
        np.save(node0 / "1npc.npy", np.zeros((2, npc_ncas or ncas, npc_ncas or ncas)))

    def test_pdm_shape_is_2_ncas_ncas(self, tmp_path) -> None:
        self._write_fake_outputs(tmp_path, ncas=4)
        _, pdm1, _ = _read_outputs(tmp_path, ncas=4)
        assert pdm1.shape == (2, 4, 4)

    def test_pdm_values_match_saved_array(self, tmp_path) -> None:
        node0 = tmp_path / "node0"
        node0.mkdir()
        expected = np.arange(8, dtype=float).reshape(2, 2, 2)
        np.save(node0 / "E_dmrg.npy", np.array(-1.0))
        np.save(node0 / "1pdm.npy", expected)
        np.save(node0 / "1npc.npy", np.zeros((2, 2, 2)))
        _, pdm1, _ = _read_outputs(tmp_path, ncas=2)
        np.testing.assert_array_equal(pdm1, expected)

    def test_npc_shape_is_2_ncas_ncas(self, tmp_path) -> None:
        self._write_fake_outputs(tmp_path, ncas=3)
        _, _, npc1 = _read_outputs(tmp_path, ncas=3)
        assert npc1.shape == (2, 3, 3)

    def test_npc_values_match_saved_array(self, tmp_path) -> None:
        node0 = tmp_path / "node0"
        node0.mkdir()
        expected = np.arange(8, dtype=float).reshape(2, 2, 2) + 10
        np.save(node0 / "E_dmrg.npy", np.array(-1.0))
        np.save(node0 / "1pdm.npy", np.zeros((2, 2, 2)))
        np.save(node0 / "1npc.npy", expected)
        _, _, npc1 = _read_outputs(tmp_path, ncas=2)
        np.testing.assert_array_equal(npc1, expected)

    def test_ncas_mismatch_pdm_raises_value_error(self, tmp_path) -> None:
        self._write_fake_outputs(tmp_path, ncas=4, pdm_ncas=3)
        with pytest.raises(ValueError, match="Unexpected block2main output shapes"):
            _read_outputs(tmp_path, ncas=4)

    def test_ncas_mismatch_npc_raises_value_error(self, tmp_path) -> None:
        self._write_fake_outputs(tmp_path, ncas=4, npc_ncas=3)
        with pytest.raises(ValueError, match="Unexpected block2main output shapes"):
            _read_outputs(tmp_path, ncas=4)

    def test_missing_energy_file_raises_file_not_found(self, tmp_path) -> None:
        node0 = tmp_path / "node0"
        node0.mkdir()
        np.save(node0 / "1pdm.npy", np.zeros((2, 2, 2)))
        np.save(node0 / "1npc.npy", np.zeros((2, 2, 2)))
        with pytest.raises(FileNotFoundError):
            _read_outputs(tmp_path, ncas=2)

    def test_missing_pdm_file_raises_file_not_found(self, tmp_path) -> None:
        node0 = tmp_path / "node0"
        node0.mkdir()
        np.save(node0 / "E_dmrg.npy", np.array(-1.0))
        np.save(node0 / "1npc.npy", np.zeros((2, 2, 2)))
        with pytest.raises(FileNotFoundError):
            _read_outputs(tmp_path, ncas=2)

    def test_missing_npc_file_raises_file_not_found(self, tmp_path) -> None:
        node0 = tmp_path / "node0"
        node0.mkdir()
        np.save(node0 / "E_dmrg.npy", np.array(-1.0))
        np.save(node0 / "1pdm.npy", np.zeros((2, 2, 2)))
        with pytest.raises(FileNotFoundError):
            _read_outputs(tmp_path, ncas=2)
