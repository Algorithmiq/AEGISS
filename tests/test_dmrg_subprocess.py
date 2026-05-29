"""Tests for aegiss.wavefunction.dmrg_subprocess."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

if TYPE_CHECKING:
    from pathlib import Path

from aegiss.utils.dmrg_utils import _default_dmrg_schedule
from aegiss.wavefunction.dmrg_subprocess import (
    _compute_soe,
    _write_input,
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
            "aegiss.wavefunction.dmrg_subprocess.mcscf.CASCI",
            return_value=mock_casci,
        ) as MockCASCI,
        patch(
            "aegiss.wavefunction.dmrg_subprocess.ao2mo.restore",
            return_value=np.zeros((_NCAS, _NCAS, _NCAS, _NCAS)),
        ),
        patch("aegiss.wavefunction.dmrg_subprocess.tools.fcidump.from_integrals"),
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
        e, *_ = run_dmrg_subprocess(mf=None, ncas=_NCAS, nelecas=2, get_qmi=False)
        assert e == _FAKE_ENERGY

    def test_get_qmi_false_returns_three_nones(self) -> None:
        _, s1, s2, mi = run_dmrg_subprocess(
            mf=None, ncas=_NCAS, nelecas=2, get_qmi=False
        )
        assert s1 is None
        assert s2 is None
        assert mi is None

    def test_get_qmi_true_returns_soe1_array(self) -> None:
        _, s1, _s2, _mi = run_dmrg_subprocess(
            mf=None, ncas=_NCAS, nelecas=2, get_qmi=True
        )
        assert s1 is not None
        assert s1.shape == (_NCAS,)
        np.testing.assert_allclose(s1, _EXPECTED_SOE, atol=1e-12)

    def test_get_qmi_true_soe2_is_none(self) -> None:
        _, _s1, _s2, _mi = run_dmrg_subprocess(
            mf=None, ncas=_NCAS, nelecas=2, get_qmi=True
        )
        assert _s2 is None

    def test_get_qmi_true_minfo_is_none(self) -> None:
        _, _s1, _s2, _mi = run_dmrg_subprocess(
            mf=None, ncas=_NCAS, nelecas=2, get_qmi=True
        )
        assert _mi is None


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
class TestRunDmrgSubprocessEnvVars:
    """Environment variable handling mirrors the pyblock2 backend."""

    def test_omp_num_threads(self, monkeypatch) -> None:
        monkeypatch.setenv("OMP_NUM_THREADS", "8")
        with patch("aegiss.wavefunction.dmrg_subprocess._write_input") as mock_wi:
            run_dmrg_subprocess(mf=None, ncas=_NCAS, nelecas=2, get_qmi=False)
        n_threads_arg = mock_wi.call_args[0][8]  # positional: n_threads
        assert n_threads_arg == 8

    def test_scratch_dir(self, monkeypatch, tmp_path) -> None:
        monkeypatch.setenv("SCRATCH_DIR", str(tmp_path))
        with patch("aegiss.wavefunction.dmrg_subprocess._write_input") as mock_wi:
            run_dmrg_subprocess(mf=None, ncas=_NCAS, nelecas=2, get_qmi=False)
        scratch_arg = mock_wi.call_args[0][4]  # positional: scratch_dir (Path)
        assert str(scratch_arg).startswith(str(tmp_path))

    def test_mem_default(self, monkeypatch) -> None:
        monkeypatch.delenv("MEM", raising=False)
        with patch("aegiss.wavefunction.dmrg_subprocess._write_input") as mock_wi:
            run_dmrg_subprocess(mf=None, ncas=_NCAS, nelecas=2, get_qmi=False)
        mem_gb_arg = mock_wi.call_args[0][9]  # positional: stack_mem_gb
        assert mem_gb_arg == 4.0

    def test_mem_env_var(self, monkeypatch) -> None:
        monkeypatch.setenv("MEM", "16")
        with patch("aegiss.wavefunction.dmrg_subprocess._write_input") as mock_wi:
            run_dmrg_subprocess(mf=None, ncas=_NCAS, nelecas=2, get_qmi=False)
        mem_gb_arg = mock_wi.call_args[0][9]
        assert mem_gb_arg == 16.0

    def test_mem_env_var_non_integer_falls_back_to_default(self, monkeypatch) -> None:
        monkeypatch.setenv("MEM", "not_an_int")
        with patch("aegiss.wavefunction.dmrg_subprocess._write_input") as mock_wi:
            run_dmrg_subprocess(mf=None, ncas=_NCAS, nelecas=2, get_qmi=False)
        mem_gb_arg = mock_wi.call_args[0][9]
        assert mem_gb_arg == 4.0


class TestWriteInput:
    """Unit-test _write_input in isolation by inspecting the written file."""

    def _make_input(self, tmp_path: Path, symm_type: str = "SZ") -> str:

        bond_dims, noises, thrds = _default_dmrg_schedule(init_bond_dim=10)
        path = tmp_path / "dmrg.conf"
        fcidump = tmp_path / "FCIDUMP"
        _write_input(
            path, fcidump, 2, 0, tmp_path, bond_dims, noises, thrds, 4, 4.0, symm_type
        )
        return path.read_text()

    def test_sz_mode_has_nonspinadapted(self, tmp_path) -> None:
        content = self._make_input(tmp_path, symm_type="SZ")
        assert "nonspinadapted" in content

    def test_su2_mode_no_nonspinadapted(self, tmp_path) -> None:
        content = self._make_input(tmp_path, symm_type="SU2")
        assert "nonspinadapted" not in content

    def test_required_keywords_present(self, tmp_path) -> None:
        content = self._make_input(tmp_path)
        for keyword in (
            "hf_occ integral",
            "noreorder",
            "onepdm",
            "correlation",
            "sym c1",
        ):
            assert keyword in content, f"Missing keyword: {keyword!r}"

    def test_schedule_block_present(self, tmp_path) -> None:
        content = self._make_input(tmp_path)
        lines = content.splitlines()
        assert "schedule" in lines
        assert "end" in lines
        sched_start = lines.index("schedule")
        sched_end = lines.index("end")
        assert sched_end > sched_start + 1, "schedule block contains no rows"


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
