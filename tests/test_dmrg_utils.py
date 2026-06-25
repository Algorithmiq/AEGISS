"""Tests for aegiss.utils.dmrg_utils."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from aegiss.utils.dmrg_utils import (
    _default_dmrg_schedule,
    _write_fcidump,
    _write_header,
    _write_input,
    _write_uhf_fcidump,
)

if TYPE_CHECKING:
    from pathlib import Path


class TestDefaultDmrgSchedule:
    def test_returns_three_lists(self) -> None:
        bond_dims, noises, thrds = _default_dmrg_schedule()
        assert isinstance(bond_dims, list)
        assert isinstance(noises, list)
        assert isinstance(thrds, list)

    def test_schedule_has_30_sweeps(self) -> None:
        bond_dims, noises, thrds = _default_dmrg_schedule()
        assert len(bond_dims) == 30
        assert len(noises) == 30
        assert len(thrds) == 30

    def test_all_lists_same_length(self) -> None:
        bond_dims, noises, thrds = _default_dmrg_schedule()
        assert len(bond_dims) == len(noises) == len(thrds)

    def test_first_segment_is_half_max_bond_dim(self) -> None:
        bond_dims, _, _ = _default_dmrg_schedule(max_bond_dim=500)
        assert bond_dims[0] == 250
        assert bond_dims[3] == 250

    def test_remaining_segments_use_max_bond_dim(self) -> None:
        bond_dims, _, _ = _default_dmrg_schedule(max_bond_dim=500)
        assert all(bd == 500 for bd in bond_dims[4:])

    def test_all_bond_dims_are_even(self) -> None:
        bond_dims, _, _ = _default_dmrg_schedule(max_bond_dim=500)
        assert all(bd % 2 == 0 for bd in bond_dims)

    def test_bond_dims_scale_with_max(self) -> None:
        bd_500, _, _ = _default_dmrg_schedule(max_bond_dim=500)
        bd_1000, _, _ = _default_dmrg_schedule(max_bond_dim=1000)
        assert bd_1000[0] == 2 * bd_500[0]
        assert bd_1000[-1] == 2 * bd_500[-1]

    def test_noises_non_negative(self) -> None:
        _, noises, _ = _default_dmrg_schedule()
        assert all(n >= 0.0 for n in noises)

    def test_noises_non_increasing(self) -> None:
        _, noises, _ = _default_dmrg_schedule()
        assert all(noises[i] >= noises[i + 1] for i in range(len(noises) - 1))

    def test_noises_end_with_zeros(self) -> None:
        _, noises, _ = _default_dmrg_schedule()
        np.testing.assert_allclose(noises[16:], 0.0, atol=1e-15)

    def test_thrds_positive(self) -> None:
        _, _, thrds = _default_dmrg_schedule()
        assert all(t > 0.0 for t in thrds)

    def test_thrds_non_increasing(self) -> None:
        _, _, thrds = _default_dmrg_schedule()
        assert all(thrds[i] >= thrds[i + 1] for i in range(len(thrds) - 1))

    def test_zero_max_bond_dim_raises(self) -> None:
        with pytest.raises(ValueError, match="max_bond_dim"):
            _default_dmrg_schedule(max_bond_dim=0)

    def test_negative_max_bond_dim_raises(self) -> None:
        with pytest.raises(ValueError, match="max_bond_dim"):
            _default_dmrg_schedule(max_bond_dim=-4)

    def test_odd_max_bond_dim_raises(self) -> None:
        with pytest.raises(ValueError, match="max_bond_dim"):
            _default_dmrg_schedule(max_bond_dim=501)

    def test_non_multiple_of_4_raises(self) -> None:
        with pytest.raises(ValueError, match="max_bond_dim"):
            _default_dmrg_schedule(max_bond_dim=250)


class TestWriteInput:
    """Unit-test _write_input in isolation by inspecting the written file."""

    def _make_input(self, tmp_path: Path, symm_type: str = "SZ") -> str:
        bond_dims, noises, thrds = _default_dmrg_schedule(max_bond_dim=500)
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


class TestWriteFcidump:
    """Unit-test _write_fcidump — verify ms is derived from spin (nalpha - nbeta)."""

    def _call_write_fcidump(self, tmp_path, nelecas):
        ncas = 3
        mock_casci = MagicMock()
        mock_casci.get_h1eff.return_value = (np.zeros((ncas, ncas)), 0.0)
        mock_casci.get_h2eff.return_value = np.zeros((4,))
        with (
            patch(
                "aegiss.utils.dmrg_utils.mcscf.CASCI",
                return_value=mock_casci,
            ),
            patch(
                "aegiss.utils.dmrg_utils.ao2mo.restore",
                return_value=np.zeros((ncas, ncas, ncas, ncas)),
            ),
            patch(
                "aegiss.utils.dmrg_utils.tools.fcidump.from_integrals"
            ) as mock_fcidump,
        ):
            _write_fcidump(tmp_path / "FCIDUMP", mf=None, ncas=ncas, nelecas=nelecas)
        return mock_fcidump.call_args

    def test_ms_equals_nalpha_minus_nbeta_closed_shell(self, tmp_path) -> None:
        call_args = self._call_write_fcidump(tmp_path, nelecas=(2, 2))
        assert call_args.kwargs["ms"] == 0

    def test_ms_equals_nalpha_minus_nbeta_doublet(self, tmp_path) -> None:
        call_args = self._call_write_fcidump(tmp_path, nelecas=(2, 1))
        assert call_args.kwargs["ms"] == 1

    def test_ms_equals_nalpha_minus_nbeta_triplet(self, tmp_path) -> None:
        call_args = self._call_write_fcidump(tmp_path, nelecas=(3, 1))
        assert call_args.kwargs["ms"] == 2


class TestWriteHeader:
    """Unit-test the UHF FCIDUMP header builder."""

    def test_norb_nelec_ms2_from_arguments(self) -> None:
        header = _write_header(orb_sym=[1, 1, 1], ncas=3, na=2, nb=1)
        assert "NORB=   3" in header
        assert "NELEC=    3" in header  # na + nb
        assert "MS2=    1" in header  # na - nb

    def test_marks_unrestricted_and_symmetry(self) -> None:
        header = _write_header(orb_sym=[1, 1], ncas=2, na=1, nb=1)
        assert "IUHF=1" in header
        assert "ISYM=   1" in header
        assert "ORBSYM=1,1," in header

    def test_terminated_by_end_sentinel(self) -> None:
        header = _write_header(orb_sym=[1], ncas=1, na=1, nb=0)
        assert header.rstrip().endswith("&END")


class TestWriteUhfFcidump:
    """Unit-test _write_uhf_fcidump: block ordering, sentinel separators, guards.

    ``write_eri`` / ``write_hcore`` (pyscf's own, already tested upstream) are
    mocked out so the file contains only the header plus the sentinel/core lines
    this function is responsible for; ``ao2mo.restore`` is the identity so the
    arrays handed to ``write_eri`` keep their object identity for order checks.
    """

    def _call(
        self,
        tmp_path: Path,
        nelecas: tuple[int, int] = (2, 1),
        ncas: int = 2,
        ecore: float = -1.25,
        h1e=None,
        g2e=None,
    ):
        if h1e is None:
            h1e = (np.eye(ncas), 0.5 * np.eye(ncas))
        if g2e is None:
            g2e = tuple(np.full((ncas,) * 4, fill) for fill in (0.1, 0.2, 0.3))
        mock_cas = MagicMock()
        mock_cas.get_h1eff.return_value = (h1e, ecore)
        mock_cas.get_h2eff.return_value = g2e
        path = tmp_path / "FCIDUMP.UHF"
        with (
            patch(
                "aegiss.utils.dmrg_utils.mcscf.UCASCI",
                return_value=mock_cas,
            ),
            patch(
                "aegiss.utils.dmrg_utils.ao2mo.restore",
                side_effect=lambda _fold, arr, _n: arr,
            ),
            patch("aegiss.utils.dmrg_utils.write_eri") as mock_eri,
            patch("aegiss.utils.dmrg_utils.write_hcore") as mock_hcore,
        ):
            _write_uhf_fcidump(path, mf=MagicMock(), ncas=ncas, nelecas=nelecas)
        return path, mock_eri, mock_hcore, h1e, g2e

    @staticmethod
    def _data_lines(path: Path) -> list[str]:
        lines = path.read_text().splitlines()
        start = next(i for i, ln in enumerate(lines) if ln.strip() == "&END") + 1
        return [ln for ln in lines[start:] if ln.split()]

    def test_header_marks_unrestricted(self, tmp_path) -> None:
        path, *_ = self._call(tmp_path, nelecas=(2, 1))
        text = path.read_text()
        assert "IUHF=1" in text
        assert "MS2=    1" in text  # na - nb

    def test_two_electron_blocks_written_in_aa_bb_ab_order(self, tmp_path) -> None:
        _, mock_eri, _, _h1e, g2e = self._call(tmp_path)
        written = [c.args[1] for c in mock_eri.call_args_list]
        assert len(written) == 3
        assert written[0] is g2e[0]  # aa
        assert written[1] is g2e[2]  # bb (before ab — matches block2's layout)
        assert written[2] is g2e[1]  # ab

    def test_one_electron_blocks_written_alpha_then_beta(self, tmp_path) -> None:
        _, _, mock_hcore, h1e, _g2e = self._call(tmp_path)
        written = [c.args[1] for c in mock_hcore.call_args_list]
        assert len(written) == 2
        assert written[0] is h1e[0]  # alpha
        assert written[1] is h1e[1]  # beta

    def test_sentinel_after_every_block_plus_core_line(self, tmp_path) -> None:
        # 5 blocks (aa, bb, ab, alpha, beta) each followed by a 0-valued
        # sentinel, then the trailing core-energy line: 6 "0 0 0 0" lines total.
        path, *_ = self._call(tmp_path, ecore=-1.25)
        data = self._data_lines(path)
        assert len(data) == 6
        for ln in data:
            assert tuple(int(x) for x in ln.split()[1:]) == (0, 0, 0, 0)
        values = [float(ln.split()[0]) for ln in data]
        assert values[:5] == [0.0, 0.0, 0.0, 0.0, 0.0]

    def test_core_energy_is_last_line(self, tmp_path) -> None:
        path, *_ = self._call(tmp_path, ecore=-57.904)
        last = self._data_lines(path)[-1]
        assert float(last.split()[0]) == pytest.approx(-57.904)

    def test_h1e_not_a_pair_raises_type_error(self, tmp_path) -> None:
        # get_h1eff returns a single array instead of an (alpha, beta) tuple
        with pytest.raises(TypeError, match="UHF h1e"):
            self._call(tmp_path, h1e=np.eye(2))

    def test_g2e_not_a_triple_raises_type_error(self, tmp_path) -> None:
        g2e = (np.full((2,) * 4, 0.1), np.full((2,) * 4, 0.2))  # only 2 blocks
        with pytest.raises(TypeError, match="UHF g2e"):
            self._call(tmp_path, g2e=g2e)
