"""Tests for aegiss.utils.orb_display and the ``aegiss config`` CLI.

Covers the resolution precedence (env var > config file > default), the
input/output conversion helpers, and round-trip safety (storage stays 0-based).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from aegiss import cli
from aegiss.active_space.ao_projection import _resolve_space, select_by_ao_projection
from aegiss.localization.orbital_localization import localize_fnos
from aegiss.utils import orb_display
from aegiss.utils.dump import save_orbitals
from aegiss.utils.mo_composition import mo_ao_contributions

if TYPE_CHECKING:
    from pathlib import Path


# Isolation from the machine's real config (XDG dir + AEGISS_ORB_BASE + cache)
# is handled suite-wide by the autouse `_default_orbital_base` fixture in
# conftest.py; tests here only opt in to a non-default base as needed.


def _write_base(base: int) -> None:
    """Persist *base* via the CLI writer and refresh the resolver cache."""
    cli._set_base(base)
    orb_display.reset_cache()


class TestResolutionPrecedence:
    def test_default_is_zero(self) -> None:
        assert orb_display.orbital_base() == 0

    def test_config_file_used_when_env_absent(self) -> None:
        _write_base(1)
        assert orb_display.orbital_base() == 1

    def test_env_overrides_config_file(self, monkeypatch) -> None:
        _write_base(1)
        monkeypatch.setenv("AEGISS_ORB_BASE", "0")
        orb_display.reset_cache()
        assert orb_display.orbital_base() == 0

    def test_env_used_when_no_file(self, monkeypatch) -> None:
        monkeypatch.setenv("AEGISS_ORB_BASE", "1")
        orb_display.reset_cache()
        assert orb_display.orbital_base() == 1

    def test_invalid_env_falls_back_to_file(self, monkeypatch) -> None:
        _write_base(1)
        monkeypatch.setenv("AEGISS_ORB_BASE", "not-an-int")
        orb_display.reset_cache()
        assert orb_display.orbital_base() == 1

    def test_out_of_range_env_falls_back_to_default(self, monkeypatch) -> None:
        monkeypatch.setenv("AEGISS_ORB_BASE", "2")
        orb_display.reset_cache()
        assert orb_display.orbital_base() == 0

    def test_malformed_config_falls_back_to_default(self) -> None:
        path = orb_display.config_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("this is not = valid = toml", encoding="utf-8")
        orb_display.reset_cache()
        assert orb_display.orbital_base() == 0


class TestConversionHelpers:
    def test_to_display_default_is_identity(self) -> None:
        assert orb_display.to_display(7) == 7

    def test_to_display_base_one(self, monkeypatch) -> None:
        monkeypatch.setenv("AEGISS_ORB_BASE", "1")
        orb_display.reset_cache()
        assert orb_display.to_display(7) == 8
        assert orb_display.to_display_all([0, 4, 9]) == [1, 5, 10]

    def test_from_display_base_one(self, monkeypatch) -> None:
        monkeypatch.setenv("AEGISS_ORB_BASE", "1")
        orb_display.reset_cache()
        assert orb_display.from_display(8) == 7
        assert orb_display.from_display_all([1, 5, 10]) == [0, 4, 9]

    @pytest.mark.parametrize("base", ["0", "1"])
    def test_round_trip_is_identity(self, monkeypatch, base: str) -> None:
        # A value the user types, converted in and back out, is unchanged.
        monkeypatch.setenv("AEGISS_ORB_BASE", base)
        orb_display.reset_cache()
        for user_value in (1, 5, 42):
            stored = orb_display.from_display(user_value)
            assert orb_display.to_display(stored) == user_value


class TestInputBoundary:
    """The plain-list input boundary in ao_projection._resolve_space."""

    def test_plain_user_list_is_shifted_to_zero_based(self, monkeypatch) -> None:
        monkeypatch.setenv("AEGISS_ORB_BASE", "1")
        orb_display.reset_cache()
        # User types 1-based [1, 2, 3]; internal space must be 0-based [0, 1, 2].
        space = _resolve_space(mf=None, restricted_space=[1, 2, 3])
        assert [idx for idx, _ in space] == [0, 1, 2]

    def test_internal_pair_list_is_not_shifted(self, monkeypatch) -> None:
        monkeypatch.setenv("AEGISS_ORB_BASE", "1")
        orb_display.reset_cache()
        # Pair-lists come from AEGISS internals and are already 0-based.
        pairs = [(0, 0.5), (1, 0.4)]
        space = _resolve_space(mf=None, restricted_space=pairs)
        assert [idx for idx, _ in space] == [0, 1]


class TestBoundaryIntegration:
    """End-to-end checks that base conversion reaches the real call sites.

    Each test proves the invariant: a user index typed in base 1 selects the
    same internal orbital as the equivalent 0-based index, while returned data
    stays 0-based.
    """

    def test_mo_ao_contributions_input_is_shifted(
        self, monkeypatch, h2o_rhf: Any
    ) -> None:
        # Base 0: internal orbital 4.
        monkeypatch.delenv("AEGISS_ORB_BASE", raising=False)
        orb_display.reset_cache()
        base0 = mo_ao_contributions(h2o_rhf, 4, top_n=3)

        # Base 1: user types 5, which must map to the same internal orbital 4.
        monkeypatch.setenv("AEGISS_ORB_BASE", "1")
        orb_display.reset_cache()
        base1 = mo_ao_contributions(h2o_rhf, 5, top_n=3)

        # Returned dict keys stay 0-based, so both select orbital 4 identically.
        assert list(base1.keys()) == [4]
        assert base1[4]["alpha"] == base0[4]["alpha"]

    def test_mo_ao_contributions_out_of_range_reports_display_base(
        self, monkeypatch, h2o_rhf: Any
    ) -> None:
        n_mo = h2o_rhf.mo_coeff.shape[1]
        monkeypatch.setenv("AEGISS_ORB_BASE", "1")
        orb_display.reset_cache()
        # In base 1 the valid range is [1, n_mo]; n_mo + 1 is the first invalid.
        with pytest.raises(ValueError, match=rf"\[1, {n_mo + 1}\)"):
            mo_ao_contributions(h2o_rhf, n_mo + 1)

    def test_localize_fnos_frozen_input_is_shifted(
        self, monkeypatch, be_rhf: Any
    ) -> None:
        monkeypatch.delenv("AEGISS_ORB_BASE", raising=False)
        orb_display.reset_cache()
        base0 = localize_fnos(be_rhf, "mp2", frozen=[0])

        monkeypatch.setenv("AEGISS_ORB_BASE", "1")
        orb_display.reset_cache()
        # frozen=[1] in base 1 must freeze the same internal orbital 0.
        base1 = localize_fnos(be_rhf, "mp2", frozen=[1])
        np.testing.assert_allclose(base0, base1)

    @pytest.mark.parametrize(("base", "expected"), [("0", [1, 3]), ("1", [2, 4])])
    def test_cube_filenames_use_display_base(
        self, monkeypatch, h2o_rhf: Any, tmp_path: Path, base: str, expected: list[int]
    ) -> None:
        recorded: list[str] = []
        monkeypatch.setattr(
            "pyscf.tools.cubegen.orbital",
            lambda _mol, filename, _coeff: recorded.append(filename),
        )
        monkeypatch.setenv("AEGISS_ORB_BASE", base)
        orb_display.reset_cache()

        save_orbitals(
            h2o_rhf.mol,
            h2o_rhf.mo_coeff,
            np.array([1, 3]),  # 0-based internal indices
            filename_prefix="orb",
            folder=str(tmp_path),
        )
        suffixes = sorted(int(name.split("_")[-1].split(".")[0]) for name in recorded)
        assert suffixes == expected


class TestBaseEquivalence:
    """Base is display-only: base-0 and base-1 runs must correspond exactly.

    Switching base must not change any internal or returned result, and a
    base-1 input of ``k`` must behave identically to a base-0 input of ``k-1``.
    """

    @staticmethod
    def _project(mf: Any, restricted_space: list[int] | None = None) -> np.ndarray:
        return select_by_ao_projection(
            mf=mf,
            ao_labels=["H 1s"],
            projected_basis="minao",
            contribution=0.0,
            contribution_type="a",
            restricted_space=restricted_space,
        )

    def test_return_values_are_base_invariant(self, monkeypatch, h2o_rhf: Any) -> None:
        monkeypatch.delenv("AEGISS_ORB_BASE", raising=False)
        orb_display.reset_cache()
        base0 = self._project(h2o_rhf)

        monkeypatch.setenv("AEGISS_ORB_BASE", "1")
        orb_display.reset_cache()
        base1 = self._project(h2o_rhf)

        # Returned indices are always 0-based, hence identical across bases.
        np.testing.assert_array_equal(base0, base1)

    def test_restricted_space_input_corresponds_across_bases(
        self, monkeypatch, h2o_rhf: Any
    ) -> None:
        base0_space = [1, 2, 3, 4, 5]
        monkeypatch.delenv("AEGISS_ORB_BASE", raising=False)
        orb_display.reset_cache()
        result_b0 = self._project(h2o_rhf, restricted_space=base0_space)

        monkeypatch.setenv("AEGISS_ORB_BASE", "1")
        orb_display.reset_cache()
        # The same physical orbitals, expressed 1-based, must select identically.
        result_b1 = self._project(
            h2o_rhf, restricted_space=[k + 1 for k in base0_space]
        )

        np.testing.assert_array_equal(result_b0, result_b1)
        # And a wrong (unshifted) base-1 input would select different orbitals:
        assert set(result_b0) <= set(base0_space)


class TestCli:
    def test_set_base_persists_and_get_base_reads_it(self, capsys) -> None:
        assert cli.main(["config", "set-base", "1"]) == 0
        orb_display.reset_cache()
        capsys.readouterr()
        assert cli.main(["config", "get-base"]) == 0
        assert capsys.readouterr().out.strip() == "1"

    def test_set_base_rejects_invalid(self) -> None:
        with pytest.raises(SystemExit):
            cli.main(["config", "set-base", "5"])

    def test_set_base_preserves_other_config_keys(self) -> None:
        path = orb_display.config_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('other = "keep me"\n', encoding="utf-8")
        cli.main(["config", "set-base", "1"])
        contents = path.read_text(encoding="utf-8")
        assert 'other = "keep me"' in contents
        assert "orbital_base = 1" in contents
