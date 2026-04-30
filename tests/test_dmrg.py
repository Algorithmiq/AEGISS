"""Tests for aegiss.wavefunction.dmrg."""

from __future__ import annotations

import pytest

pytest.importorskip("pyblock2")

from aegiss.wavefunction.dmrg import run_dmrg  # noqa: E402


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
