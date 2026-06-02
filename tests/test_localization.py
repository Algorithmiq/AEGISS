"""Tests for frozen-orbital support in localize_nos and localize_fnos."""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import numpy as np

from aegiss.localization.orbital_localization import (
    localize,
    localize_fnos,
    localize_nos,
)
from aegiss.wavefunction.methods import build_correlated_wfn

# Be/sto-3g: 5 MOs (occ=[0,1], virt=[2,3,4]) — small enough to run fast.
# Freezing orbital 0 (1s core) is the natural choice throughout.


def _capture_wfn(target_fn: Any) -> tuple[dict[str, Any], Any]:
    """Return (store, patched_fn) that captures the last wfn returned by target_fn."""
    store: dict[str, Any] = {}
    original = target_fn

    def spy(mf: Any, method: str, **kwargs: Any) -> Any:
        wfn = original(mf, method, **kwargs)
        store["wfn"] = wfn
        return wfn

    return store, spy


class TestLocalizeNosWithFrozen:
    def test_output_shape_no_frozen(self, be_rhf: Any) -> None:
        result = localize_nos(be_rhf, "mp2")
        nao, nmo = be_rhf.mo_coeff.shape
        assert result.shape == (nao, nmo)

    def test_output_shape_with_frozen(self, be_rhf: Any) -> None:
        result = localize_nos(be_rhf, "mp2", frozen=[0])
        nao, nmo = be_rhf.mo_coeff.shape
        assert result.shape == (nao, nmo)

    def test_frozen_mask_matches_passed_frozen(self, be_rhf: Any) -> None:
        """_get_frozen_mask of the underlying wfn must mark exactly the passed indices."""
        frozen = [0]
        store, spy = _capture_wfn(build_correlated_wfn)

        with patch(
            "aegiss.localization.orbital_localization.build_correlated_wfn", spy
        ):
            localize_nos(be_rhf, "mp2", frozen=frozen)

        mask = store["wfn"].get_frozen_mask()
        assert not mask[0], "orbital 0 should be frozen"
        assert mask[1:].all(), "all other orbitals should be active"

    def test_no_frozen_all_orbitals_active(self, be_rhf: Any) -> None:
        store, spy = _capture_wfn(build_correlated_wfn)

        with patch(
            "aegiss.localization.orbital_localization.build_correlated_wfn", spy
        ):
            localize_nos(be_rhf, "mp2")

        mask = store["wfn"].get_frozen_mask()
        assert mask.all(), "no orbital should be frozen when frozen=None"


class TestLocalizeFnosWithFrozen:
    def test_output_shape_no_frozen(self, be_rhf: Any) -> None:
        result = localize_fnos(be_rhf, "mp2")
        nao, nmo = be_rhf.mo_coeff.shape
        assert result.shape == (nao, nmo)

    def test_output_shape_with_frozen(self, be_rhf: Any) -> None:
        result = localize_fnos(be_rhf, "mp2", frozen=[0])
        nao, nmo = be_rhf.mo_coeff.shape
        assert result.shape == (nao, nmo)

    def test_frozen_mask_matches_passed_frozen(self, be_rhf: Any) -> None:
        """_get_frozen_mask of the underlying wfn must mark exactly the passed indices."""
        frozen = [0]
        store, spy = _capture_wfn(build_correlated_wfn)

        with patch(
            "aegiss.localization.orbital_localization.build_correlated_wfn", spy
        ):
            localize_fnos(be_rhf, "mp2", frozen=frozen)

        mask = store["wfn"].get_frozen_mask()
        assert not mask[0], "orbital 0 should be frozen"
        assert mask[1:].all(), "all other orbitals should be active"

    def test_frozen_kwarg_equivalent_to_active_complement(self, be_rhf: Any) -> None:
        """frozen=[0] and active=[1] (its complement in Be's occ space) must agree."""
        # Be has occ=[0,1]; complement of frozen={0} is active={1}
        result_frozen = localize_fnos(be_rhf, "mp2", frozen=[0])
        result_active = localize_fnos(be_rhf, "mp2", active=[1])
        np.testing.assert_allclose(result_frozen, result_active)


class TestLocalizeDispatcherFrozen:
    def test_nos_frozen_via_localize(self, be_rhf: Any) -> None:
        result = localize("nos", be_rhf, "mp2", frozen=[0])
        nao, nmo = be_rhf.mo_coeff.shape
        assert result.shape == (nao, nmo)

    def test_fnos_frozen_via_localize(self, be_rhf: Any) -> None:
        result = localize("fnos", be_rhf, "mp2", frozen=[0])
        nao, nmo = be_rhf.mo_coeff.shape
        assert result.shape == (nao, nmo)
