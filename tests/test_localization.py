"""Tests for orbital localization (Boys, PM, NOs/FNOs freezing, IBO/IAO, dispatcher aliases)."""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import numpy as np

from aegiss.localization.orbital_localization import (
    localize,
    localize_boys,
    localize_fnos,
    localize_iaos,
    localize_ibos,
    localize_nos,
    localize_pipek_mezey,
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


class TestLocalizeNos:
    def test_output_shape(self, be_rhf: Any) -> None:
        result = localize_nos(be_rhf, "mp2")
        nao, nmo = be_rhf.mo_coeff.shape
        assert result.shape == (nao, nmo)

    def test_all_orbitals_active(self, be_rhf: Any) -> None:
        store, spy = _capture_wfn(build_correlated_wfn)

        with patch(
            "aegiss.localization.orbital_localization.build_correlated_wfn", spy
        ):
            localize_nos(be_rhf, "mp2")

        mask = store["wfn"].get_frozen_mask()
        assert mask.all(), "NOs always use the full space — no orbital should be frozen"


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
    def test_fnos_frozen_via_localize(self, be_rhf: Any) -> None:
        result = localize("fnos", be_rhf, "mp2", frozen=[0])
        nao, nmo = be_rhf.mo_coeff.shape
        assert result.shape == (nao, nmo)


class TestLocalizeIbos:
    def test_output_shape(self, be_rhf: Any) -> None:
        mol = be_rhf.mol
        result = localize_ibos(mol, be_rhf)
        nao, nmo = be_rhf.mo_coeff.shape
        assert result.shape == (nao, nmo)

    def test_occupied_block_changes(self, h2o_rhf: Any) -> None:
        mol = h2o_rhf.mol
        n_occ = int(np.count_nonzero(h2o_rhf.mo_occ > 0))
        result = localize_ibos(mol, h2o_rhf)
        assert not np.allclose(result[:, :n_occ], h2o_rhf.mo_coeff[:, :n_occ])

    def test_virtual_block_unchanged(self, be_rhf: Any) -> None:
        mol = be_rhf.mol
        n_occ = int(np.count_nonzero(be_rhf.mo_occ > 0))
        result = localize_ibos(mol, be_rhf)
        np.testing.assert_array_equal(result[:, n_occ:], be_rhf.mo_coeff[:, n_occ:])

    def test_single_atom_returns_unchanged(self, be_rhf: Any) -> None:
        mol = be_rhf.mol
        result = localize_ibos(mol, be_rhf)
        np.testing.assert_array_equal(result, be_rhf.mo_coeff)

    def test_via_dispatcher_ibo_alias(self, be_rhf: Any) -> None:
        mol = be_rhf.mol
        nao, nmo = be_rhf.mo_coeff.shape
        assert localize("ibo", mol, be_rhf).shape == (nao, nmo)
        assert localize("ibos", mol, be_rhf).shape == (nao, nmo)


class TestLocalizeIaos:
    # H2O/sto-3g: 5 occupied MOs but 7 MINAO functions (O 1s2s2p + 2x H 1s), so an
    # IAO set the size of the occupied block would be wrong.

    def test_one_iao_per_minao_function(self, h2o_rhf: Any) -> None:
        mol = h2o_rhf.mol
        n_minao = mol.copy().build(basis="minao").nao
        result = localize_iaos(mol, h2o_rhf)
        assert result.shape == (mol.nao, n_minao)

    def test_iaos_are_orthonormal(self, h2o_rhf: Any) -> None:
        mol = h2o_rhf.mol
        result = localize_iaos(mol, h2o_rhf)
        s = mol.intor_symmetric("int1e_ovlp")
        np.testing.assert_allclose(
            result.T @ s @ result, np.eye(result.shape[1]), atol=1e-10
        )

    def test_iaos_span_occupied_space(self, h2o_rhf: Any) -> None:
        mol = h2o_rhf.mol
        n_occ = int(np.count_nonzero(h2o_rhf.mo_occ > 0))
        occ = h2o_rhf.mo_coeff[:, :n_occ]
        result = localize_iaos(mol, h2o_rhf)
        s = mol.intor_symmetric("int1e_ovlp")
        np.testing.assert_allclose(result @ result.T @ s @ occ, occ, atol=1e-10)

    def test_via_dispatcher_iao_alias(self, h2o_rhf: Any) -> None:
        mol = h2o_rhf.mol
        expected = localize_iaos(mol, h2o_rhf)
        np.testing.assert_allclose(localize("iao", mol, h2o_rhf), expected)
        np.testing.assert_allclose(localize("iaos", mol, h2o_rhf), expected)


class TestLocalizePipekMezey:
    def test_pop_method_applies_to_occupied_block(self, h2o_rhf: Any) -> None:
        mol = h2o_rhf.mol
        n_occ = int(np.count_nonzero(h2o_rhf.mo_occ > 0))
        with patch("aegiss.localization.orbital_localization.lo.PM") as mock_pm:
            mock_pm.return_value.kernel.side_effect = lambda: np.zeros((mol.nao, n_occ))
            localize_pipek_mezey(mol, h2o_rhf, pop_method="iao")
        assert mock_pm.call_args.kwargs["pop_method"] == "iao"

    def test_pop_method_applies_to_both_blocks(self, h2o_rhf: Any) -> None:
        mol = h2o_rhf.mol
        n_occ = int(np.count_nonzero(h2o_rhf.mo_occ > 0))
        nmo = h2o_rhf.mo_coeff.shape[1]
        with patch("aegiss.localization.orbital_localization.lo.PM") as mock_pm:
            mock_pm.return_value.kernel.side_effect = [
                np.zeros((mol.nao, n_occ)),
                np.zeros((mol.nao, nmo - n_occ)),
            ]
            localize_pipek_mezey(mol, h2o_rhf, on_virtual=True)
        assert [c.kwargs["pop_method"] for c in mock_pm.call_args_list] == [
            "mulliken",
            "mulliken",
        ]

    def test_occupied_block_is_orthonormal_and_virtual_unchanged(
        self, h2o_rhf: Any
    ) -> None:
        mol = h2o_rhf.mol
        n_occ = int(np.count_nonzero(h2o_rhf.mo_occ > 0))
        result = localize_pipek_mezey(mol, h2o_rhf)
        s = mol.intor_symmetric("int1e_ovlp")
        occ = result[:, :n_occ]
        np.testing.assert_allclose(occ.T @ s @ occ, np.eye(n_occ), atol=1e-8)
        np.testing.assert_array_equal(result[:, n_occ:], h2o_rhf.mo_coeff[:, n_occ:])


class TestLocalizeBoys:
    def test_occupied_block_is_orthonormal_and_virtual_unchanged(
        self, h2o_rhf: Any
    ) -> None:
        mol = h2o_rhf.mol
        n_occ = int(np.count_nonzero(h2o_rhf.mo_occ > 0))
        result = localize_boys(mol, h2o_rhf)
        assert result.shape == h2o_rhf.mo_coeff.shape
        s = mol.intor_symmetric("int1e_ovlp")
        occ = result[:, :n_occ]
        np.testing.assert_allclose(occ.T @ s @ occ, np.eye(n_occ), atol=1e-8)
        np.testing.assert_array_equal(result[:, n_occ:], h2o_rhf.mo_coeff[:, n_occ:])

    def test_on_virtual_localizes_virtual_block(self, h2o_rhf: Any) -> None:
        mol = h2o_rhf.mol
        n_occ = int(np.count_nonzero(h2o_rhf.mo_occ > 0))
        result = localize_boys(mol, h2o_rhf, on_virtual=True)
        s = mol.intor_symmetric("int1e_ovlp")
        np.testing.assert_allclose(
            result.T @ s @ result, np.eye(result.shape[1]), atol=1e-8
        )
        # The rotation stays inside the virtual space.
        virt = h2o_rhf.mo_coeff[:, n_occ:]
        proj = virt @ virt.T @ s
        np.testing.assert_allclose(
            proj @ result[:, n_occ:], result[:, n_occ:], atol=1e-8
        )
