"""Tests for aegiss.wavefunction.methods (MP2 / CISD / CCSD runners and dispatcher)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from pyscf import cc, ci, fci, mp

from aegiss.wavefunction.methods import (
    build_correlated_wfn,
    run_ccsd,
    run_cisd,
    run_mp2,
)


def _fci_energy(mf: Any) -> float:
    return float(fci.FCI(mf).kernel()[0])


class TestRunMp2:
    def test_matches_pyscf(self, h2o_rhf: Any) -> None:
        reference = mp.MP2(h2o_rhf).run()
        assert run_mp2(h2o_rhf).e_corr == pytest.approx(reference.e_corr, abs=1e-10)

    def test_correlation_energy_is_negative(self, h2o_rhf: Any) -> None:
        assert run_mp2(h2o_rhf).e_corr < 0.0

    def test_kwargs_are_set_on_solver(self, h2o_rhf: Any) -> None:
        full = run_mp2(h2o_rhf)
        frozen = run_mp2(h2o_rhf, frozen=[0])
        assert frozen.frozen == [0]
        # Freezing the O 1s core removes part of the correlation energy.
        assert frozen.e_corr > full.e_corr

    def test_rdm1_traces_to_electron_count(self, h2o_rhf: Any) -> None:
        rdm1 = run_mp2(h2o_rhf).make_rdm1()
        assert np.trace(rdm1) == pytest.approx(h2o_rhf.mol.nelectron, abs=1e-8)


class TestRunCisd:
    def test_converges_and_matches_pyscf(self, h2o_rhf: Any) -> None:
        reference = ci.CISD(h2o_rhf).run()
        result = run_cisd(h2o_rhf)
        assert result.converged
        assert result.e_corr == pytest.approx(reference.e_corr, abs=1e-8)

    def test_exact_for_two_electrons(self, h2_rhf: Any) -> None:
        # With two electrons CISD spans the full CI space.
        assert run_cisd(h2_rhf).e_tot == pytest.approx(_fci_energy(h2_rhf), abs=1e-8)

    def test_kwargs_are_set_on_solver(self, h2o_rhf: Any) -> None:
        assert run_cisd(h2o_rhf, frozen=[0]).frozen == [0]

    def test_rdm1_traces_to_electron_count(self, h2o_rhf: Any) -> None:
        rdm1 = run_cisd(h2o_rhf).make_rdm1()
        assert np.trace(rdm1) == pytest.approx(h2o_rhf.mol.nelectron, abs=1e-8)


class TestRunCcsd:
    def test_converges_and_matches_pyscf(self, h2o_rhf: Any) -> None:
        reference = cc.CCSD(h2o_rhf).run()
        result = run_ccsd(h2o_rhf)
        assert result.converged
        assert result.e_corr == pytest.approx(reference.e_corr, abs=1e-8)

    def test_exact_for_two_electrons(self, h2_rhf: Any) -> None:
        # With two electrons CCSD is equivalent to full CI. PySCF's default
        # conv_tol (1e-7) leaves a ~1e-7 Ha gap, so tighten it through **kwargs.
        result = run_ccsd(h2_rhf, conv_tol=1e-12, conv_tol_normt=1e-10)
        assert result.e_tot == pytest.approx(_fci_energy(h2_rhf), abs=1e-8)

    def test_kwargs_are_set_on_solver(self, h2o_rhf: Any) -> None:
        assert run_ccsd(h2o_rhf, frozen=[0]).frozen == [0]


class TestBuildCorrelatedWfn:
    @pytest.mark.parametrize(
        ("method", "solver_type"),
        [("mp2", mp.mp2.MP2), ("cisd", ci.cisd.CISD), ("ccsd", cc.ccsd.CCSD)],
    )
    def test_dispatches_to_solver(
        self, h2o_rhf: Any, method: str, solver_type: type
    ) -> None:
        assert isinstance(build_correlated_wfn(h2o_rhf, method), solver_type)

    def test_forwards_kwargs(self, h2o_rhf: Any) -> None:
        assert build_correlated_wfn(h2o_rhf, "mp2", frozen=[0]).frozen == [0]

    def test_unknown_method_raises(self, h2o_rhf: Any) -> None:
        with pytest.raises(ValueError, match="Unsupported method"):
            build_correlated_wfn(h2o_rhf, "fci")  # type: ignore[arg-type]
