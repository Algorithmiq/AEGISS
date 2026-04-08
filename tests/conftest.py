"""Shared pytest fixtures for AEGISS tests."""
from __future__ import annotations

import types
import numpy as np
import pytest


@pytest.fixture
def mock_mf_7mo():
    """Mock SCF object for a 7-MO system: 5 occupied, 2 virtual (water STO-3G-like)."""
    mf = types.SimpleNamespace()
    mf.mo_occ = np.array([2., 2., 2., 2., 2., 0., 0.])
    mol = types.SimpleNamespace()
    mol.nao = 7
    mol.tot_electrons = lambda: 10
    mf.mol = mol
    return mf


@pytest.fixture
def mock_mf_6mo():
    """Mock SCF object: 3 occ, 3 virt."""
    mf = types.SimpleNamespace()
    mf.mo_occ = np.array([2., 2., 2., 0., 0., 0.])
    mol = types.SimpleNamespace()
    mol.nao = 6
    mol.tot_electrons = lambda: 6
    mf.mol = mol
    return mf


@pytest.fixture(scope="session")
def h2_rhf():
    """Actual RHF/STO-3G calculation on H2 for PySCF integration tests."""
    from aegiss.initial_guess import build_mol, run_scf
    mol = build_mol("H 0 0 0; H 0 0 0.74", basis="sto-3g", verbose=0)
    mf = run_scf(mol, method="RHF")
    return mol, mf
