"""Shared pytest fixtures for AEGISS tests."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from aegiss.initial_guess import build_mol, run_scf


@pytest.fixture
def mock_mf_7mo() -> MagicMock:
    """Mock mean-field object with 7 MOs (4 occ, 3 virt)."""
    mf = MagicMock()
    mf.mo_occ = np.array([2.0, 2.0, 2.0, 2.0, 0.0, 0.0, 0.0])
    mf.mo_coeff = np.eye(7)
    mol = MagicMock()
    mol.nao = 7
    mf.mol = mol
    return mf


@pytest.fixture
def mock_mf_6mo() -> MagicMock:
    """Mock mean-field object with 6 MOs (3 occ, 3 virt)."""
    mf = MagicMock()
    mf.mo_occ = np.array([2.0, 2.0, 2.0, 0.0, 0.0, 0.0])
    mf.mo_coeff = np.eye(6)
    mol = MagicMock()
    mol.nao = 6
    mf.mol = mol
    return mf


@pytest.fixture(scope="session")
def h2_rhf():
    """H2 RHF/sto-3g mean-field object — computed once per session."""
    mol = build_mol("H 0 0 0; H 0 0 0.74", basis="sto-3g", symmetry=False, verbose=0)
    return run_scf(mol, method="RHF", verbose=0)
