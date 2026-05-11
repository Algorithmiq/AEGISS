"""Shared pytest fixtures for AEGISS tests."""

from __future__ import annotations

from typing import Any
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
def h2_rhf() -> Any:
    """H2 RHF/sto-3g mean-field object — computed once per session."""
    mol = build_mol("H 0 0 0; H 0 0 0.74", basis="sto-3g", symmetry=False, verbose=0)
    return run_scf(mol, method="RHF", verbose=0)


@pytest.fixture(scope="session")
def he_rhf() -> Any:
    """He atom RHF/sto-3g mean-field object."""
    mol = build_mol("He 0 0 0", basis="sto-3g", symmetry=False, verbose=0, spin=0)
    return run_scf(mol, method="RHF", verbose=0)


@pytest.fixture(scope="session")
def be_rhf() -> Any:
    """Be atom RHF/sto-3g mean-field object."""
    mol = build_mol("Be 0 0 0", basis="sto-3g", symmetry=False, verbose=0, spin=0)
    return run_scf(mol, method="RHF", verbose=0)


@pytest.fixture(scope="session")
def ne_rhf() -> Any:
    """Ne atom RHF/sto-3g mean-field object."""
    mol = build_mol("Ne 0 0 0", basis="sto-3g", symmetry=False, verbose=0, spin=0)
    return run_scf(mol, method="RHF", verbose=0)


@pytest.fixture(scope="session")
def c_sto3g_uhf() -> Any:
    """C atom UHF/sto-3g mean-field object (spin=2, triplet ground state)."""
    mol = build_mol("C 0 0 0", basis="sto-3g", symmetry=False, verbose=0, spin=2)
    return run_scf(mol, method="UHF", verbose=0)


@pytest.fixture(scope="session")
def o_sto3g_uhf() -> Any:
    """O atom UHF/sto-3g mean-field object (spin=2, triplet ground state)."""
    mol = build_mol("O 0 0 0", basis="sto-3g", symmetry=False, verbose=0, spin=2)
    return run_scf(mol, method="UHF", verbose=0)


@pytest.fixture(scope="session")
def c_631g_uhf() -> Any:
    """C atom UHF/6-31g mean-field object (spin=2, triplet ground state)."""
    mol = build_mol("C 0 0 0", basis="6-31g", symmetry=False, verbose=0, spin=2)
    return run_scf(mol, method="UHF", verbose=0)
