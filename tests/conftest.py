"""Shared pytest fixtures for AEGISS tests."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from aegiss.initial_guess import build_mol, run_scf
from aegiss.utils import orb_display

if TYPE_CHECKING:
    from collections.abc import Iterator


@pytest.fixture(autouse=True)
def _default_orbital_base(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[None]:
    """Isolate every test from the machine's real orbital-base configuration.

    ``orb_display.orbital_base()`` resolves from ``AEGISS_ORB_BASE`` and the
    persisted ``~/.config/aegiss/config.toml``. Without this, a developer who
    ran ``aegiss config set-base 1`` would see unrelated tests fail on 0-based
    assertions. Point ``XDG_CONFIG_HOME`` at an empty dir, clear the env var,
    and reset the cache so the base defaults to 0 unless a test opts in.
    """
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path_factory.mktemp("xdg")))
    monkeypatch.delenv("AEGISS_ORB_BASE", raising=False)
    orb_display.reset_cache()
    yield
    orb_display.reset_cache()


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
def h2o_rhf() -> Any:
    """H2O RHF/sto-3g mean-field object — 5 occupied MOs for localization tests."""
    mol = build_mol(
        "O 0 0 0; H 0 0.757 0.586; H 0 -0.757 0.586",
        basis="sto-3g",
        symmetry=False,
        verbose=0,
    )
    return run_scf(mol, method="RHF", verbose=0)


@pytest.fixture(scope="session")
def ne_rhf() -> Any:
    """Ne atom RHF/sto-3g mean-field object."""
    mol = build_mol("Ne 0 0 0", basis="sto-3g", symmetry=False, verbose=0, spin=0)
    return run_scf(mol, method="RHF", verbose=0)


@pytest.fixture(scope="session")
def ne_x2c_rhf() -> Any:
    """Ne atom X2C-RHF/sto-3g mean-field object."""
    mol = build_mol("Ne 0 0 0", basis="sto-3g", symmetry=False, verbose=0, spin=0)
    return run_scf(mol, method="RHF", verbose=0, relativistic="x2c")


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
