"""Utilities for saving molecular orbitals as cube files."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from pyscf.tools import cubegen

if TYPE_CHECKING:
    import numpy as np


def save_orbitals(
    mol: Any,
    mo_coeff: np.ndarray,
    candidate_indices: np.ndarray,
    filename_prefix: str = "mo",
    folder: str = "./",
) -> None:
    """Save molecular orbitals as cube files for visualization.

    Parameters
    ----------
    mol : pyscf.gto.Mole
        Molecule object containing the geometry and basis set information.
    mo_coeff : np.ndarray
        Molecular orbital coefficients in the AO basis. Shape (n_ao, n_mo).
    candidate_indices : np.ndarray
        Indices of the orbitals to save.
    filename_prefix : str
        Prefix for the output cube files. Each orbital is saved as
        ``"{filename_prefix}_{i}.cube"``.
    folder : str
        Output directory path. Created if it does not exist.
    """
    output_dir = Path(folder)
    output_dir.mkdir(parents=True, exist_ok=True)

    for i in candidate_indices:
        filename = output_dir / f"{filename_prefix}_{i}.cube"
        cubegen.orbital(mol, str(filename), mo_coeff[:, i])


def save_orbitals_from_entropies(
    mol: Any,
    mo_coeff: np.ndarray,
    candidate_indices: list[tuple[int, float]],
    filename_prefix: str = "mo",
    folder: str = "./",
) -> None:
    """Save molecular orbitals (selected by entropy) as cube files.

    Parameters
    ----------
    mol : pyscf.gto.Mole
        Molecule object containing the geometry and basis set information.
    mo_coeff : np.ndarray
        Molecular orbital coefficients in the AO basis. Shape (n_ao, n_mo).
    candidate_indices : list of tuple[int, float]
        Orbital index/entropy pairs; only the indices are used for saving.
    filename_prefix : str
        Prefix for the output cube files. Each orbital is saved as
        ``"{filename_prefix}_{i}.cube"``.
    folder : str
        Output directory path. Created if it does not exist.
    """
    output_dir = Path(folder)
    output_dir.mkdir(parents=True, exist_ok=True)

    for i, _ in candidate_indices:
        filename = output_dir / f"{filename_prefix}_{i}.cube"
        cubegen.orbital(mol, str(filename), mo_coeff[:, i])
