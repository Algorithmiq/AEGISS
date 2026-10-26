"""Utilities for saving molecular orbitals and intermediate DMRG quantities to disk."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from aegiss.utils.orb_display import to_display

logger = logging.getLogger(__name__)


def save_orbitals(
    mol: Any,
    mo_coeff: np.ndarray,
    candidate_indices: np.ndarray,
    filename_prefix: str = "mo",
    folder: str = "./",
) -> None:
    """Save molecular orbitals as cube files for visualization.

    Args:
        mol (pyscf.gto.Mole): Molecule object containing the geometry and basis set information.
        mo_coeff (np.ndarray): Molecular orbital coefficients in the AO basis. Shape (n_ao, n_mo).
        candidate_indices (np.ndarray): Indices of the orbitals to save.
        filename_prefix (str): Prefix for the output cube files. Each orbital is saved as
            ``"{filename_prefix}_{i}.cube"``, where ``i`` is the orbital index
            in the configured display base (see
            :func:`aegiss.utils.orb_display.to_display`).
        folder (str): Output directory path. Created if it does not exist.
    """
    from pyscf.tools import cubegen  # noqa: PLC0415

    output_dir = Path(folder)
    output_dir.mkdir(parents=True, exist_ok=True)

    for i in candidate_indices:
        # File name carries the display-base index; slicing stays 0-based.
        filename = output_dir / f"{filename_prefix}_{to_display(i)}.cube"
        cubegen.orbital(mol, str(filename), mo_coeff[:, i])


def save_orbitals_from_entropies(
    mol: Any,
    mo_coeff: np.ndarray,
    candidate_indices: list[tuple[int, float]],
    filename_prefix: str = "mo",
    folder: str = "./",
) -> None:
    """Save molecular orbitals (selected by entropy) as cube files.

    Args:
        mol (pyscf.gto.Mole):
            Molecule object containing the geometry and basis set information.
        mo_coeff (np.ndarray):
            Molecular orbital coefficients in the AO basis. Shape (n_ao, n_mo).
        candidate_indices (list of tuple[int, float]):
            Orbital index/entropy pairs; only the indices are used for saving.
        filename_prefix (str):
            Prefix for the output cube files. Each orbital is saved as ``"{filename_prefix}_{i}.cube"``,
            where ``i`` is the orbital index in the configured display base (see
            :func:`aegiss.utils.orb_display.to_display`).
        folder (str):
            Output directory path. Created if it does not exist.
    """
    from pyscf.tools import cubegen  # noqa: PLC0415

    output_dir = Path(folder)
    output_dir.mkdir(parents=True, exist_ok=True)

    for i, _ in candidate_indices:
        # File name carries the display-base index; slicing stays 0-based.
        filename = output_dir / f"{filename_prefix}_{to_display(i)}.cube"
        cubegen.orbital(mol, str(filename), mo_coeff[:, i])


def dump_qmi(
    minfo: np.ndarray,
    soe1: np.ndarray,
    soe2: np.ndarray | None = None,
    path: Path | str = "./qmi_data.npz",
) -> None:
    """Dump QMI, SOE1, and optional SOE2 data to disk.

    The arrays are stored under the keys ``"minfo"``, ``"soe1"`` and ``"soe2"``
    and can be read back with ``np.load``. The AEGISS pipeline does not compute
    mutual information, so *minfo* comes from an external calculation (see
    :func:`aegiss.utils.plotting.mi_plot`).

    Args:
        minfo (np.ndarray): Mutual information matrix.
        soe1 (np.ndarray): Single orbital entropy values.
        soe2 (np.ndarray or None): Double orbital entropy values, or ``None`` if not computed.
        path (Path or str): Output file path. numpy appends ``.npz`` when the
            path does not already end with it.
    """
    filename = Path(path)
    data: dict[str, np.ndarray] = {"minfo": minfo, "soe1": soe1}
    if soe2 is not None:
        data["soe2"] = soe2
    np.savez(filename, **data)  # ty: ignore[invalid-argument-type]
    logger.info("QMI data dumped to %s", filename)
