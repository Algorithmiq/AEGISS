"""Atomic data: minimal basis sets, bond detection, and tagged XYZ I/O."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from ase import Atoms
from ase.data import covalent_radii

if TYPE_CHECKING:
    import ase.atoms

BOND_TOLERANCE: float = 1.3

MINIMAL_BASIS: dict[int, dict[str, list[str]]] = {
    1:  {"all": ["1s"],                                                        "valence": ["1s"]},
    2:  {"all": ["1s"],                                                        "valence": ["1s"]},
    3:  {"all": ["1s", "2s"],                                                  "valence": ["2s"]},
    4:  {"all": ["1s", "2s"],                                                  "valence": ["2s"]},
    5:  {"all": ["1s", "2s", "2p"],                                            "valence": ["2s", "2p"]},
    6:  {"all": ["1s", "2s", "2p"],                                            "valence": ["2s", "2p"]},
    7:  {"all": ["1s", "2s", "2p"],                                            "valence": ["2s", "2p"]},
    8:  {"all": ["1s", "2s", "2p"],                                            "valence": ["2s", "2p"]},
    9:  {"all": ["1s", "2s", "2p"],                                            "valence": ["2s", "2p"]},
    10: {"all": ["1s", "2s", "2p"],                                            "valence": ["2s", "2p"]},
    11: {"all": ["1s", "2s", "2p", "3s"],                                      "valence": ["3s"]},
    12: {"all": ["1s", "2s", "2p", "3s"],                                      "valence": ["3s"]},
    13: {"all": ["1s", "2s", "2p", "3s", "3p"],                                "valence": ["3s", "3p"]},
    14: {"all": ["1s", "2s", "2p", "3s", "3p"],                                "valence": ["3s", "3p"]},
    15: {"all": ["1s", "2s", "2p", "3s", "3p"],                                "valence": ["3s", "3p"]},
    16: {"all": ["1s", "2s", "2p", "3s", "3p"],                                "valence": ["3s", "3p"]},
    17: {"all": ["1s", "2s", "2p", "3s", "3p"],                                "valence": ["3s", "3p"]},
    18: {"all": ["1s", "2s", "2p", "3s", "3p"],                                "valence": ["3s", "3p"]},
    19: {"all": ["1s", "2s", "2p", "3s", "3p", "4s"],                          "valence": ["4s"]},
    20: {"all": ["1s", "2s", "2p", "3s", "3p", "4s"],                          "valence": ["4s"]},
    21: {"all": ["1s", "2s", "2p", "3s", "3p", "3d", "4s"],                    "valence": ["3d", "4s"]},
    22: {"all": ["1s", "2s", "2p", "3s", "3p", "3d", "4s"],                    "valence": ["3d", "4s"]},
    23: {"all": ["1s", "2s", "2p", "3s", "3p", "3d", "4s"],                    "valence": ["3d", "4s"]},
    24: {"all": ["1s", "2s", "2p", "3s", "3p", "3d", "4s"],                    "valence": ["3d", "4s"]},
    25: {"all": ["1s", "2s", "2p", "3s", "3p", "3d", "4s"],                    "valence": ["3d", "4s"]},
    26: {"all": ["1s", "2s", "2p", "3s", "3p", "3d", "4s"],                    "valence": ["3d", "4s"]},
    27: {"all": ["1s", "2s", "2p", "3s", "3p", "3d", "4s"],                    "valence": ["3d", "4s"]},
    28: {"all": ["1s", "2s", "2p", "3s", "3p", "3d", "4s"],                    "valence": ["3d", "4s"]},
    29: {"all": ["1s", "2s", "2p", "3s", "3p", "3d", "4s"],                    "valence": ["3d", "4s"]},
    30: {"all": ["1s", "2s", "2p", "3s", "3p", "3d", "4s"],                    "valence": ["3d", "4s"]},
    31: {"all": ["1s", "2s", "2p", "3s", "3p", "3d", "4s", "4p"],              "valence": ["4s", "4p"]},
    32: {"all": ["1s", "2s", "2p", "3s", "3p", "3d", "4s", "4p"],              "valence": ["4s", "4p"]},
    33: {"all": ["1s", "2s", "2p", "3s", "3p", "3d", "4s", "4p"],              "valence": ["4s", "4p"]},
    34: {"all": ["1s", "2s", "2p", "3s", "3p", "3d", "4s", "4p"],              "valence": ["4s", "4p"]},
    35: {"all": ["1s", "2s", "2p", "3s", "3p", "3d", "4s", "4p"],              "valence": ["4s", "4p"]},
    36: {"all": ["1s", "2s", "2p", "3s", "3p", "3d", "4s", "4p"],              "valence": ["4s", "4p"]},
}


def _guess_basis(Z: int) -> dict[str, list[str]]:
    """Generate a rough minimal basis via aufbau for elements beyond Kr (Z=36).

    Args:
        Z: Atomic number.

    Returns:
        Dictionary with ``"all"`` and ``"valence"`` shell lists.
    """
    order = [
        (1, "s"), (2, "s"), (2, "p"), (3, "s"), (3, "p"), (4, "s"),
        (3, "d"), (4, "p"), (5, "s"), (4, "d"), (5, "p"), (6, "s"),
        (4, "f"), (5, "d"), (6, "p"), (7, "s"), (5, "f"), (6, "d"), (7, "p"),
    ]
    max_e: dict[str, int] = {"s": 2, "p": 6, "d": 10, "f": 14}
    remaining = Z
    shells: list[str] = []
    last_n = 0
    for n, l_qn in order:
        if remaining <= 0:
            break
        shells.append(f"{n}{l_qn}")
        remaining -= min(remaining, max_e[l_qn])
        last_n = max(last_n, n)
    valence = [
        s for s in shells
        if int(s[0]) >= last_n - 1 and (s[1] != "s" or int(s[0]) == last_n)
    ]
    if not valence:
        valence = shells[-2:] if len(shells) >= 2 else shells[:]
    return {"all": shells, "valence": valence}


def get_basis(Z: int) -> dict[str, list[str]]:
    """Return the minimal basis definition for atomic number *Z*.

    Args:
        Z: Atomic number.

    Returns:
        Dictionary with ``"all"`` (complete shell list) and ``"valence"``
        (valence-only shell list) keys.
    """
    return MINIMAL_BASIS.get(Z, _guess_basis(Z))


def detect_bonds(atoms: ase.atoms.Atoms) -> list[tuple[int, int]]:
    """Detect covalent bonds using scaled sums of covalent radii.

    Args:
        atoms: ASE Atoms object.

    Returns:
        List of ``(i, j)`` index pairs where a bond is detected.
    """
    pos = atoms.get_positions()
    nums = atoms.get_atomic_numbers()
    n = len(atoms)
    bonds: list[tuple[int, int]] = []
    for i in range(n):
        ri = covalent_radii[nums[i]]
        for j in range(i + 1, n):
            dist = float(np.linalg.norm(pos[i] - pos[j]))
            if 0.4 < dist < BOND_TOLERANCE * (ri + covalent_radii[nums[j]]):
                bonds.append((i, j))
    return bonds


def load_xyz(filepath: str | Path) -> tuple[Atoms, dict[int, str]]:
    """Read an XYZ file and recover any embedded ``Element.N`` tags.

    Handles both plain XYZ files and files previously saved by MolTagger,
    where atom labels may be ``Element.N`` (e.g. ``H.4``). The ``Element``
    part is used to build the ASE ``Atoms`` object; the full label is stored
    as the tag so it can be restored in the viewer.

    Args:
        filepath: Path to the XYZ file.

    Returns:
        Tuple of ``(atoms, tags)`` where *tags* maps atom indices to their
        original ``Element.N`` label (empty dict if no tags are present).
    """
    lines = Path(filepath).read_text().splitlines()
    n = int(lines[0].strip())
    symbols: list[str] = []
    positions: list[tuple[float, float, float]] = []
    tags: dict[int, str] = {}
    for idx, line in enumerate(lines[2 : 2 + n]):
        parts = line.split()
        label = parts[0]
        pos = (float(parts[1]), float(parts[2]), float(parts[3]))
        if "." in label:
            element, _ = label.split(".", 1)
            symbols.append(element)
            tags[idx] = label
        else:
            symbols.append(label)
        positions.append(pos)
    return Atoms(symbols=symbols, positions=positions), tags
