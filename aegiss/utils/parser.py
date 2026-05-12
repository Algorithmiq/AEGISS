"""Utilities for parsing geometry into atoms cluster given a tagged geometry string."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

def parse_atoms_tag(atoms_tag: str) -> tuple[str, list[tuple[int, str, str | None]]]:
    """Parse the geometry string to extract atom labels and coordinates.
    It returns a tuple containing the untagged geometry and a dictionary mapping cluster
    labels to lists of atom indices.
    The untagged geometry is `str` with the same format as the input but without the
    cluster assignments in the atom labels.
    The cluster assignment is a dictionary where the keys are cluster labels and the
    values are lists of atom indices that belong to that cluster.

    Parameters:
        atoms_tag (str):
            A string containing the geometry information, with each line representing
            an atom in the format `Element x y z [Cluster]`.
    Returns:
        tuple[str, list[tuple[int, str, str]]]:
            A tuple containing the untagged geometry string and a list of tuples with
            atom index, element symbol, and cluster assignment.

    Example:
        Given the following geometry string:
        ```
        C.1 0.000 0.000 0.000
        H.2 0.000 0.000 1.000
        O.3 1.000 0.000 0.000
        ```
        The function will return:
        [
            (0, 'C', '1'),
            (1, 'H', '2'),
            (2, 'O', '3')
        ]
    """

    atoms = []
    with open(atoms_tag, "r") as f:
        atoms_tag = f.read()
    untagged_lines = ""
    stripped_lines = atoms_tag.strip().splitlines()
    header = stripped_lines[:2]
    untagged_lines += "\n".join(header) + "\n"

    for index, line in enumerate(stripped_lines[2:]):
        parts = line.split()
        if len(parts) < 4:
            logger.warning(f"Line {index + 1} is malformed: '{line}'. Expected at least 4 parts.")
            continue
        element = parts[0].split('.')[0]
        cluster = parts[0].split('.')[1] if '.' in parts[0] else None
        untagged_line = f"{element} {' '.join(parts[1:4])}"
        untagged_lines += untagged_line + "\n"
        atoms.append((index, element, cluster))

    logger.info(f"Parsed {len(atoms)} atoms from the geometry string.")

    _print_cluster_assignment(atoms)

    return untagged_lines.strip(), atoms

def _print_cluster_assignment(atoms: list[tuple[int, str, str | None]]) -> None:
    """Print the cluster assignment in a readable format.

    Parameters:
        atoms (list[tuple[int, str, str | None]]):
            A list of tuples containing atom index, element symbol, and cluster assignment.
    """
    clusters: dict = dict()
    for index, element, cluster in atoms:
        if cluster not in clusters:
            clusters[cluster] = []
        clusters[cluster].append((index, element))

    logger.info("Cluster\tAtoms")
    for cluster, atom_list in clusters.items():
        atom_str = ", ".join(f"{element}({index})" for index, element in atom_list)
        logger.info(f"{cluster}\t{atom_str}")
