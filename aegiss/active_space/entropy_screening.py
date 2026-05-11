"""Active space selection via single-orbital entropy screening."""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np

logger = logging.getLogger(__name__)
_ORB_S1_FMT = "\tOrbital %d: S(1) = %.4f"


def _rank_and_threshold(
    soe1: np.ndarray,
    orbitals_index: np.ndarray,
    fraction: float,
    fraction_type: Literal["r", "a"],
) -> list[tuple[int, float]]:
    """Sort orbitals by entropy, apply a threshold, and return selected pairs.

    Args:
        soe1 (np.ndarray): Single-orbital entropies for the orbitals in *orbitals_index*.
        orbitals_index (np.ndarray): Orbital indices corresponding to *soe1*.
        fraction (float): Threshold value.
        fraction_type ({'r', 'a'}): Relative (normalise by max before thresholding) or absolute.

    Returns:
        selected (list of tuple[int, float]): Orbital index / (possibly normalised) entropy pairs above the threshold,
            sorted by descending entropy.
    """
    max_s1 = float(np.max(soe1))
    logger.debug("S(1) values for candidate orbitals:")
    for orb, s1 in zip(orbitals_index, soe1, strict=False):
        logger.debug(_ORB_S1_FMT, orb, s1)
    logger.debug(
        "MAX S(1): %.4f (orbital %d)\n", max_s1, orbitals_index[np.argmax(soe1)]
    )

    if fraction_type == "r":
        logger.info("Using relative threshold")
        if max_s1 == 0.0:
            logger.info(
                "All candidate entropies are zero; keeping relative entropies at 0.0."
            )
            entropies = np.zeros_like(soe1, dtype=float)
        else:
            entropies = soe1 / max_s1
    else:
        logger.info("Using absolute threshold")
        entropies = soe1

    sortkey = np.argsort(entropies)[::-1]
    entropies = entropies[sortkey]
    sorted_indices = orbitals_index[sortkey]
    for orb, s1 in zip(sorted_indices, entropies, strict=False):
        logger.debug(_ORB_S1_FMT, orb, s1)

    stop_idx = next((k for k, s in enumerate(entropies) if s < fraction), None)
    if stop_idx is None:
        logger.info(
            "All %d orbitals are selected by the entropy threshold %g.",
            len(soe1),
            fraction,
        )
    else:
        entropies = entropies[:stop_idx]
        sorted_indices = sorted_indices[:stop_idx]
        logger.info(
            "\nSelected %d orbitals by entropy threshold %g (fraction type: %s).",
            len(sorted_indices),
            fraction,
            fraction_type,
        )

    return list(zip(sorted_indices.tolist(), entropies.tolist(), strict=False))


def _global_entropy_ranking(
    soe1: np.ndarray,
    fraction_type: Literal["r", "a"],
    fraction: float,
    occ_orbs: np.ndarray,
    virt_orbs: np.ndarray,
) -> tuple[list[tuple[int, float]], list[tuple[int, float]]]:
    """Rank all orbitals globally by their single-orbital entropy.

    Args:
        soe1 (np.ndarray): Single-orbital entropies for all orbitals.
        fraction_type ({'r', 'a'}): Relative or absolute threshold type.
        fraction (float): Threshold value.
        occ_orbs (np.ndarray): Indices of occupied orbitals.
        virt_orbs (np.ndarray): Indices of virtual orbitals.

    Returns:
        occ_selected (list of tuple[int, float]): Selected occupied orbitals with their entropy values.
        virt_selected (list of tuple[int, float]): Selected virtual orbitals with their entropy values.
    """
    orbitals_index = np.concatenate((occ_orbs, virt_orbs))
    selected = _rank_and_threshold(soe1, orbitals_index, fraction, fraction_type)
    occ_set = set(occ_orbs.tolist())
    virt_set = set(virt_orbs.tolist())
    occ_selected = [(orb, s1) for orb, s1 in selected if orb in occ_set]
    virt_selected = [(orb, s1) for orb, s1 in selected if orb in virt_set]
    return occ_selected, virt_selected


def _relative_entropy_ranking(
    soe1: np.ndarray,
    fraction_type: Literal["r", "a"],
    fraction: float,
    subspace: np.ndarray,
) -> list[tuple[int, float]]:
    """Rank orbitals within a subspace by their single-orbital entropy.

    Args:
        soe1 (np.ndarray): Single-orbital entropies for the subspace.
        fraction_type ({'r', 'a'}): Relative or absolute threshold type.
        fraction (float): Threshold value.
        subspace (np.ndarray): Orbital indices in the subspace.

    Returns:
        selected (list of tuple[int, float]): Selected orbitals with their (possibly normalised) entropy values.
    """
    return _rank_and_threshold(soe1, subspace, fraction, fraction_type)


def select_by_entropy(
    occ_orbs: np.ndarray,
    virt_orbs: np.ndarray,
    soe1: np.ndarray | None,
    fraction: float = 0.1,
    fraction_type: Literal["r", "a"] = "r",
    ranking: Literal["global", "subset"] = "global",
) -> tuple[list[tuple[int, float]], list[tuple[int, float]]]:
    """Select active orbitals by single-orbital entanglement entropy.

    Args:
        occ_orbs (np.ndarray, shape (n_occ,)): Indices of occupied orbitals.
        virt_orbs (np.ndarray, shape (n_virt,)): Indices of virtual orbitals.
        soe1 (np.ndarray, shape (n_occ + n_virt,)): Single-orbital entropies for all orbitals.
        fraction (float, optional): Threshold on the entropy value. Interpretation depends on *fraction_type*.
        fraction_type ({'r', 'a'}, optional): ``'r'`` normalises entropy values by the maximum before thresholding;
            ``'a'`` applies *fraction* as an absolute threshold.
        ranking ({'global', 'subset'}, optional): ``'global'`` ranks all orbitals together; ``'subset'`` ranks occupied
            and virtual subspaces separately.

    Returns:
        selected_occ (list of tuple[int, float]): Selected occupied orbitals with entropy values.
        selected_virt (list of tuple[int, float]): Selected virtual orbitals with entropy values.
    """
    logger.info("----------------Entropy Screening for AEGISS----------------\n")
    if soe1 is None:
        logger.info("No single-orbital entropies provided. Returning empty selection.")
        return [], []
    if ranking == "global":
        logger.info("Using global ranking of orbitals by entropy values.")
        selected_occ, selected_virt = _global_entropy_ranking(
            soe1, fraction_type, fraction, occ_orbs, virt_orbs
        )
    else:
        logger.info(
            "Using relative ranking of orbitals by entropy values within the subspace."
        )
        n_occ = len(occ_orbs)
        selected_occ = _relative_entropy_ranking(
            soe1[:n_occ], fraction_type, fraction, occ_orbs
        )
        selected_virt = _relative_entropy_ranking(
            soe1[n_occ:], fraction_type, fraction, virt_orbs
        )

    for i in selected_occ:
        logger.info("\tSelected occupied orbital %d: S(1) = %.4f", i[0], i[1])
    for i in selected_virt:
        logger.info("\tSelected virtual orbital %d:  S(1) = %.4f", i[0], i[1])

    return selected_occ, selected_virt
