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

    Parameters
    ----------
    soe1 : np.ndarray
        Single-orbital entropies for the orbitals in *orbitals_index*.
    orbitals_index : np.ndarray
        Orbital indices corresponding to *soe1*.
    fraction : float
        Threshold value.
    fraction_type : {'r', 'a'}
        Relative (normalise by max before thresholding) or absolute.

    Returns:
    -------
    selected : list of tuple[int, float]
        Orbital index / (possibly normalised) entropy pairs above the threshold,
        sorted by descending entropy.
    """
    max_s1 = float(np.max(soe1))
    print("S(1) values for candidate orbitals:")
    for orb, s1 in zip(orbitals_index, soe1, strict=False):
        print(f"\tOrbital {orb}: S(1) = {s1:.4f}")
    print(f"MAX S(1): {max_s1:.4f} (orbital {orbitals_index[np.argmax(soe1)]})\n")

    if fraction_type == "r":
        print("Using relative threshold")
        if max_s1 == 0.0:
            print(
                "All candidate entropies are zero; keeping relative entropies at 0.0."
            )
            entropies = np.zeros_like(soe1, dtype=float)
        else:
            entropies = soe1 / max_s1
    else:
        print("Using absolute threshold")
        entropies = soe1

    sortkey = np.argsort(entropies)[::-1]
    entropies = entropies[sortkey]
    sorted_indices = orbitals_index[sortkey]
    for orb, s1 in zip(sorted_indices, entropies, strict=False):
        print(f"\tOrbital {orb}: S(1) = {s1:.4f}")

    stop_idx = next((k for k, s in enumerate(entropies) if s < fraction), None)
    if stop_idx is None:
        print(
            f"All {len(soe1)} orbitals are selected by the entropy threshold {fraction}."
        )
    else:
        entropies = entropies[:stop_idx]
        sorted_indices = sorted_indices[:stop_idx]
        print(
            f"\nSelected {len(sorted_indices)} orbitals by entropy threshold"
            f" {fraction} (fraction type: {fraction_type})."
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

    Parameters
    ----------
    soe1 : np.ndarray
        Single-orbital entropies for all orbitals.
    fraction_type : {'r', 'a'}
        Relative or absolute threshold type.
    fraction : float
        Threshold value.
    occ_orbs : np.ndarray
        Indices of occupied orbitals.
    virt_orbs : np.ndarray
        Indices of virtual orbitals.

    Returns:
    -------
    occ_selected : list of tuple[int, float]
        Selected occupied orbitals with their entropy values.
    virt_selected : list of tuple[int, float]
        Selected virtual orbitals with their entropy values.
    """
    orbitals_index = np.concatenate((occ_orbs, virt_orbs))
    ths_lists = soe1
    logger.debug("S(1) values for candidate orbitals:")
    for idx, s1 in zip(orbitals_index, ths_lists, strict=False):
        logger.debug(_ORB_S1_FMT, idx, s1)

    logger.debug(
        "MAX S(1): %.4f (orbital %d)\n", max_s1, orbitals_index[np.argmax(soe1)]
    )

    if fraction_type == "r":
        logger.info("Using relative threshold")
        ths_lists = ths_lists / max_s1
    else:
        logger.info("Using absolute threshold")
    sortkey = np.argsort(ths_lists)[::-1]
    ths_lists = ths_lists[sortkey]
    orbitals_index = orbitals_index[sortkey]
    for idx, s1 in zip(orbitals_index, ths_lists, strict=False):
        logger.debug(_ORB_S1_FMT, idx, s1)

    stop_idx = None
    for k, ths in enumerate(ths_lists):
        if ths < fraction:
            stop_idx = k
            break
    if stop_idx is None:
        logger.info(
            "All %d orbitals are selected by the entropy threshold %g.",
            n_orbs,
            fraction,
        )
    else:
        ths_lists = ths_lists[:stop_idx]
        orbitals_index = orbitals_index[:stop_idx]
        logger.info(
            "Selected %d orbitals by entropy threshold %g (fraction type: %s).",
            len(orbitals_index),
            fraction,
            fraction_type,
        )

    occ_selected = []
    virt_selected = []
    for idx, n_orb in enumerate(orbitals_index):
        if n_orb in occ_orbs:
            occ_selected.append((n_orb, ths_lists[idx]))
        elif n_orb in virt_orbs:
            virt_selected.append((n_orb, ths_lists[idx]))

    return occ_selected, virt_selected


def _relative_entropy_ranking(
    soe1: np.ndarray,
    fraction_type: Literal["r", "a"],
    fraction: float,
    subspace: np.ndarray,
) -> list[tuple[int, float]]:
    """Rank orbitals within a subspace by their single-orbital entropy.

    Parameters
    ----------
    soe1 : np.ndarray
        Single-orbital entropies for the subspace.
    fraction_type : {'r', 'a'}
        Relative or absolute threshold type.
    fraction : float
        Threshold value.
    subspace : np.ndarray
        Orbital indices in the subspace.

    Returns:
    -------
    selected : list of tuple[int, float]
        Selected orbitals with their (possibly normalised) entropy values.
    """
    max_s1 = np.max(soe1)

    n_orbs = len(soe1)

    ths_lists = soe1
    logger.debug("S(1) values for candidate orbitals:")
    for idx, s1 in zip(subspace, ths_lists, strict=False):
        logger.debug(_ORB_S1_FMT, idx, s1)

    logger.debug("MAX S(1): %.4f (orbital %d)\n", max_s1, subspace[np.argmax(soe1)])

    if fraction_type == "r":
        logger.info("Using relative threshold")
        ths_lists = ths_lists / max_s1
    else:
        logger.info("Using absolute threshold")
    sortkey = np.argsort(ths_lists)[::-1]
    ths_lists = ths_lists[sortkey]
    orbitals_index = subspace[sortkey]
    for idx, s1 in zip(orbitals_index, ths_lists, strict=False):
        logger.debug(_ORB_S1_FMT, idx, s1)

    stop_idx = None
    for k, ths in enumerate(ths_lists):
        if ths < fraction:
            stop_idx = k
            break
    if stop_idx is None:
        logger.info(
            "All %d orbitals are selected by the entropy threshold %g.",
            n_orbs,
            fraction,
        )
    else:
        ths_lists = ths_lists[:stop_idx]
        orbitals_index = orbitals_index[:stop_idx]
        logger.info(
            "Selected %d orbitals by entropy threshold %g (fraction type: %s).",
            len(orbitals_index),
            fraction,
            fraction_type,
        )

    return [(n_orb, ths_lists[idx]) for idx, n_orb in enumerate(orbitals_index)]


def select_by_entropy(
    occ_orbs: np.ndarray,
    virt_orbs: np.ndarray,
    soe1: np.ndarray | None,
    fraction: float = 0.1,
    fraction_type: Literal["r", "a"] = "r",
    ranking: Literal["global", "subset"] = "global",
) -> tuple[list[tuple[int, float]], list[tuple[int, float]]]:
    """Select active orbitals by single-orbital entanglement entropy.

    Parameters
    ----------
    occ_orbs : np.ndarray, shape (n_occ,)
        Indices of occupied orbitals.
    virt_orbs : np.ndarray, shape (n_virt,)
        Indices of virtual orbitals.
    soe1 : np.ndarray, shape (n_occ + n_virt,)
        Single-orbital entropies for all orbitals.
    fraction : float, optional
        Threshold on the entropy value. Interpretation depends on *fraction_type*.
    fraction_type : {'r', 'a'}, optional
        ``'r'`` normalises entropy values by the maximum before thresholding;
        ``'a'`` applies *fraction* as an absolute threshold.
    ranking : {'global', 'subset'}, optional
        ``'global'`` ranks all orbitals together; ``'subset'`` ranks occupied
        and virtual subspaces separately.

    Returns:
    -------
    selected_occ : list of tuple[int, float]
        Selected occupied orbitals with entropy values.
    selected_virt : list of tuple[int, float]
        Selected virtual orbitals with entropy values.
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
