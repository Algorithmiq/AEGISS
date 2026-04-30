"""Active space selection via single-orbital entropy screening."""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np

logger = logging.getLogger(__name__)
_ORB_S1_FMT = "\tOrbital %d: S(1) = %.4f"


def _global_entropy_ranking(
    soe1: np.ndarray,
    fraction_type: Literal["r", "a"],
    fraction: float,
    n_orbs: int,
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
    n_orbs : int
        Total number of orbitals.
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
    max_s1 = np.max(soe1)
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
    n_orbs = len(soe1)

    if ranking == "global":
        logger.info("Using global ranking of orbitals by entropy values.")
        selected_occ, selected_virt = _global_entropy_ranking(
            soe1, fraction_type, fraction, n_orbs, occ_orbs, virt_orbs
        )
    else:
        logger.info(
            "Using relative ranking of orbitals by entropy values within the subspace."
        )
        soe1_occ = np.array([soe1[idx] for idx in range(len(occ_orbs))])

        off_set = len(occ_orbs)
        soe1_virt = np.array([soe1[off_set + idx] for idx in range(len(virt_orbs))])

        selected_occ = _relative_entropy_ranking(
            soe1_occ, fraction_type, fraction, occ_orbs
        )
        selected_virt = _relative_entropy_ranking(
            soe1_virt, fraction_type, fraction, virt_orbs
        )

    for i in selected_occ:
        logger.info("\tSelected occupied orbital %d: S(1) = %.4f", i[0], i[1])
    for i in selected_virt:
        logger.info("\tSelected virtual orbital %d:  S(1) = %.4f", i[0], i[1])

    return selected_occ, selected_virt
