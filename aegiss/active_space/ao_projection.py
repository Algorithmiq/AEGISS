"""Active space selection via AO projection onto a target set of atoms/AOs."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
from pyscf import gto

from aegiss.utils.orb_display import from_display_all, to_display, to_display_all
from aegiss.utils.orbital_indices import combine_candidate_entropies
from aegiss.utils.plotting import plot_ao_contribution

logger = logging.getLogger(__name__)


def _compute_contribution(
    mf: Any,
    label: str | list[str],
    full_mo_coeff: np.ndarray,
    reduced_space_indices: list[tuple[int, float]],
    contribution: float,
    contribution_type: Literal["r", "a"],
    secondary_basis: str | None = "minao",
) -> tuple[list[tuple[int, float]], list[tuple[int, float]]]:
    """Compute the AO contribution of each orbital to a given label.

    Args:
        mf (pyscf.scf.hf.SCF): Mean-field object containing the molecule and MO coefficients.
        label (str or list of str): AO label(s) to compute contributions for.
        full_mo_coeff (np.ndarray): Coefficients of the active MOs.
        reduced_space_indices (list of tuple[int, float]): Orbital index/entropy pairs in the reduced space.
        contribution (float): Threshold for selecting orbitals.
        contribution_type ({'r', 'a'}): Relative or absolute contribution type.
        secondary_basis (str or None): Basis for projection; ``None`` uses the original molecule basis.

    Returns:
        selected (list of tuple[int, float]): Orbital index/weight pairs above the contribution threshold.
        all_contributions (list of tuple[int, float]): Orbital index/weight pairs for all orbitals, ordered by index,
            for potential further analysis or plotting.
    """
    mol2 = mf.mol  # original basis
    projection_mol = mol2.copy()  # extended basis
    if secondary_basis is not None:
        projection_mol.atom = mol2._atom
        projection_mol.unit = (
            "B"  # in Bohr because `mol._atom` returns the coordinates in Bohr
        )
        projection_mol.symmetry = False
        projection_mol.basis = secondary_basis
        projection_mol.build(False, False)  # noqa: FBT003
    else:
        logger.info(
            "No secondary basis provided. Using original molecule for projection."
        )
    baslst = projection_mol.search_ao_label(label)
    s21 = gto.intor_cross("int1e_ovlp", projection_mol, mol2)[baslst]
    if len(baslst) == 0:
        return [], []
    if s21.shape[1] != full_mo_coeff.shape[0]:
        msg = "Mismatch in number of orbitals between s21 and active_mo_coeff"
        raise ValueError(msg)

    reduced_mos = full_mo_coeff[:, np.array([idx for idx, _ in reduced_space_indices])]

    s22 = np.dot(s21, reduced_mos[:, :])
    raw_weights: list[tuple[int, float]] = []
    logger.info("Computing contributions for AO label(s): %s", label)
    for idx in range(len(s22[0, :])):
        weight = np.sum(np.abs(s22[:, idx]))
        orb = reduced_space_indices[idx][0]
        raw_weights.append((orb, weight))

    projected_weights: list[tuple[int, float]]
    if contribution_type == "r":
        max_contribution = max(weight for _, weight in raw_weights)
        if max_contribution == 0.0:
            projected_weights = [(i, 0.0) for i, _ in raw_weights]
        else:
            projected_weights = [(i, w / max_contribution) for i, w in raw_weights]
    else:
        projected_weights = raw_weights
    sorted_weights = sorted(projected_weights, key=lambda x: x[1], reverse=True)

    first_below = False
    logger.info("\t%s", "-" * 10 + "above" + "-" * 10)
    for idx, tmp in sorted_weights:
        if tmp < contribution and not first_below:
            logger.info("\t%s", "-" * 10 + "below" + "-" * 10)
            first_below = True
        logger.info("\tOrbital %d: contribution = %.8f", to_display(idx), tmp)

    return [
        (idx, tmp) for idx, tmp in sorted_weights if tmp >= contribution
    ], projected_weights


def select_by_label(
    label: str | list[str],
    mf: Any,
    full_mo_coeff: np.ndarray,
    space: list[tuple[int, float]],
    contribution: float,
    projected_basis: str | None,
    contribution_type: Literal["r", "a"],
) -> tuple[list[tuple[int, float]], list[tuple[int, float]]]:
    """Select orbitals based on their contribution to a specific AO label.

    Args:
        label (str or list of str): AO label(s) to compute contributions for. Each label should be in the
            format ``"Element.Index AtomicOrbital"``, e.g. ``"O.1 1s"`` for the
            second oxygen atom. If a list is provided, contributions are computed
            for each label separately and combined.
        mf (pyscf.scf.hf.SCF): Mean-field object containing the molecule and MO coefficients.
        full_mo_coeff (np.ndarray): Coefficients of the full MOs.
        space (list of tuple[int, float]): Orbital index/entropy pairs to restrict the selection to.
        contribution (float): Threshold for selecting orbitals based on their contribution.
        projected_basis (str or None): Basis set used for projection; ``None`` uses the original basis.
        contribution_type ({'r', 'a'}): Relative (``'r'``) or absolute (``'a'``) contribution type.

    Returns:
        selected_indices (list of tuple[int, float]): Orbital index/weight pairs selected based on their contribution.
        projected_weights (list of tuple[int, float]): Orbital index/weight pairs for all orbitals, ordered by index,
            for potential further analysis or plotting.
    """
    return _compute_contribution(
        mf,
        label,
        full_mo_coeff,
        space,
        contribution,
        contribution_type,
        projected_basis,
    )


def _resolve_space(
    mf: Any,
    restricted_space: list[int]
    | list[tuple[int, float]]
    | tuple[list[tuple[int, float]], list[tuple[int, float]]]
    | None,
) -> list[tuple[int, float]]:
    """Resolve the orbital search space from the ``restricted_space`` argument.

    Args:
        mf (pyscf.scf.hf.SCF): Converged mean-field object; used to determine the total number of MOs
            when *restricted_space* is ``None``.
        restricted_space (list of int, list of tuple, tuple of lists, or None):
            If ``None``, all MOs are returned with a placeholder entropy of 0.0.
            If a list of plain integers, converted to ``(index, 0.0)`` pairs.
            If a flat list of ``(index, entropy)`` pairs, returned unchanged.
            If a ``(occ_list, virt_list)`` tuple (as returned by
            ``select_by_entropy``), the two lists are merged with
            ``combine_candidate_entropies``.

    Returns:
        space (list of tuple[int, float]): Orbital index / entropy pairs for the search space.
    """
    if restricted_space is None:
        logger.info(
            "No restriction on orbital indices for selection. Using all orbitals."
        )
        return [(idx, 0.0) for idx in range(mf.mo_coeff.shape[1])]
    if isinstance(restricted_space, tuple):
        space = combine_candidate_entropies(*restricted_space)
        logger.info(
            "Combining candidate orbitals from different groups: %s",
            to_display_all(int(idx[0]) for idx in space),
        )
        return space
    if restricted_space and not isinstance(restricted_space[0], tuple):
        # Plain integer indices are user input, expressed in the display base;
        # convert them back to the internal 0-based representation. Pair-lists
        # and (occ, virt) tuples come from AEGISS internals and stay 0-based.
        return [(i, 0.0) for i in from_display_all(cast("list[int]", restricted_space))]
    return cast("list[tuple[int, float]]", restricted_space)


def _resolve_contribution(
    contribution: float | list[float],
    n_labels: int,
) -> list[float]:
    """Helper function to resolve contribution thresholds for AO projection selection.

    Args:
        contribution (float or list of float): Contribution threshold(s) for selecting orbitals.
            A single float applies to all labels; a list specifies per-label thresholds.
        n_labels (int): Number of AO labels being considered for selection.

    Returns:
        resolved_contribution (list of float): List of contribution thresholds for each AO label,
            resolved based on the input parameters.
    """
    if isinstance(contribution, list):
        logger.info(
            "\nSelecting orbitals based on contribution thresholds per AO label: %s",
            contribution,
        )
        if len(contribution) != n_labels:
            logger.info(
                "Warning: Length of contribution list does not match number of AO labels. "
                "Using the first contribution value for all labels."
            )
            return [contribution[0]] * n_labels
        return contribution
    logger.info(
        "\nSelecting orbitals based on a single contribution threshold for all AO labels: %s",
        contribution,
    )
    return [contribution] * n_labels


def _resolve_top_n(
    top_n: int | list[int],
    n_labels: int,
) -> list[int]:
    """Helper function to resolve the number of top orbitals to select per AO label.

    Args:
        top_n (int, list of int): Number of top orbitals to select per AO label. A single int applies to all labels;
            a list specifies per-label counts.
        n_labels (int): Number of AO labels being considered for selection.

    Returns:
        resolved_top_n (list of int): List of top-N counts for each AO label, one entry per label.
    """
    if isinstance(top_n, list):
        logger.info(
            "\nSelecting orbitals based on top-N counts per AO label: %s",
            top_n,
        )
        if len(top_n) != n_labels:
            logger.info(
                "Warning: Length of top-N list does not match number of AO labels. "
                "Using the first top-N value for all labels."
            )
            return [top_n[0]] * n_labels
        return top_n
    logger.info(
        "\nSelecting orbitals based on a single top-N count for all AO labels: %s",
        top_n,
    )
    return [top_n] * n_labels


def select_by_ao_projection(
    mf: Any,
    ao_labels: list[str | list[str]],
    projected_basis: str | None = "minao",
    contribution_type: Literal["r", "a"] = "a",
    contribution: float | list[float] = 0.0,
    top_n: int | list[int] | None = None,
    restricted_space: list[int]
    | list[tuple[int, float]]
    | tuple[list[tuple[int, float]], list[tuple[int, float]]]
    | None = None,
    save_path: str | None = None,
) -> np.ndarray:
    """Select active orbitals by their projection onto a set of target AOs.

    Args:
        mf (pyscf.scf.hf.SCF or pyscf.dft.rks.KohnShamDFT):
            Mean-field object containing the molecular orbitals and occupation numbers.
        ao_labels (list[str | list[str]]): List of AO label entries. Each entry is either a single label string
            (e.g. ``"Fe 3d"``) or a list of strings treated as one group
            (e.g. ``["Fe 3d", "Fe 4d"]``). Entries can be freely mixed, e.g.
            ``["C 2pz", "Fe 3d", ["Fe 3d", "Fe 4d"]]``.
        projected_basis (str or None, default 'minao'):
            Basis set to use for the projection. ``None`` uses the original basis.
            A minimal basis (e.g. ``'minao'``) helps focus the selection on
            chemically relevant orbitals and reduces noise from diffuse functions.
        contribution_type ({'r', 'a'}): Relative (``'r'``) or absolute (``'a'``) contribution. Required when
            *contribution* is specified.
        contribution (float or list of float, default 0.0): Threshold for selecting
            orbitals based on their contribution.
            A single float applies to all AO groups; a list specifies per-group
            thresholds and must match the length of *ao_labels*.
        top_n (int or list of int, optional): Alternative to *contribution* thresholding. Selects the top N orbitals
            per AO group. A list specifies per-group counts.
        restricted_space (list of int, list of tuple, or tuple of lists, optional):
            Orbital indices to restrict selection to. Can be a plain list of
            integers (e.g. ``list(range(34, 131))``), a list of ``(index, entropy)``
            pairs, or a ``(occ_list, virt_list)`` tuple from ``select_by_entropy``.
            ``None`` considers all orbitals.
        save_path (str or None, optional):
            If provided, saves a heatmap of AO contributions to the specified path.

    Returns:
        active_indices (np.ndarray of int): Array of orbital indices selected for the active space.
    """
    logger.info("----------------AO Projection Screening for AEGISS----------------\n")
    logger.info("AO labels for projection: %s", ao_labels)

    logger.info("Contribution threshold: %s", contribution)
    logger.info("Contribution type: %s", contribution_type)

    if top_n is not None:
        logger.info("Top N orbitals to select: %s", top_n)

    space = _resolve_space(mf, restricted_space)
    resolved_contribution = _resolve_contribution(contribution, len(ao_labels))
    if top_n is not None:
        resolved_top_n = _resolve_top_n(top_n, len(ao_labels))
        resolved_contribution = [0.0] * len(
            ao_labels
        )  # Override contribution thresholds if top-N is specified

    if projected_basis is not None:
        logger.info("Using projected basis for AO projection: %s", projected_basis)
    else:
        logger.info("Using molecule for AO projection")

    selected_indices: set[int] = set()
    projected_total_weights: dict[int, list[tuple[int, float]]] = {}
    labels_idx: dict[int, str | list[str]] = dict(enumerate(ao_labels))
    for loop_idx, label in enumerate(ao_labels):
        selected_indices_for_label, projected_weights = select_by_label(
            label=label,
            projected_basis=projected_basis,
            mf=mf,
            full_mo_coeff=mf.mo_coeff,
            space=space,
            contribution=resolved_contribution[loop_idx],
            contribution_type=contribution_type,
        )
        if top_n is not None:
            selected_indices_for_label = selected_indices_for_label[
                : resolved_top_n[loop_idx]
            ]
        selected_indices.update(orb_idx for orb_idx, _ in selected_indices_for_label)
        if save_path is not None:
            projected_total_weights[loop_idx] = projected_weights

    if save_path is not None and projected_total_weights:
        _p = Path(save_path)
        plot_ao_contribution(
            ao_contributions=projected_total_weights,
            folder=str(_p.parent),
            filename=_p.name,
            ao_labels=labels_idx,
            mo_labels=[orb_idx for orb_idx, _ in space],
        )

    logger.info("Final selected orbital indices:")
    for idx in sorted(selected_indices):
        logger.info("\tOrbital %d", to_display(idx))
    # Return values stay 0-based so results round-trip safely into further logic.
    return np.asarray(sorted(selected_indices))
