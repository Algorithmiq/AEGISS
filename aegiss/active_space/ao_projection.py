"""Active space selection via AO projection onto a target set of atoms/AOs."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Literal, cast, overload

import numpy as np
from pyscf import gto

from aegiss.utils.orb_display import from_display_all, to_display, to_display_all
from aegiss.utils.orbital_indices import combine_candidate_entropies
from aegiss.utils.plotting import plot_ao_contribution

logger = logging.getLogger(__name__)

_RestrictedSpace = (
    list[int]
    | list[tuple[int, float]]
    | tuple[list[tuple[int, float]], list[tuple[int, float]]]
    | dict[str, tuple[list[tuple[int, float]], list[tuple[int, float]]]]
    | None
)
_AoContributions = dict[int, list[tuple[int, float]]]


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
            If a list of plain integers, these are treated as user input in the
            configured display base (0-based by default; 1-based when
            ``AEGISS_ORB_BASE=1``), converted to 0-based via
            :func:`aegiss.utils.orb_display.from_display_all`, and paired as
            ``(index, 0.0)``.
            If a flat list of ``(index, entropy)`` pairs, returned unchanged
            (these come from AEGISS internals and are already 0-based).
            If a ``(occ_list, virt_list)`` tuple (as returned by
            ``select_by_entropy``), the two lists are merged with
            ``combine_candidate_entropies`` (also already 0-based).

    Returns:
        space (list of tuple[int, float]): Orbital index / entropy pairs for the
            search space; indices are always 0-based.
    """
    if restricted_space is None:
        logger.info(
            "No restriction on orbital indices for selection. Using all orbitals."
        )
        # ``shape[-1]`` is the number of MOs for both a restricted ``(nao, nmo)``
        # and an unrestricted ``(2, nao, nmo)`` coefficient array.
        n_mo = np.asarray(mf.mo_coeff).shape[-1]
        return [(idx, 0.0) for idx in range(n_mo)]
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


def _resolve_thresholds(
    contribution: float | list[float],
    top_n: int | list[int] | None,
    n_labels: int,
) -> tuple[list[float], list[int] | None]:
    """Resolve the per-label contribution thresholds and top-N counts.

    Wraps :func:`_resolve_contribution` and :func:`_resolve_top_n` and applies the
    rule that *top_n*, when given, overrides *contribution* (the thresholds are
    zeroed so selection is purely top-N). Called once per spin channel so an
    unrestricted run can resolve different thresholds for alpha and beta.

    Args:
        contribution (float or list of float): Contribution threshold(s) for one
            channel, already extracted from any per-channel dict.
        top_n (int, list of int, or None): Top-N count(s), or ``None`` to
            threshold by *contribution* instead.
        n_labels (int): Number of AO labels being considered.

    Returns:
        resolved (tuple[list of float, list of int or None]): The per-label
            contribution thresholds and per-label top-N counts (``None`` when
            *top_n* is ``None``).
    """
    resolved_contribution = _resolve_contribution(contribution, n_labels)
    resolved_top_n: list[int] | None = None
    if top_n is not None:
        resolved_top_n = _resolve_top_n(top_n, n_labels)
        # Override contribution thresholds if top-N is specified.
        resolved_contribution = [0.0] * n_labels
    return resolved_contribution, resolved_top_n


def _channel_save_path(save_path: str | None, channel: str) -> str | None:
    """Insert a spin-channel tag before the extension of *save_path*.

    ``".../ao.png"`` becomes ``".../ao_alpha.png"`` so the alpha and beta
    heatmaps of an unrestricted run do not overwrite each other.

    Args:
        save_path (str or None): Base plot path, or ``None`` for no plotting.
        channel (str): Spin-channel tag to insert (``"alpha"`` / ``"beta"``).

    Returns:
        path (str or None): Tagged path, or ``None`` when *save_path* is ``None``.
    """
    if save_path is None:
        return None
    p = Path(save_path)
    return str(p.with_name(f"{p.stem}_{channel}{p.suffix}"))


def _run_ao_selection(
    mf: Any,
    ao_labels: list[str | list[str]],
    full_mo_coeff: np.ndarray,
    space: list[tuple[int, float]],
    resolved_contribution: list[float],
    resolved_top_n: list[int] | None,
    contribution_type: Literal["r", "a"],
    projected_basis: str | None,
    save_path: str | None,
    labels_idx: dict[int, str | list[str]],
) -> tuple[np.ndarray, _AoContributions]:
    """Run the AO-projection selection for a single 2D coefficient matrix.

    Loops over *ao_labels*, accumulates the selected orbital indices and the
    per-group AO contributions, and optionally writes the AO-contribution heatmap. This is the per-spin-channel
    core shared by the restricted and unrestricted paths of
    :func:`select_by_ao_projection`; the AO overlaps are spin-independent, so
    only *full_mo_coeff* differs between channels.

    Args:
        mf (pyscf.scf.hf.SCF): Mean-field object (used for the molecule/basis).
        ao_labels (list[str | list[str]]): AO label entries to screen against.
        full_mo_coeff (np.ndarray): 2D ``(nao, nmo)`` coefficient matrix for one
            spin channel (or the restricted MOs).
        space (list of tuple[int, float]): Orbital index/entropy search space.
        resolved_contribution (list of float): Per-label contribution thresholds.
        resolved_top_n (list of int or None): Per-label top-N counts, or ``None``
            to threshold by *resolved_contribution* instead.
        contribution_type ({'r', 'a'}): Relative or absolute contribution type.
        projected_basis (str or None): Projection basis; ``None`` uses the
            molecule's own basis.
        save_path (str or None): If provided, save the AO-contribution heatmap.
        labels_idx (dict[int, str | list[str]]): Loop-index to AO-label map for
            plot annotation.

    Returns:
        active_indices (np.ndarray of int): Sorted selected orbital indices.
        ao_contributions (dict[int, list[tuple[int, float]]]): Group index to
            ``(orbital_index, weight)`` pairs over the whole search space, in the
            format expected by :func:`~aegiss.utils.plotting.plot_ao_contribution`.
    """
    if not space:
        # An empty search space (e.g. a spin channel with no screened orbitals)
        # has nothing to project; return early before any ``max()`` over weights.
        logger.info("Empty search space for this channel; nothing selected.")
        return np.asarray([], dtype=int), {}

    selected_indices: set[int] = set()
    projected_total_weights: _AoContributions = {}
    for loop_idx, label in enumerate(ao_labels):
        selected_indices_for_label, projected_weights = select_by_label(
            label=label,
            projected_basis=projected_basis,
            mf=mf,
            full_mo_coeff=full_mo_coeff,
            space=space,
            contribution=resolved_contribution[loop_idx],
            contribution_type=contribution_type,
        )
        if resolved_top_n is not None:
            selected_indices_for_label = selected_indices_for_label[
                : resolved_top_n[loop_idx]
            ]
        selected_indices.update(orb_idx for orb_idx, _ in selected_indices_for_label)
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
    return np.asarray(sorted(selected_indices)), projected_total_weights


@overload
def select_by_ao_projection(
    mf: Any,
    ao_labels: list[str | list[str]],
    projected_basis: str | None = "minao",
    contribution_type: Literal["r", "a"] = "a",
    contribution: float | list[float] | dict[str, float | list[float]] = 0.0,
    top_n: int | list[int] | None = None,
    restricted_space: _RestrictedSpace = None,
    save_path: str | None = None,
    *,
    return_contributions: Literal[False] = False,
) -> np.ndarray | dict[str, np.ndarray]: ...


@overload
def select_by_ao_projection(
    mf: Any,
    ao_labels: list[str | list[str]],
    projected_basis: str | None = "minao",
    contribution_type: Literal["r", "a"] = "a",
    contribution: float | list[float] | dict[str, float | list[float]] = 0.0,
    top_n: int | list[int] | None = None,
    restricted_space: _RestrictedSpace = None,
    save_path: str | None = None,
    *,
    return_contributions: Literal[True],
) -> (
    tuple[np.ndarray, _AoContributions]
    | tuple[dict[str, np.ndarray], dict[str, _AoContributions]]
): ...


def select_by_ao_projection(
    mf: Any,
    ao_labels: list[str | list[str]],
    projected_basis: str | None = "minao",
    contribution_type: Literal["r", "a"] = "a",
    contribution: float | list[float] | dict[str, float | list[float]] = 0.0,
    top_n: int | list[int] | None = None,
    restricted_space: _RestrictedSpace = None,
    save_path: str | None = None,
    *,
    return_contributions: bool = False,
) -> (
    np.ndarray
    | dict[str, np.ndarray]
    | tuple[np.ndarray, _AoContributions]
    | tuple[dict[str, np.ndarray], dict[str, _AoContributions]]
):
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
        contribution (float, list of float, or dict, default 0.0): Threshold for
            selecting orbitals based on their contribution.
            A single float applies to all AO groups; a list specifies per-group
            thresholds and must match the length of *ao_labels*.
            For an unrestricted mean-field it may also be a spin-resolved
            ``{"alpha": ..., "beta": ...}`` mapping (each value itself a float or
            per-group list), applying a different threshold to each spin channel.
            A non-dict value is shared by both channels. A dict is rejected for a
            restricted mean-field.
        top_n (int or list of int, optional): Alternative to *contribution* thresholding. Selects the top N orbitals
            per AO group. A list specifies per-group counts.
        restricted_space (list of int, list of tuple, tuple of lists, dict, optional):
            Orbital indices to restrict selection to. Can be a plain list of
            integers (e.g. ``list(range(34, 131))``), a list of ``(index, entropy)``
            pairs, or a ``(occ_list, virt_list)`` tuple from ``select_by_entropy``.
            For an unrestricted mean-field it may also be a spin-resolved
            ``{"alpha": ..., "beta": ...}`` mapping (as produced by
            ``select_by_spin_entropy``), in which case each spin channel is
            restricted to its own space. A non-dict value is shared by both
            channels. ``None`` considers all orbitals.
        save_path (str or None, optional):
            If provided, saves a heatmap of AO contributions to the specified path.
            For an unrestricted mean-field the alpha/beta channels are written to
            separate ``*_alpha`` / ``*_beta`` files.
        return_contributions (bool, default False): If ``True``, also return the
            per-group AO contributions so they can be passed straight to
            :func:`~aegiss.utils.plotting.plot_ao_contribution` for custom plots.

    Returns:
        active_indices (np.ndarray of int, or dict[str, np.ndarray]):
            Selected orbital indices. For a restricted mean-field a single array
            is returned. For an unrestricted (UHF) mean-field -- detected from a
            3D ``mf.mo_coeff`` of shape ``(2, nao, nmo)`` -- the selection is
            always done per spin (the AO overlaps are spin-independent, only the
            MO coefficients differ) and the result is a split
            ``{"alpha": ..., "beta": ...}`` mapping. Whether the two channels
            share one search space (spin-summed *restricted_space*) or use their
            own (spin-resolved dict *restricted_space*), the alpha and beta
            selections are kept separate here: they are deliberately *not*
            reconciled into a common set of spatial orbitals, since an orbital
            index refers to a different MO in each spin channel.
        ao_contributions (dict, only if *return_contributions* is ``True``):
            Group index (position in *ao_labels*) to ``(orbital_index, weight)``
            pairs over the whole search space. For an unrestricted mean-field
            this is a ``{"alpha": ..., "beta": ...}`` mapping of such dicts.
    """
    logger.info("----------------AO Projection Screening for AEGISS----------------\n")
    logger.info("AO labels for projection: %s", ao_labels)

    logger.info("Contribution threshold: %s", contribution)
    logger.info("Contribution type: %s", contribution_type)

    if top_n is not None:
        logger.info("Top N orbitals to select: %s", top_n)

    if projected_basis is not None:
        logger.info("Using projected basis for AO projection: %s", projected_basis)
    else:
        logger.info("Using molecule for AO projection")

    labels_idx: dict[int, str | list[str]] = dict(enumerate(ao_labels))
    mo_coeff = np.asarray(mf.mo_coeff)

    if mo_coeff.ndim != 3:
        if isinstance(restricted_space, dict):
            msg = (
                "A spin-resolved restricted_space ({'alpha', 'beta'} dict) requires "
                "an unrestricted mean-field with a 3D mo_coeff; got a restricted one."
            )
            raise ValueError(msg)
        if isinstance(contribution, dict):
            msg = (
                "A spin-resolved contribution ({'alpha', 'beta'} dict) requires an "
                "unrestricted mean-field with a 3D mo_coeff; got a restricted one."
            )
            raise ValueError(msg)
        resolved_contribution, resolved_top_n = _resolve_thresholds(
            contribution, top_n, len(ao_labels)
        )
        indices, contributions = _run_ao_selection(
            mf=mf,
            ao_labels=ao_labels,
            full_mo_coeff=mo_coeff,
            space=_resolve_space(mf, restricted_space),
            resolved_contribution=resolved_contribution,
            resolved_top_n=resolved_top_n,
            contribution_type=contribution_type,
            projected_basis=projected_basis,
            save_path=save_path,
            labels_idx=labels_idx,
        )
        return (indices, contributions) if return_contributions else indices

    logger.info(
        "Unrestricted mean-field detected (mo_coeff shape %s); screening the "
        "alpha and beta channels per spin (%s search space).",
        mo_coeff.shape,
        "per-channel" if isinstance(restricted_space, dict) else "shared",
    )
    selection: dict[str, np.ndarray] = {}
    channel_contributions: dict[str, _AoContributions] = {}
    for spin, channel in enumerate(("alpha", "beta")):
        logger.info("=============== %s channel ===============", channel.upper())
        # A dict restricted_space carries its own window per spin channel
        # (spin-resolved entropy screening); anything else is shared by both.
        channel_space = (
            restricted_space[channel]
            if isinstance(restricted_space, dict)
            else restricted_space
        )
        # A dict contribution carries its own threshold(s) per spin channel;
        # anything else is shared by both.
        channel_contribution = (
            contribution[channel] if isinstance(contribution, dict) else contribution
        )
        resolved_contribution, resolved_top_n = _resolve_thresholds(
            channel_contribution, top_n, len(ao_labels)
        )
        selection[channel], channel_contributions[channel] = _run_ao_selection(
            mf=mf,
            ao_labels=ao_labels,
            full_mo_coeff=mo_coeff[spin],
            space=_resolve_space(mf, channel_space),
            resolved_contribution=resolved_contribution,
            resolved_top_n=resolved_top_n,
            contribution_type=contribution_type,
            projected_basis=projected_basis,
            save_path=_channel_save_path(save_path, channel),
            labels_idx=labels_idx,
        )
    return (selection, channel_contributions) if return_contributions else selection
